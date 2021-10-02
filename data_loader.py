# -*- coding: utf-8 -*-
# @Time        : 16/1/2019 5:11 PM
# @Description :
# @Author      : li rui hui
# @Email       : ruihuili@gmail.com
# @File        : data_loader.py

import numpy as np
import h5py
import queue
import threading
from utils import point_operation
import random
import os
from utils import pc_util

def normalize_point_cloud(input):
    if len(input.shape)==2:
        axis = 0
    elif len(input.shape)==3:
        axis = 1
    centroid = np.mean(input, axis=axis, keepdims=True)
    input = input - centroid
    furthest_distance = np.amax(np.sqrt(np.sum(input ** 2, axis=-1)),axis=axis,keepdims=True)
    input = input / furthest_distance
    return input, centroid,furthest_distance

def batch_sampling(input_data,num):
    B,N,C = input_data.shape
    out_data = np.zeros([B,num,C])
    for i in range(B):
        idx = np.arange(N)
        np.random.shuffle(idx)
        idx = idx[:num]
        out_data[i,...] = input_data[i,idx]
    return out_data

def load_h5_data(h5_filename='', opts=None, skip_rate = 1, use_randominput=True):
    num_point = opts.num_point
    num_4X_point = int(opts.num_point*4)
    num_out_point = int(opts.num_point*4)

    print("h5_filename : ",h5_filename)
    if use_randominput:
        print("use randominput, input h5 file is:", h5_filename)
        f = h5py.File(h5_filename,'r')
        input = f['poisson_%d'%num_4X_point][:]
        gt = f['poisson_%d'%num_out_point][:]
    else:
        print("Do not randominput, input h5 file is:", h5_filename)
        f = h5py.File(h5_filename,'r')
        input = f['poisson_%d' % num_point][:]
        gt = f['poisson_%d' % num_out_point][:]

    assert len(input) == len(gt)
    # print("Normalization the data")
    data_radius = np.ones(shape=(len(input)))
    centroid = np.mean(gt[:, :, 0:3], axis=1, keepdims=True)
    gt[:, :, 0:3] = gt[:, :, 0:3] - centroid
    furthest_distance = np.amax(np.sqrt(np.sum(gt[:, :, 0:3] ** 2, axis=-1)), axis=1, keepdims=True)
    gt[:, :, 0:3] = gt[:, :, 0:3] / np.expand_dims(furthest_distance, axis=-1)
    input[:, :, 0:3] = input[:, :, 0:3] - centroid
    input[:, :, 0:3] = input[:, :, 0:3] / np.expand_dims(furthest_distance, axis=-1)

    # print("Normalization the color")
    # input[:,:,3:] = ((input[:,:,3:]/255)-0.5)*2

    input = input[::skip_rate]
    gt = gt[::skip_rate]
    data_radius = data_radius[::skip_rate]
    # print("total %d samples" % (len(input)))
    return input, gt, data_radius


class Fetcher(threading.Thread):
    def __init__(self, opts,h5_file):
        super(Fetcher,self).__init__()
        self.queue = queue.Queue(50)
        self.stopped = False
        self.opts = opts
        self.use_random_input = self.opts.use_non_uniform
        self.input_data, self.gt_data, self.radius_data = load_h5_data(h5_file,opts=self.opts,use_randominput=self.use_random_input)
        self.batch_size = self.opts.batch_size
        self.sample_cnt = self.input_data.shape[0]
        self.patch_num_point = self.opts.patch_num_point
        self.num_batches = self.sample_cnt//self.batch_size
        print ("NUM_BATCH is %s"%(self.num_batches))

    def run(self):
        while not self.stopped:
            idx = np.arange(self.sample_cnt)
            np.random.shuffle(idx)
            self.input_data = self.input_data[idx, ...]
            self.gt_data = self.gt_data[idx, ...]
            self.radius_data = self.radius_data[idx, ...]

            for batch_idx in range(self.num_batches):
                if self.stopped:
                    return None
                start_idx = batch_idx * self.batch_size
                end_idx = (batch_idx + 1) * self.batch_size
                batch_input_data = self.input_data[start_idx:end_idx, :, :].copy()
                batch_data_gt = self.gt_data[start_idx:end_idx, :, :].copy()
                radius = self.radius_data[start_idx:end_idx].copy()

                if self.use_random_input:
                    new_batch_input = np.zeros((self.batch_size, self.patch_num_point, batch_input_data.shape[2]))
                    new_batch_gt = np.zeros((self.batch_size, self.patch_num_point*4, batch_input_data.shape[2]))
                    for i in range(self.batch_size):
                        idx = point_operation.nonuniform_sampling(self.input_data.shape[1], sample_num=self.patch_num_point)
                        new_batch_input[i, ...] = batch_input_data[i][idx]

                    #     gt_idx = point_operation.nonuniform_sampling(self.input_data.shape[1], sample_num=self.patch_num_point*4)
                    #     new_batch_gt[i, ...] = batch_data_gt[i][gt_idx]
                    # batch_data_gt = new_batch_gt

                    batch_input_data = new_batch_input

                if self.opts.augment:
                    batch_input_data = point_operation.jitter_perturbation_point_cloud(batch_input_data, sigma=self.opts.jitter_sigma, clip=self.opts.jitter_max)
                    batch_input_data, batch_data_gt = point_operation.rotate_point_cloud_and_gt(batch_input_data, batch_data_gt)
                    batch_input_data, batch_data_gt, scales = point_operation.random_scale_point_cloud_and_gt(batch_input_data,
                                                                                              batch_data_gt,
                                                                                              scale_low=0.8,
                                                                                              scale_high=1.2)
                    radius = radius * scales

                self.queue.put((batch_input_data[:,:,:], batch_data_gt[:,:,:],radius))
        return None

    def fetch(self):
        if self.stopped:
            return None
        return self.queue.get()

    def shutdown(self):
        self.stopped = True
        print ("Shutdown .....")
        while not self.queue.empty():
            self.queue.get()
        print ("Remove all queue data")



class data_loader_dtu:
   
    def __init__(self,root_dir,opts):

        self.files = os.listdir(root_dir)
        random.shuffle(self.files)
        self.root_dir = root_dir
        self.batch_size = opts.batch_size


        self.batch_num = len(self.files)//self.batch_size
        
        self.opts = opts
        self.use_random_input = opts.use_random_input
        self.patch_num_point = opts.patch_num_point

        
        self.input_npoint = None
        
       

    def get_data(self,idx):
        assert idx<self.batch_num
        batch_files = [os.path.join(self.root_dir,file) for file in self.files[idx*self.batch_size:(idx+1)*self.batch_size]]

        x = pc_util.load(batch_files[0])[:, :]
        self.input_npoint = x.shape[0]

        datas = np.zeros([self.batch_size,self.input_npoint,6])

        for k,file in enumerate(batch_files):
          
            pc = pc_util.load(file)[:, :]
            datas[k,:,:]=pc[:,:]
        input = datas
        gt = np.copy(datas)
        input, gt, data_radius = self.norm_datas(input,gt,skip_rate=1)
        batch_input_data,batch_data_gt,radius=self.aug_data(input,gt,data_radius)

       
        return  batch_input_data,batch_data_gt,radius,batch_files[0]


    def aug_data(self,batch_input_data,batch_data_gt,radius):

        # start_idx = batch_idx * self.batch_size
        # end_idx = (batch_idx + 1) * self.batch_size
        # batch_input_data = self.input_data[start_idx:end_idx, :, :].copy()
        # batch_data_gt = self.gt_data[start_idx:end_idx, :, :].copy()
        # radius = self.radius_data[start_idx:end_idx].copy()

        if self.use_random_input:
           
            new_batch_input = np.zeros((self.batch_size, self.patch_num_point, batch_input_data.shape[2]))
            new_batch_gt = np.zeros((self.batch_size, self.patch_num_point*6, batch_input_data.shape[2]))
            for i in range(self.batch_size):
                idx = point_operation.nonuniform_sampling(self.input_npoint, sample_num=self.patch_num_point)
                new_batch_input[i, ...] = batch_input_data[i][idx]

                gt_idx = np.random.choice(self.input_npoint,self.patch_num_point*6,replace=False)

                new_batch_gt[i, ...] = batch_data_gt[i][gt_idx]
            
            batch_data_gt = new_batch_gt

            batch_input_data = new_batch_input

        if self.opts.augment:
            batch_input_data = point_operation.jitter_perturbation_point_cloud(batch_input_data, sigma=self.opts.jitter_sigma, clip=self.opts.jitter_max)
            batch_input_data, batch_data_gt = point_operation.rotate_point_cloud_and_gt(batch_input_data, batch_data_gt)
            batch_input_data, batch_data_gt, scales = point_operation.random_scale_point_cloud_and_gt(batch_input_data,
                                                                                        batch_data_gt,
                                                                                        scale_low=0.8,
                                                                                        scale_high=1.2)
            radius = radius * scales


        return batch_input_data,batch_data_gt,radius

    def norm_datas(self,input,gt,skip_rate=1):

      
        data_radius = np.ones(shape=(len(input)))
        centroid = np.mean(gt[:, :, 0:3], axis=1, keepdims=True)
        gt[:, :, 0:3] = gt[:, :, 0:3] - centroid
        furthest_distance = np.amax(np.sqrt(np.sum(gt[:, :, 0:3] ** 2, axis=-1)), axis=1, keepdims=True)
        gt[:, :, 0:3] = gt[:, :, 0:3] / np.expand_dims(furthest_distance, axis=-1)
        input[:, :, 0:3] = input[:, :, 0:3] - centroid
        input[:, :, 0:3] = input[:, :, 0:3] / np.expand_dims(furthest_distance, axis=-1)

      
        input[:,:,3:] = ((input[:,:,3:]/255)-0.5)*2
        gt[:,:,3:] = ((gt[:,:,3:]/255 )-0.5)*2

        input = input[::skip_rate]
        gt = gt[::skip_rate]
        data_radius = data_radius[::skip_rate]
      
        return input, gt, data_radius
