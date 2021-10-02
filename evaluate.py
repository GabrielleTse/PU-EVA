import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))

# import evaluate_pugan
import importlib
import numpy as np
import tensorflow as tf
from glob import glob
import re
import csv
from collections import OrderedDict

from tqdm import tqdm
from utils import pc_util
from utils.pc_util import load, save_ply_property
from utils.tf_util import normalize_point_cloud

import test
from tf_ops.sampling.tf_sampling import farthest_point_sample, gather_point
from matplotlib import pyplot as plt
from configs import FLAGS
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


MODEL_GEN = importlib.import_module('model') # import network module
BATCH_SIZE = 1
UP_RATIO = 6
FLAGS.batch_size = 1


def prediction_whole_model(restore_model_path,test_dir,out_folder,POINT_NUM=2048 ,show=False):

    TEST_FILES = [os.path.join(test_dir,file) for file in os.listdir(test_dir)]
    pointclouds_ipt = tf.placeholder(tf.float32, shape=(BATCH_SIZE, FLAGS.patch_num_point, 3))
    pred_val = MODEL_GEN.get_gen_model(pointclouds_ipt,UP_RATIO,FLAGS, is_training=False, bn_decay=None)
    
    saver = tf.train.Saver()
    ops ={
        'pred_pc':pred_val,
        'inputs':pointclouds_ipt
         }

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    with tf.Session(config=config) as sess:
        saver.restore(sess, restore_model_path)
        
        for point_path in TEST_FILES:
            pc = pc_util.load(point_path)[:, :]
            pc, centroid, furthest_distance = pc_util. normalize_point_cloud(pc)
            _,pred_list = pc_prediction(sess,ops,pc)
            
            pred_pc = np.concatenate(pred_list, axis=0)
            pred_pc = np.reshape(pred_pc, [-1, 3])
            pred_pc[:,:3] = (pred_pc[:,:3] * furthest_distance) + centroid
            idx = farthest_point_sample(POINT_NUM*4, pred_pc[np.newaxis, ...]).eval()[0]
            pred_pc = pred_pc[idx, 0:3]
            path = os.path.join(out_folder, point_path.split('/')[-1][:-4] + '.xyz')
            np.savetxt(path,pred_pc,fmt='%.6f')


def pc_prediction(sess,ops,pc):
    
    points = tf.convert_to_tensor(np.expand_dims(pc[:,:3], axis=0), dtype=tf.float32)
    seed1_num = int(pc.shape[0] / FLAGS.patch_num_point * FLAGS.patch_num_ratio)  # 24
    ## FPS sampling
    seed = farthest_point_sample(seed1_num, points).eval()[0] 
    seed_list = seed[:seed1_num]
    input_list = []
    up_point_list = []
    patches = pc_util.extract_knn_patch(pc[np.asarray(seed_list), :], pc, FLAGS.patch_num_point)

    for point in patches:
        up_point = patch_prediction(sess,ops,point.copy())
        up_point = np.squeeze(up_point, axis=0)
        input_list.append(point)
        up_point_list.append(up_point)

    return input_list, up_point_list

def patch_prediction(sess,ops, patch_point):
  # normalize the point clouds
  patch_point, centroid, furthest_distance = pc_util.normalize_point_cloud(patch_point)
  patch_point = np.expand_dims(patch_point, axis=0)
  pred = sess.run([ops['pred_pc']], feed_dict={ops['inputs']: patch_point})

  pred = np.squeeze(pred, axis=0)
  pred = centroid+pred*furthest_distance
  return pred


if __name__ == '__main__':

    log_dir = "./logs/2021-09-26-09-26-07/"
    for out_num in [1024,8192,16384]:
        test_dir = 'data/test_xyz/{}_{}_input/'.format(out_num,out_num//4)
        out_folder=  './data/pred_xyz/'+str(out_num)

        restore_model_paths = sorted(glob.glob(os.path.join(log_dir,"model*.meta")))
        restore_model_paths = [x.split(".")[0] for x in restore_model_paths]
        for model_path in restore_model_paths:
            prediction_whole_model(model_path,test_dir,out_folder,POINT_NUM= out_num//4,show=False)
            test.eval(model_path,'./data/test_xyz/{}/'.format(out_num),out_folder,out_num)
            tf.reset_default_graph()
  