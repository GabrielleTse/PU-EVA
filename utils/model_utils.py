# -*- coding: utf-8 -*-
import os
import tensorflow as tf
from tf_ops.emd import tf_auctionmatch
from tf_ops.CD import tf_nndistance
from tf_ops.sampling import tf_sampling
from tf_ops.grouping.tf_grouping import query_ball_point,group_point,knn_point
from tf_ops.sampling.tf_sampling import gather_point, farthest_point_sample
import numpy as np
import math

def pre_load_checkpoint(checkpoint_dir):
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        # print(" [*] Reading checkpoint from {}".format(ckpt.model_checkpoint_path))
        epoch_step = int(os.path.basename(ckpt.model_checkpoint_path).split('-')[1])
        return epoch_step,ckpt.model_checkpoint_path
    else:
        return 0,None


def get_repulsion_loss4(pred, nsample=20, radius=0.07):
    # pred: (batch_size, npoint,3)
    idx, pts_cnt = query_ball_point(radius, nsample, pred, pred)
    tf.summary.histogram('smooth/unque_index', pts_cnt)

    grouped_pred = group_point(pred, idx)  # (batch_size, npoint, nsample, 3)
    grouped_pred -= tf.expand_dims(pred, 2)

    ##get the uniform loss
    h = 0.03
    dist_square = tf.reduce_sum(grouped_pred ** 2, axis=-1)
    dist_square, idx = tf.nn.top_k(-dist_square, 5)
    dist_square = -dist_square[:, :, 1:]  # remove the first one
    dist_square = tf.maximum(1e-12,dist_square)
    dist = tf.sqrt(dist_square)
    weight = tf.exp(-dist_square/h**2)
    uniform_loss = tf.reduce_mean(radius-dist*weight)
    return uniform_loss



def get_repulsion_loss(pred, nsample=20, radius=0.07, knn=False, use_l1=False, h=0.001):

        # pred: (batch_size, npoint,3)
    if knn:
        _, idx = knn_point(nsample, pred, pred)
        pts_cnt = tf.constant(nsample, shape=(30, 1024))
    else:
        idx, pts_cnt = query_ball_point(radius, nsample, pred, pred)
    tf.summary.histogram('smooth/unque_index', pts_cnt)

    grouped_pred = group_point(pred, idx)  # (batch_size, npoint, nsample, 3)
    grouped_pred -= tf.expand_dims(pred, 2)

    # get the uniform loss
    if use_l1:
        dists = tf.reduce_sum(tf.abs(grouped_pred), axis=-1)
    else:
        dists = tf.reduce_sum(grouped_pred ** 2, axis=-1)

    val, idx = tf.nn.top_k(-dists, 6)
    val = val[:, :, 1:]  # remove the first one

    if use_l1:
        h = np.sqrt(h)*2
    print(("h is ", h))

    val = tf.maximum(0.0, h + val)  # dd/np.sqrt(n)
    repulsion_loss = tf.reduce_mean(val)
    return repulsion_loss


#simplfied version, faster
def get_uniform_loss(pcd, percentages=[0.004,0.006,0.008,0.010,0.012], radius=1.0):

    B,N,C = pcd.get_shape().as_list()
    # print(batch_size,num_points)
    # indices = tf.convert_to_tensor(generate_indices(B,N,k=3)) #(B*N*(r-1),3)
    # pcd = tf.gather_nd(pc,indices=indices) #
    # pcd = tf.reshape(pcd,[B,N,3])


    npoint = int(N * 0.05)
    loss=[]
    for p in percentages:
        nsample = int(N*p)
        r = math.sqrt(p*radius)
        disk_area = math.pi *(radius ** 2) * p/nsample
        #print(npoint,nsample)
        new_xyz = gather_point(pcd, farthest_point_sample(npoint, pcd))  # (batch_size, npoint, 3)
        idx, pts_cnt = query_ball_point(r, nsample, pcd, new_xyz)#(batch_size, npoint, nsample)

        #expect_len =  tf.sqrt(2*disk_area/1.732)#using hexagon
        expect_len = tf.sqrt(disk_area+1e-8)  # using square

        grouped_pcd = group_point(pcd, idx)
        grouped_pcd = tf.concat(tf.unstack(grouped_pcd, axis=1), axis=0)

        var, _ = knn_point(2, grouped_pcd, grouped_pcd)
        uniform_dis = -var[:, :, 1:]
        uniform_dis = tf.sqrt(tf.abs(uniform_dis+1e-8))
        uniform_dis = tf.reduce_mean(uniform_dis,axis=[-1])
        uniform_dis = tf.square(uniform_dis - expect_len) / (expect_len + 1e-8)
        uniform_dis = tf.reshape(uniform_dis, [-1])

        mean, variance = tf.nn.moments(uniform_dis, axes=0)
        mean = mean*math.pow(p*100,2)
        #nothing 4
        loss.append(mean)

    return tf.add_n(loss)/len(percentages)



def generate_indices(batch_size,numpoint,k,start=0):
    N = numpoint
    B = batch_size
    K =k
    x = range(N)
    y = range(B)
    z = range(start,K+start)
    X,Y,Z = np.meshgrid(x,y,z)
    W = np.stack([Y,X,Z],axis=3)
    indices = W.reshape(-1,3)
    # print(indices)

    return indices


def get_emd_loss(pred, gt, radius):
    """ pred: BxNxC,
        label: BxN, """
    batch_size = pred.get_shape()[0].value
    matchl_out, matchr_out = tf_auctionmatch.auction_match(pred, gt)
    matched_out = tf_sampling.gather_point(gt, matchl_out)
    dist = tf.reshape((pred - matched_out) ** 2, shape=(batch_size, -1))
    dist = tf.reduce_mean(dist, axis=1, keep_dims=True)
    dist_norm = dist / radius

    emd_loss = tf.reduce_mean(dist_norm)
    return emd_loss,matchl_out

def get_cd_loss_test(pred, gt):
    """ pred: BxNxC,
        label: BxN, """
    dists_forward, _, dists_backward, _ = tf_nndistance.nn_distance(gt, pred)
    #dists_forward is for each element in gt, the cloest distance to this element
    cd_loss= tf.reduce_sum(dists_forward) + tf.reduce_mean(dists_backward)
    return cd_loss


def get_cd_loss(pred, gt, forward_weight=1.0, threshold=100.0):
    """
    pred: BxNxC,
    gt: BxNxC,
    forward_weight: relative weight for forward_distance
    threshold: 2.0
    """

    with tf.name_scope("cd_loss"):
        dists_forward, idx1, dists_backward, idx2 = tf_nndistance.nn_distance(gt, pred)
        '''output: idx1:  (batch_size,#point_1)   nearest neighbor from first to second'''
        if threshold is not None:
            forward_threshold = tf.reduce_mean(dists_forward, keep_dims=True, axis=1) * threshold
            backward_threshold = tf.reduce_mean(dists_backward, keep_dims=True, axis=1) * threshold
            # only care about distance within threshold (ignore strong outliers)
            dists_forward = tf.where(dists_forward < forward_threshold, dists_forward, tf.zeros_like(dists_forward))
            dists_backward = tf.where(dists_backward < backward_threshold, dists_backward, tf.zeros_like(dists_backward))
        # dists_forward is for each element in gt, the closest distance to this element
        dists_forward = tf.reduce_mean(dists_forward, axis=1)
        dists_backward = tf.reduce_mean(dists_backward, axis=1)
        CD_dist = forward_weight * dists_forward + dists_backward
        # CD_dist_norm = CD_dist/radius
        cd_loss = tf.reduce_mean(CD_dist)
        return cd_loss


def color_L1_loss(pred_s,gt_s,idx1,idx2,forward_weight=1.0,threshold = 100.0):
    '''output: idx1:  (batch_size,#point_1)   nearest neighbor from first to second'''

    B1,N1,C = pred_s.get_shape().as_list()
    # print(batch_size,num_points)
    indices_pred = tf.convert_to_tensor(generate_indices(B1,N1,k=3,start=3)) #(BN3,3)
    pred = tf.gather_nd(pred_s,indices=indices_pred) #
    pred_color = tf.reshape(pred,[B1,N1,3])
    B2,N2,C = gt_s.get_shape().as_list()
    # print(batch_size,num_points)
    indices_gt = tf.convert_to_tensor(generate_indices(B2,N2,k=3,start=3)) #
    gt = tf.gather_nd(gt_s,indices=indices_gt) #
    gt_color = tf.reshape(gt,[B2,N2,3])
    forward = gather_point(gt_color,idx1)
    dists_forward = tf.sqrt(tf.reduce_sum((pred_color-forward)**2,axis=-1))
    backward = gather_point(pred_color,idx2)
    dists_backward = tf.sqrt(tf.reduce_sum((gt_color-backward)**2,axis=-1))

    if threshold is not None:
        forward_threshold = tf.reduce_mean(dists_forward, keep_dims=True, axis=1) * threshold
        backward_threshold = tf.reduce_mean(dists_backward, keep_dims=True, axis=1) * threshold
        # only care about distance within threshold (ignore strong outliers)
        dists_forward = tf.where(dists_forward < forward_threshold, dists_forward, tf.zeros_like(dists_forward))
        dists_backward = tf.where(dists_backward < backward_threshold, dists_backward, tf.zeros_like(dists_backward))
        # dists_forward is for each element in gt, the closest distance to this element
    dists_forward = tf.reduce_mean(dists_forward, axis=1)
    dists_backward = tf.reduce_mean(dists_backward, axis=1)
    CD_dist = forward_weight * dists_forward + dists_backward
        # CD_dist_norm = CD_dist/radius
    cd_loss = tf.reduce_mean(CD_dist)

    return cd_loss


def get_hausdorff_loss(pred, gt, radius, forward_weight=1.0, threshold=None):
    """
    pred: BxNxC,
    label: BxN,
    forward_weight: relative weight for forward_distance
    """
    with tf.name_scope("cd_loss"):
        dists_forward, _, dists_backward, _ = tf_nndistance.nn_distance(gt, pred)
        # only care about distance within threshold (ignore strong outliers)
        if threshold is not None:
            dists_forward = tf.where(dists_forward < threshold, dists_forward, tf.zeros_like(dists_forward))
            dists_backward = tf.where(dists_backward < threshold, dists_backward, tf.zeros_like(dists_backward))
        # dists_forward is for each element in gt, the closest distance to this element
        dists_forward = tf.reduce_max(dists_forward, axis=1)
        dists_backward = tf.reduce_max(dists_backward, axis=1)
        CD_dist = forward_weight * dists_forward + dists_backward
        CD_dist_norm = CD_dist/radius
        cd_loss = tf.reduce_max(CD_dist_norm)
        return cd_loss, None


def get_cd_losses(preds, gt, radius, weights):
    losses = []
    for pred, weight in zip(preds, weights):
        loss, _ = get_cd_loss(pred, gt, radius)
        loss = weight*loss
        losses.append(loss)

    return losses, None


if __name__ == '__main__':
    gt = tf.constant([[[1,0,0],[2,0,0],[3,0,0],[4,0,0]]],tf.float32)
    pred = tf.constant([[[-10,0,0], [1,0, 0], [2,0, 0], [3,0,0]]],tf.float32)

    dists_forward, idx1, dists_backward, idx2 = tf_nndistance.nn_distance(gt, pred)
    with tf.Session() as sess:
        print(idx1.eval()) # for each element in gt, the idx of pred
        print(idx2.eval())# for each element in pred,