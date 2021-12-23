# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import math
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

import ops
# import model_utils

def placeholder_inputs(batch_size, num_point,up_ratio = 6):
    up_ratio = up_ratio-2
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    pointclouds_gt = tf.placeholder(tf.float32, shape=(batch_size, num_point*up_ratio, 3))
    pointclouds_normal = tf.placeholder(tf.float32, shape=(batch_size, num_point *up_ratio, 3))
    pointclouds_radius = tf.placeholder(tf.float32, shape=(batch_size))
    return pointclouds_pl, pointclouds_gt,pointclouds_normal, pointclouds_radius


def get_gen_model(point_cloud,up_ratio,FLAGS,is_training, bn_decay=None):
  """ Classification PointNet, input is BxNx3, output Bx40 """
  batch_size = point_cloud.get_shape()[0].value
  num_point = point_cloud.get_shape()[1].value
  end_points = {}

  with tf.variable_scope('generator') as sc:

      ## features extraction
      features = ops.feature_extraction(point_cloud, scope='spatio_feature_extraction2', is_training=is_training, bn_decay=None)

      coarse,errors_features = ops.attention_point_extend_unit(features,point_cloud,up_ratio,k=8, FLAGS=FLAGS,scope='attention_unit', is_training=True)

      coarse = tf.reshape(coarse ,[batch_size,num_point*(up_ratio),1,-1])
      errors_features = tf.reshape(errors_features ,[batch_size,num_point*(up_ratio),1,-1])
      errors_coord= ops.coordinate_reconstruction_unit(errors_features ,scope="coord_reconstruction",is_training=True,bn_decay=bn_decay,use_bias=True)
      
      coord = tf.squeeze(coarse+errors_coord,axis=-2) #(B,rN,3)
    
  return coord


if __name__=='__main__':
  batch_size = 2
  num_pt = 124
  pos_dim = 3

  input_feed = np.random.rand(batch_size, num_pt, pos_dim)
  label_feed = np.random.rand(batch_size)
  label_feed[label_feed>=0.5] = 1
  label_feed[label_feed<0.5] = 0
  label_feed = label_feed.astype(np.int32)


  with tf.Graph().as_default():
    pointclouds_pl, pointclouds_gt, pointclouds_gt_normal, pointclouds_radius = MODEL_GEN.placeholder_inputs(
      BATCH_SIZE=2, NUM_POINT=124, UP_RATIO=2)

    pred = get_gen_model(pointclouds_pl, up_ratio=2,is_training=True, bn_decay=None)
    # loss = get_loss(logits, label_pl, None)

    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      feed_dict = {pointclouds_pl: input_feed}
      pred = sess.run(pred, feed_dict=feed_dict)
      pred(pred,pred.shape)


