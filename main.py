import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))

import argparse
import os
import time
import numpy as np
import tensorflow as tf

from tqdm import tqdm
from glob import glob
import socket
from matplotlib import pyplot as plt
import importlib
from utils import model_utils

from utils import pc_util
from tf_ops.sampling.tf_sampling import farthest_point_sample, gather_point
from tf_ops.grouping.tf_grouping import query_ball_point, group_point, knn_point
from datetime import datetime
import json
from configs import FLAGS
from data_loader import Fetcher
from log import logger_tb, message_logger
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


loggerDir_ = FLAGS.log_dir
logger = logger_tb(FLAGS.log_dir, FLAGS.description, FLAGS.code_backup, FLAGS.use_tb)
loggerDir = os.path.join(os.path.dirname(__file__), loggerDir_, logger.log_dir())
if not os.path.exists(loggerDir):
    os.mkdir(loggerDir)
    
sys.stdout = message_logger(loggerDir)
MODEL_GEN = importlib.import_module('model') # import network module

ASSIGN_MODEL_PATH= None
USE_DATA_NORM = True
USE_RANDOM_INPUT = True
USE_REPULSION_LOSS = True
PHASE = FLAGS.phase
GPU_INDEX = FLAGS.gpu
BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
UP_RATIO = FLAGS.up_ratio
MAX_EPOCH = FLAGS.training_epoch


current_time = datetime.now().strftime("%Y%m%d-%H%M")
print(socket.gethostname())
print(FLAGS)

def train(assign_model_path=None):
    is_training = True
    bn_decay = 0.95
    step = tf.Variable(0,trainable=False)
    tf.summary.scalar('bn_decay', bn_decay)

    # get placeholder
    pointclouds_pl, pointclouds_gt, pointclouds_gt_normal, pointclouds_radius = MODEL_GEN.placeholder_inputs(BATCH_SIZE, NUM_POINT, UP_RATIO)
    #create the generator model
    pred_val= MODEL_GEN.get_gen_model(pointclouds_pl,UP_RATIO,FLAGS, is_training=is_training, bn_decay=None)
    pred_val = gather_point(pred_val,farthest_point_sample(FLAGS.num_point*4, pred_val))
    
    dis_loss = model_utils.get_cd_loss(pred_val, pointclouds_gt)
    uniform_loss =  model_utils.get_uniform_loss(pred_val)

    pre_gen_loss = FLAGS.dis_w*dis_loss + FLAGS.uniform_w*uniform_loss+tf.losses.get_regularization_loss()
    
    tf.summary.scalar('loss/dis_loss', dis_loss)
    tf.summary.scalar('loss/uniform_loss', uniform_loss)
    tf.summary.scalar('loss/regularation', tf.losses.get_regularization_loss())
    tf.summary.scalar('loss/pre_gen_total', pre_gen_loss)
    
    pretrain_merged = tf.summary.merge_all()

    learning_rate_g = tf.where(
        tf.greater_equal(step, FLAGS.start_decay_step),
        tf.train.exponential_decay(FLAGS.base_lr_g, step - FLAGS.start_decay_step,
                                   FLAGS.lr_decay_steps, FLAGS.lr_decay_rate, staircase=True),
        FLAGS.base_lr_g
    )
    learning_rate_g = tf.maximum(learning_rate_g, FLAGS.lr_clip)
    tf.summary.scalar('learning_rate/learning_rate_g', learning_rate_g, collections=['gen'])

    gen_update_ops = [op for op in tf.get_collection(tf.GraphKeys.UPDATE_OPS) if op.name.startswith("generator")]
    gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]

    with tf.control_dependencies(gen_update_ops):
        pre_gen_train = tf.train.AdamOptimizer(learning_rate_g,beta1=0.9).minimize(pre_gen_loss,var_list=gen_tvars,
                                                                                 colocate_gradients_with_ops=True,
                                                                                 global_step=step)

    pointclouds_image_input = tf.placeholder(tf.float32, shape=[None, 500, 1500, 1])
    pointclouds_input_summary = tf.summary.image('pointcloud_input', pointclouds_image_input, max_outputs=1)
    pointclouds_image_pred = tf.placeholder(tf.float32, shape=[None, 500, 1500, 1])

    pointclouds_pred1_summary = tf.summary.image('pointcloud_pred1', pointclouds_image_pred, max_outputs=1)
    pointclouds_image_gt = tf.placeholder(tf.float32, shape=[None, 500, 1500, 1])
    pointclouds_gt_summary = tf.summary.image('pointcloud_gt', pointclouds_image_gt, max_outputs=1)
    image_merged = tf.summary.merge([pointclouds_input_summary,pointclouds_pred1_summary,
                                     pointclouds_gt_summary])


    # Create a session
    config = tf.ConfigProto()
    config.gpu_options
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False
    with tf.Session(config=config) as sess:
        train_writer = tf.summary.FileWriter(os.path.join(loggerDir, 'train'), sess.graph)
        init = tf.global_variables_initializer()
        sess.run(init)
        ops = {'pointclouds_pl': pointclouds_pl,
               'pointclouds_gt': pointclouds_gt,
               'pointclouds_gt_normal':pointclouds_gt_normal,
               'pointclouds_radius': pointclouds_radius,
               'pointclouds_image_input':pointclouds_image_input,
               'pointclouds_image_pred': pointclouds_image_pred,
               'pointclouds_image_gt': pointclouds_image_gt,
               'pretrain_merged':pretrain_merged,
               'image_merged': image_merged,
               'uniform_loss':uniform_loss,
               'dis_loss': dis_loss,
               'pre_gen_train':pre_gen_train,
               'pred': pred_val,
               'step': step,
               }
        
        #restore the model
        saver = tf.train.Saver(max_to_keep=6)
        restore_epoch, checkpoint_path = model_utils.pre_load_checkpoint(loggerDir)
        global LOG_FOUT
        if restore_epoch==0:
            pass
        else:
            saver.restore(sess,checkpoint_path)
        if assign_model_path is not None:
            print("Load pre-train model from %s"%(assign_model_path))
            assign_saver = tf.train.Saver(var_list=[var for var in tf.trainable_variables() if var.name.startswith("generator")])
            assign_saver.restore(sess, assign_model_path)

        fetchworker = Fetcher(FLAGS,FLAGS.h5_file)
        fetchworker.start()

        for epoch in tqdm(range(restore_epoch,MAX_EPOCH+1),ncols=55):
            train_one_epoch(sess, ops, fetchworker, train_writer,epoch)
            if epoch % 5 == 0:
                saver.save(sess, os.path.join(loggerDir,"model"), global_step=epoch)
        fetchworker.shutdown()


def train_one_epoch(sess, ops, data_loader, train_writer,epoch):
    loss_sum = []
    fetch_time = 0
    epoch_time = 0
    
    epoch_start =  time.time()
    for batch_idx in range(data_loader.num_batches):
        batch_input_data, batch_data_gt, radius = data_loader.fetch()
        batch_input_data=batch_input_data[:,:,:]
        batch_data_gt= batch_data_gt[:,:,:]
        feed_dict = {ops['pointclouds_pl']: batch_input_data,
                     ops['pointclouds_gt']: batch_data_gt[:,:,:],
                     ops['pointclouds_gt_normal']:batch_data_gt[:,:,0:3],
                     ops['pointclouds_radius']: radius}
        summary,step, _, pred_val,dis_loss,uniform_loss= sess.run( [ops['pretrain_merged'],ops['step'],ops['pre_gen_train'],
                                                            ops['pred'],ops['dis_loss'],ops['uniform_loss']], feed_dict=feed_dict)

        print("\n [step/epoch]: [{}/{}] [{}/{}] dis_loss: {:>10.5f}  uniform_loss:{:>10.5f}".format(data_loader.num_batches,FLAGS.training_epoch,batch_idx,epoch,dis_loss,uniform_loss))
        train_writer.add_summary(summary, step)
        loss_sum.append(dis_loss)

      
        if step%30 == 0:
            pointclouds_image_input = pc_util.point_cloud_three_views(batch_input_data[0,:,:])
            pointclouds_image_input = np.expand_dims(np.expand_dims(pointclouds_image_input,axis=-1),axis=0)

            pointclouds_image_pred = pc_util.point_cloud_three_views(pred_val[0, :, 0:3])
            pointclouds_image_pred = np.expand_dims(np.expand_dims(pointclouds_image_pred, axis=-1), axis=0)
       
            pointclouds_image_gt = pc_util.point_cloud_three_views(batch_data_gt[0, :, 0:3])
            pointclouds_image_gt = np.expand_dims(np.expand_dims(pointclouds_image_gt, axis=-1), axis=0)

            feed_dict ={ops['pointclouds_image_input']:pointclouds_image_input,
                        ops['pointclouds_image_pred']: pointclouds_image_pred,
                        ops['pointclouds_image_gt']: pointclouds_image_gt,
                        }
            summary = sess.run(ops['image_merged'],feed_dict)
            train_writer.add_summary(summary,step)


    epoch_end = time.time()
    epoch_time = epoch_end-epoch_start
    print('----------------------------------',epoch_time,'--------------------------------------')
   

if __name__ == "__main__":
    np.random.seed(int(time.time()))
    tf.set_random_seed(int(time.time()))
    train(assign_model_path=ASSIGN_MODEL_PATH)
