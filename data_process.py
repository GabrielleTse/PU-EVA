

import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))

import importlib
import numpy as np
import tensorflow as tf
from glob import glob
import re
import csv
from collections import OrderedDict
import os
import glob 

from tqdm import tqdm
from utils import pc_util
from utils.pc_util import load, save_ply_property
from utils.tf_util import normalize_point_cloud

from tf_ops.grouping.tf_grouping import knn_point

import evaluate_pugan
# from tf_ops.sampling.tf_sampling import farthest_point_sample, gather_point
from tf_ops.sampling.tf_sampling import farthest_point_sample, gather_point
from matplotlib import pyplot as plt
from configs import FLAGS



def generate_fixed_input(root_dir="./data/test_xyz/8192/", save_dir = "./data/test_xyz/8192_2048_input/",POINT_NUM=2048):
    

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    files = glob.glob(root_dir+"*.xyz")
    # print(files)
    for file in files:

        pc = pc_util.load(file,count=POINT_NUM)[:, :]
        path = os.path.join(save_dir,file.split('/')[-1])
        print(path,pc.shape)
        np.savetxt(path, pc,fmt='%.6f')

if __name__ == "__main__":
    generate_fixed_input(root_dir="./data/test_xyz/8192/", save_dir = "./data/test_xyz/9096_2048_input/",POINT_NUM=2048)
    