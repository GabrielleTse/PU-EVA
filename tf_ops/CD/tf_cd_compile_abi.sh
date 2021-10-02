#!/usr/bin/env bash
nvcc=/usr/local/cuda-10.0/bin/nvcc
cudalib=/usr/local/cuda-10.0/lib64/
TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')

$nvcc tf_nndistance_g.cu -c -o tf_nndistance_g.cu.o  -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++11  -I $TF_INC -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -O2

g++ tf_nndistance.cpp tf_nndistance_g.cu.o -o tf_nndistance_so.so -std=c++11 -shared -fPIC -I /home/anaconda3/envs/wanyi/lib/python3.7/site-packages/tensorflow/include  -I /usr/local/cuda-10.0/include -I /home/anaconda3/envs/wanyi/lib/python3.7/site-packages/tensorflow/include/external/nsync/public -lcudart -L /usr/local/cuda-10.0/lib64/ -L/home/anaconda3/envs/wanyi/lib/python3.7/site-packages/tensorflow -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=1
