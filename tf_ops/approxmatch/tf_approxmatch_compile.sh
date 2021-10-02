#!/usr/bin/env bash
#/bin/bash

#/bin/bash
TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
echo $TF_INC
TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
c


# nvcc tf_nndistance_g.cu -o tf_nndistance_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -ccbin=/usr/bin/gcc-6 -I $TF_INC

# g++ -DCMAKE_C_COMPILER=/usr/bin/gcc-6 -std=c++11 tf_nndistance.cpp tf_nndistance_g.cu.o -o tf_nndistance_so.so  
# -shared -fPIC -I $TF_INC -I $CUDA_ROOT/include -I $TF_INC/external/nsync/public -lcudart -L $CUDA_ROOT/lib64/ -L$TF_LIB -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0


########################
# /usr/local/cuda-8.0/bin/nvcc tf_approxmatch_g.cu -o tf_approxmatch_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
# g++ -std=c++11 tf_approxmatch.cpp tf_approxmatch_g.cu.o -o tf_approxmatch_so.so -shared -fPIC -I /usr/local/lib/python2.7/dist-packages/tensorflow/include -I /usr/local/cuda-8.0/include -I /usr/local/lib/python2.7/dist-packages/tensorflow/include/external/nsync/public -lcudart -L /usr/local/cuda-8.0/lib64/ -L/usr/local/lib/python2.7/dist-packages/tensorflow -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0
#g++ -std=c++11 tf_approxmatch.cpp tf_approxmatch_g.cu.o -o tf_approxmatch_so.so -shared -fPIC -I /home/lirh/anaconda3/envs/tensorflow11/lib/python3.6/site-packages/tensorflow/include  -I /usr/local/cuda-8.0/include -I /home/lirh/anaconda3/envs/tensorflow11/lib/python3.6/site-packages/tensorflow/include/external/nsync/public -lcudart -L /usr/local/cuda-8.0/lib64/ -L/home/lirh/anaconda3/envs/tensorflow11/lib/python3.6/site-packages/tensorflow -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=1

###########################

/usr/local/cuda-10.0/bin/nvcc tf_grouping_g.cu -o tf_grouping_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
# g++ -std=c++11 tf_approxmatch.cpp tf_approxmatch_g.cu.o -o tf_approxmatch_so.so -shared -fPIC -I $TF_INC -I $CUDA_ROOT/include -I $TF_INC/external/nsync/public -lcudart -L $CUDA_ROOT/lib64/ -L$TF_LIB -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0
g++ -std=c++11 tf_approxmatch.cpp tf_approxmatch_g.cu.o -o tf_approxmatch_so.so -shared -fPIC -I /home/anaconda3/envs/wanyi/lib/python3.7/site-packages/tensorflow/include  -I /usr/local/cuda-10.0/include -I /home/anaconda3/envs/wanyi/lib/python3.7/site-packages/tensorflow/include/external/nsync/public -lcudart -L /usr/local/cuda-10.0/lib64/ -L/home/anaconda3/envs/wanyi/lib/python3.7/site-packages/tensorflow -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=1
