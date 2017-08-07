#!/usr/bin/env bash

#Runs tensorflow 1.0.1
# Sets up python 3.6 with LD Preload for Centos6
#6/3/17


# Activate python
#source /share/apps/examples/python-3.6.source.sh
source /home/rsmith/python-2.7.13.source.sh

#export CUDA_VISIBLE_DEVICES=`nvidia-smi --query-gpu=index,memory.used --format=csv,nounits,noheader | awk -F "," '/, 0/ {print $1;exit}'` 

#export CUDA_VISIBLE_DEVICES='1,2,-1,0'
export CUDA_VISIBLE_DEVICES='-1'

echo $CUDA_VISIBLE_DEVICES


# Set the LD Lib path to include CUDA and gcc6.2
LD_LIBRARY_PATH="/share/apps/libc6_2.17/lib/x86_64-linux-gnu/:/share/apps/libc6_2.17/usr/lib64/:/share/apps/gcc-6.2.0/lib64:/share/apps/gcc-6.2.0/lib:/share/apps/python-3.6.0-shared/lib:/usr/local/cuda-8.0/lib64:/usr/local/cuda-8.0/lib:${LD_LIBRARY_PATH}" 

# Start main program
echo 'starting kDS TENSORBOARD script'

SECONDS=0

#kDS
/share/apps/libc6_2.17/lib/x86_64-linux-gnu/ld-2.17.so $(command -v /home/rsmith/Python-2.7.13/bin/python) -m tensorflow.tensorboard --logdir=./tensorboard/ --port=64444



duration=$SECONDS
echo "$(($duration / 60)) minutes and $(($duration % 60)) seconds elapsed."

echo 'ending script'

