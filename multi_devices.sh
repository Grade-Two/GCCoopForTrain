#!/bin/bash

# download cifar10 if necessery
cd ResNet
if [ ! -d "cifar-10-python/" ];then
wget http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar -zxvf cifar-10-python.tar.gz
fi

# run experiment
python main_multi.py \
    --device="/cpu:0" \
    --batch_size=100 \
    --dataset_dir="cifar-10-python/cifar-10-batches-py/" \
    --train_dir="/tmp/train" \
    --ps_hosts="192.168.32.145:22221" \
    --worker_hosts="192.168.32.145:22222, 192.168.32.165:22223, 192.168.32.165:22224, 192.168.32.165:22225" \
    --job_name="ps" \
    --task_index=0
    > single_node.txt &

python main_multi.py \
    --device="/cpu:1" \
    --batch_size=100 \
    -- dataset_dir="cifar-10-python/cifar-10-batches-py/" \
    --train_dir="/tmp/train" \
    --ps_hosts="192.168.32.145:22221" \
    --worker_hosts="192.168.32.145:22222, 192.168.32.165:22223, 192.168.32.165:22224, 192.168.32.165:22225" \
    --job_name="worker" \
    --task_index=1
    > single_node.txt &
    
python main_multi.py \
    --device="/cpu:2" \
    --batch_size=100 \
    -- dataset_dir="cifar-10-python/cifar-10-batches-py/" \
    --train_dir="/tmp/train" \
    --ps_hosts="192.168.32.145:22221" \
    --worker_hosts="192.168.32.145:22222, 192.168.32.165:22223, 192.168.32.165:22224, 192.168.32.165:22225" \
    --job_name="ps" \
    --task_index=2
    > single_node.txt &
