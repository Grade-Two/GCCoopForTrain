#!/bin/bash

# download cifar10 if necessery
cd ResNet
if [ ! -d "cifar-10-batches-py/" ];then
wget http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar -zxvf cifar-10-python.tar.gz
fi

# run experiment
python main_single.py --device="/cpu:0" --batch_size=64 --dataset_dir="cifar-10-batches-py/" > single_node.txt
