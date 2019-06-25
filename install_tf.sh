#!/bin/bash

# install anaconda3
cd ..
wget https://repo.anaconda.com/archive/Anaconda3-2019.03-Linux-x86_64.sh
chmod a+x Anaconda3-2019.03-Linux-x86_64.sh
(
echo 
# echo "q"
echo "yes"
echo 
echo "yes"
) | ./Anaconda3-2019.03-Linux-x86_64.sh
source ~/.bashrc

# install tensorflow
echo "y" | conda install tensorflow
