#!/bin/bash

# Download and install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda
rm miniconda.sh
export PATH="$HOME/miniconda/bin:$PATH"

# Create a Conda environment
conda create -y --name sersformer python=3.11.4

# Activate the Conda environment
source activate sersformer

# Install PyTorch with CUDA 11
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install PyTorch Lightning
pip install pytorch-lightning==1.8.6

# Install other required packages from required_packages.txt
while read package; do
    conda install -y $package
done < required_packges.txt
