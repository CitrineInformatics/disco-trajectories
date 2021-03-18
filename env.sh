#!/bin/bash

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86-64.sh -b
echo `which conda`
conda create -n disco
conda activate disco
conda install --yes numpy scipy matplotlib scikit-learn
conda install --yes --channel conda-forge pymatgen=2021.2.16
conda install --yes --channel conda-forge matminer
