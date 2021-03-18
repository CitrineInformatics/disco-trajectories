#!/bin/bash

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b
source miniconda3/bin/activate
echo `which conda`
conda create -n disco python --yes && conda activate disco
echo `which python`
conda install --yes numpy scipy matplotlib scikit-learn
pip install pymatgen==2021.2.16 matminer==0.6.5
conda install git
git clone https://github.com/CitrineInformatics/disco-trajectories.git
cd disco-trajectores/src/discworld && python pso.py
