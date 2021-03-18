#!/usr/bin/env bash

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda
export PATH="$HOME/miniconda/bin:$PATH"
echo `which conda`
conda install --yes numpy scipy matplotlib scikit-learn git
conda install --yes --channel conda-forge pymatgen==2021.2.16 matminer==0.6.5
git clone https://github.com/CitrineInformatics/disco-trajectories.git
cd disco-trajectories && pip install -e .
