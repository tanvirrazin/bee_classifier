#! /bin/bash
#$ -cwd
#$ -j y
#$ -S /bin/bash
#$ -m beas

export PATH=/usr/local/cuda-10.1/bin:/usr/local/cuda-10.1/NsightCompute-2019.1${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export CUDA_VISIBLE_DEVICES=0,1,2,3

/home/gpu-machine/anaconda3/envs/tanvir_env/bin/python train.py
