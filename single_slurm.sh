#!/bin/sh
##SBATCH --export=ALL
#SBATCH -N1
##SBATCH --nodelist=gpu018
##SBATCH --ntasks=1
#SBATCH --qos=long
#SBATCH --time=10-0
#SBATCH --cpus-per-task=1
#SBATCH -G0
#SBATCH --mem-per-cpu=100G

# export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX
#export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH



python /home/spotter5/combustion_v2/LCC.py








