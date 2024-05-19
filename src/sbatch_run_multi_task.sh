#!/bin/bash
#SBATCH --job-name=nlp-multi-task 
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1 #--gpus=1
#SBATCH --output=multi-task.out
#SBATCH --reservation=fri
#SBATCH --time=04:00:00

module load CUDA/12.1.1

srun singularity exec --nv ./containers/container-torch.sif python "multi_task_run.py"