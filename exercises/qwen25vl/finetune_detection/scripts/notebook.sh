#!/bin/bash
#SBATCH --job-name=notebook
#SBATCH --output=notebook.out
#SBATCH --nodelist=nv174
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=4G
#SBATCH --comment="VLM 테스트"
#SBATCH --output=./logs/notebook_%j.out

##### Number of total processes
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "
echo "Job name:= " "$SLURM_JOB_NAME"
echo "Nodelist:= " "$SLURM_JOB_NODELIST"
echo "Number of nodes:= " "$SLURM_JOB_NUM_NODES"
echo "Ntasks per node:= "  "$SLURM_NTASKS_PER_NODE"
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "

echo "Run started at:- "
date

srun jupyter notebook --ip 0.0.0.0
