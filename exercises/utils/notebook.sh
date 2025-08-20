#!/bin/bash
#SBATCH --job-name=notebook
#SBATCH --time=999:000
#SBATCH --partition=80g
##SBATCH --nodelist=nv174
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=4G
#SBATCH --output=notebook.out
#SBATCH --comment="LLM 테스트"

export CONTAINER_IMAGE_PATH='/purestorage/AILAB/AI_1/tyk/0_Software/sqsh/llm_27_v16.sqsh'
export CACHE_FOR_PATH='/purestorage/AILAB/AI_1/tyk/0_Software/cache'
export MY_WORKSPACE_PATH='/purestorage/AILAB/AI_1/tyk/3_CUProjects/language_model'

PORT=$(shuf -i 10000-19999 -n 1)

echo $PORT

# --NotebookApp.token='' --NotebookApp.allow_remote_access=True 
srun --container-image $CONTAINER_IMAGE_PATH \
    --container-mounts /purestorage:/purestorage,$CACHE_FOR_PATH:/home/$USER/.cache \
    --no-container-mount-home \
    --container-writable \
    --container-workdir $MY_WORKSPACE_PATH \
    jupyter notebook --ip 0.0.0.0 --port=${PORT} --ServerApp.allow_origin='*' --ServerApp.trust_xheaders=True  --ServerApp.disable_check_xsrf=True

# srun jupyter notebook --ip 0.0.0.0 --port=${PORT} --ServerApp.allow_origin='*' --ServerApp.trust_xheaders=True  --ServerApp.disable_check_xsrf=True