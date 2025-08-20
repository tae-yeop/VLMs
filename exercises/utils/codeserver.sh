#!/bin/bash -l

#SBATCH --job-name=vlm-test
#SBATCH --time=99:999:000
#SBATCH --nodelist=nv180
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --comment='gemma 테스트'
#SBATCH --output=./logs/codeserver_%j.out

export CONTAINER_IMAGE_PATH='/purestorage/AILAB/AI_1/tyk/0_Software/sqsh/llm_27_v16.sqsh'
export CACHE_FOR_PATH='/purestorage/AILAB/AI_1/tyk/0_Software/cache'
export MY_WORKSPACE_PATH='/purestorage/AILAB/AI_1/tyk/3_CUProjects/VLMs/smart-surveillance'
export PASSWORD=''

export XDG_RUNTIME_DIR="/tmp/${USER}-runtime"
mkdir -p "$XDG_RUNTIME_DIR"
chmod 700 "$XDG_RUNTIME_DIR"

srun bash -lc '
module load devel/code-server && \
PASSWORD=tyk code-server "$MY_WORKSPACE_PATH" --bind-addr 0.0.0.0:8081
'

# 이후 로컬 윈도우 CMD에서 ssh -L 8081:172.100.100.11:8081 tyk@172.100.100.39
# 접속 로컬 윈도우 크롬에서 http://127.0.0.1:8081/ 접속