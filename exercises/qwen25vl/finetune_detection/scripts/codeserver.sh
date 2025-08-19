#!/bin/bash -l

#SBATCH --job-name=vlm-test
#SBATCH --time=999:999:000
#SBATCH --nodelist=hpe160
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --ntasks-per-node=1
#SBATCH --comment='qwen 테스트'
#SBATCH --qos=normal
#SBATCH --output=./logs/codeserver_%j.out

export CACHE_FOR_PATH='/purestorage/AILAB/AI_1/tyk/0_Software/cache'
export MY_WORKSPACE_PATH='/purestorage/AILAB/AI_1/tyk/3_CUProjects/VLM'
export PASSWORD='tyk'

export XDG_RUNTIME_DIR="/tmp/${USER}-runtime"
mkdir -p "$XDG_RUNTIME_DIR"
chmod 700 "$XDG_RUNTIME_DIR"

# srun bash -lc '
# module load devel/code-server && \
# PASSWORD=tyk code-server "$MY_WORKSPACE_PATH" --bind-addr 0.0.0.0:8081
# '

module load devel/code-server
exec code-server "$MY_WORKSPACE_PATH" --bind-addr 0.0.0.0:8081
# 이후 로컬 윈도우 CMD에서 ssh -o ServerAliveInterval=60 -L 8081:172.100.100.11:8081 tyk@172.100.100.39
# 접속 로컬 윈도우 크롬에서 http://127.0.0.1:8081/ 접속
# enroot create --name pix /purestorage/project/tyk/0_Software/sqsh/pix.sqsh
# enroot start --rw --mount /purestorage/ ngc-2305