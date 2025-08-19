#!/bin/bash -l
#SBATCH --job-name=vlm-test
#SBATCH --time=99:999:000
#SBATCH --nodelist=nv180
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --comment='스마트 관제'
#SBATCH --output=./logs/ss_%j.out

export CONTAINER_IMAGE_PATH='/purestorage/AILAB/AI_1/tyk/0_Software/sqsh/llm_27_v15.sqsh'
export CACHE_FOR_PATH='/purestorage/AILAB/AI_1/tyk/0_Software/cache'
export MY_WORKSPACE_PATH='/purestorage/AILAB/AI_1/tyk/3_CUProjects/VLMs/smart-surveillance'

# Make sure Python can find our local package inside src
export PYTHONPATH="$MY_WORKSPACE_PATH/src:$PYTHONPATH"

srun --container-image $CONTAINER_IMAGE_PATH \
    --container-mounts /purestorage:/purestorage,$CACHE_FOR_PATH:/home/$USER/.cache \
    --no-container-mount-home \
    --container-writable \
    --container-workdir $MY_WORKSPACE_PATH \
    python -u app.py 2>&1 | tee gradio_output.log
    # python app.py # --video /purestorage/AILAB/AI_1/tyk/3_CUProjects/VLMs/smart-surveillance/samples/intrusion_climb-over-fence_rgb_0954_cctv2_original.mp4 --backend yoloworld