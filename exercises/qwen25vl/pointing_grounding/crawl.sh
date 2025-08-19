#!/bin/bash -l

#SBATCH --job-name=vlm
#SBATCH --time=999:000
#SBATCH --partition=40g
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --qos=normal
#SBATCH --ntasks-per-node=1
#SBATCH --comment="crawling"
#SBATCH --output=crawling_%j.out

export CONTAINER_IMAGE_PATH='/purestorage/AILAB/AI_1/tyk/0_Software/sqsh/llm_27_v4.sqsh'
export CACHE_FOR_PATH='/purestorage/AILAB/AI_1/tyk/0_Software/cache'
export MY_WORKSPACE_PATH='/purestorage/AILAB/AI_1/tyk/3_CUProjects/language_model/VLM/qwen25vl/pointing_grounding'

export UNSPLASH_ACCESS_KEY='OrMJs9mvywQP8VWE5EVXhtfGXqqLlmBNk1IZD_kuoC0'
export OUTPUT_PATH='/purestorage/AILAB/AI_1/tyk/1_Data/crawling'

srun --container-image $CONTAINER_IMAGE_PATH \
    --container-mounts /purestorage:/purestorage,$CACHE_FOR_PATH:/home/$USER/.cache \
    --no-container-mount-home \
    --container-writable \
    --container-workdir $MY_WORKSPACE_PATH \
    bash -c 'pip install openimages && which openimages || echo "will fall back to python -m openimages" && python crawl.py --output-dir $OUTPUT_PATH --keywords "airport bird flock" "flock of birds airport" --limit 900 --providers "unsplash" "openimages"'