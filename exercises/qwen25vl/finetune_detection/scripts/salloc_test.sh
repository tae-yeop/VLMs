#!/bin/bash

# SLURM 리소스 요청
salloc \
  --job-name=vlm-test \
  --time=04:00:00 \
  --nodelist=hpe160 \
  --gres=gpu:1 \
  --cpus-per-task=32 \
  --mem=64G \
  --ntasks=1 \
  --qos=normal \
  --comment="qwen 테스트"