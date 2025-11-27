#!/bin/bash

# python train_classifier.py \
#   --train splits/train.zst \
#   --val splits/val.zst \
#   --mapping college_to_id.py \
#   --model answerdotai/ModernBERT-base \
#   --outdir model_out \
#   --epochs 1 \
#   --batch-size 8 \
#   --max-train 2000 \
#   --max-val 500

python train_classifier.py \
  --train splits/train.zst \
  --val splits/val.zst \
  --mapping college_to_id.py \
  --model answerdotai/ModernBERT-base \
  --outdir model_out \
  --epochs 1 \
  --batch-size 2 \
  --max-train 200 \
  --max-val 5 \
  --max-length 128