#!/bin/bash

#SBATCH --job-name=Project-train
#SBATCH --account=eecs595f25_class
#SBATCH --partition=spgpu
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=40g
#SBATCH --mail-user=bchao@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/train_BERT.out

python train_classifier.py \
  --train splits/train.zst \
  --val splits/val.zst \
  --mapping college_to_id.py \
  --model answerdotai/ModernBERT-base \
  --outdir model_out \
  --wandb-project college-classification \
  --epochs 2 \
  --batch-size 8 \
  --max-length 128 \
  --log-steps 500 \
  --eval-steps 5000 \
  --save-steps 7500 \
  --enable-tf32 \
  --use-bf16
# python train_classifier.py \
#   --train splits/train.zst \
#   --val splits/val.zst \
#   --mapping college_to_id.py \
#   --model answerdotai/ModernBERT-base \
#   --outdir model_out \
#   --epochs 1 \
#   --batch-size 2 \
#   --max-train 200 \
#   --max-val 5 \
#   --max-length 128 \
#   --no-wandb
  