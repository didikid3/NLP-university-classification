#!/bin/bash

#SBATCH --job-name=Project-train
#SBATCH --account=cse585f25_class
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
  --class-weight balanced \
  --epochs 1 \
  --batch-size 16 \
  --max-length 256 \
  --log-steps 500 \
  --eval-steps 50000 \
  --eval-samples 10000 \
  --eval-random \
  --save-steps 200000 \
  --seed 1234 \
  --enable-tf32 \
  --use-bf16
