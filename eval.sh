#!/bin/bash

#SBATCH --job-name=Project-eval
#SBATCH --account=eecs595f25_class
#SBATCH --partition=spgpu
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=40g
#SBATCH --mail-user=bchao@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/eval_BERT.out

python eval_model.py \
    --val splits/test.zst \
    --mapping college_to_id.py \
    --model answerdotai/ModernBERT-base \
    --checkpoint model_out/pytorch_model_epoch1.pt \
    # --max-val 100