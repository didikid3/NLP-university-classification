#!/bin/bash
#
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
#SBATCH --output=logs/train_modernbert.out

set -euo pipefail
date; echo "Starting training job on $(hostname)"

# -------------------------
# Environment (edit as needed)
# -------------------------
# Example: activate your conda/venv
# conda activate myenv
# or
# source ~/venvs/modernbert/bin/activate

# Optional: tune HF cache / tokenizers / wandb directories (uncomment & edit)
# export HF_HOME=/scratch/$USER/hf_cache
# export TRANSFORMERS_CACHE=$HF_HOME
# export WANDB_DIR=/scratch/$USER/wandb


# -------------------------
# Paths / hyperparams (edit these)
# -------------------------
TRAIN_FILE="splits/train.zst"
VAL_FILE="splits/val.zst"
MAPPING_PY="college_to_id.py"            # your python mapping file (contains COLLEGE_TO_ID)
MODEL_NAME="answerdotai/ModernBERT-base"
OUTDIR="model_out"
LOG_STEPS=1000
SAVE_STEPS=200000
MAX_CHECKPOINTS=5

# mid-epoch evaluation (small subset)
MID_EVAL_STEPS=10000         # run mini-eval every N training steps (0 = disabled)
MID_EVAL_BATCHES=100        # number of validation batches to evaluate during mid-epoch eval

# training hyperparams
EPOCHS=1
BATCH_SIZE=16
GRADIENT_ACCUMULATION=4      # set >1 to simulate larger batch size
MAX_LENGTH=128
LR=2e-5
MAX_TRAIN=""                 # set to a number to limit train samples (or leave empty)
MAX_VAL=""                   # same for val

# wandb
USE_WANDB="--wandb"                 # set to empty string to disable
WANDB_PROJECT="college-classification"

# optional resume path (directory with training_state.pt or training_state.pt file)
# RESUME_ARG="--resume path/to/checkpoint_dir"
RESUME_ARG=""

# -------------------------
# Run training
# -------------------------
PYTHON=python  # or full path to your python

$PYTHON simple_train.py \
  --train "$TRAIN_FILE" \
  --val "$VAL_FILE" \
  --mapping "$MAPPING_PY" \
  --model "$MODEL_NAME" \
  --outdir "$OUTDIR" \
  ${USE_WANDB} \
  --wandb-project "$WANDB_PROJECT" \
  --epochs $EPOCHS \
  --batch-size $BATCH_SIZE \
  --max-length $MAX_LENGTH \
  --lr $LR \
  $( [ -n "$MAX_TRAIN" ] && echo "--max-train $MAX_TRAIN" || true ) \
  $( [ -n "$MAX_VAL" ] && echo "--max-val $MAX_VAL" || true ) \
  --grad-accum-steps $GRADIENT_ACCUMULATION \
  --log-steps $LOG_STEPS \
  --save-steps $SAVE_STEPS \
  --max-checkpoints $MAX_CHECKPOINTS \
  --mid-eval-steps $MID_EVAL_STEPS \
  --mid-eval-batches $MID_EVAL_BATCHES \
  $RESUME_ARG

EXIT_CODE=$?
date; echo "Finished training job with exit code $EXIT_CODE"
exit $EXIT_CODE
