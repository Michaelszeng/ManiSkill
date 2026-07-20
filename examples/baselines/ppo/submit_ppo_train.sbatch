#!/bin/bash
#SBATCH --job-name=ppo_train
#SBATCH --account=<>
#SBATCH --partition=<>
#SBATCH --exclude=<>
#SBATCH --qos=<>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --requeue

# Usage:
#   sbatch examples/baselines/ppo/submit_ppo_train.sbatch [env_id] [checkpoint]
#
# SLURM wrapper around examples/baselines/ppo/train_ppo.sh.
# Handles all environment setup, then hands off to the training script.
# Defaults to Planar-PushT-v1 if no env_id is provided.
# Optionally warm-start training from a checkpoint path.

ENV_ID="${1:-Planar-PushT-v1}"
CHECKPOINT="${2:-}"

echo "SLURM Job ID:         $SLURM_JOB_ID"
echo "SLURM Node:           $HOSTNAME"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "SLURM_SUBMIT_DIR:     $SLURM_SUBMIT_DIR"

# --- Working directory ---
echo "Changing directory to $SLURM_SUBMIT_DIR"
cd "$SLURM_SUBMIT_DIR" || { echo "Failed to change directory to $SLURM_SUBMIT_DIR"; exit 1; }

# --- WANDB Authentication ---
export WANDB_USERNAME="michzeng"
SECRETS_FILE="$SLURM_SUBMIT_DIR/.secrets"
if [ -f "$SECRETS_FILE" ]; then
    source "$SECRETS_FILE"
    echo "Loaded secrets from $SECRETS_FILE"
else
    echo "ERROR: $SECRETS_FILE not found. Create it from .secrets.template before submitting." >&2
    exit 1
fi
echo "WANDB_USERNAME and WANDB_API_KEY set for authentication."

# --- Conda setup ---
conda activate PATH-TO-CONDA-ENV

mkdir -p logs

export PYTHONPATH=$PWD:$PYTHONPATH

bash examples/baselines/ppo/train_ppo.sh "$ENV_ID" "$CHECKPOINT"
