#!/bin/bash
#SBATCH --job-name=evaluate_pusht
#SBATCH --account=locomotion
#SBATCH --partition=vision-shared-rtx3090
#SBATCH --exclude=freeman-titanrtx-2,freeman-titanrtx-1,gpu20-2.drl,improbablex002
#SBATCH --qos=shared-if-available
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --requeue

# Usage:
#   sbatch diffusion_policy/submit_evaluation.sbatch <action_horizon> [action_horizon ...]
#
# SLURM runner for diffusion_policy/evaluation.py.
# Pass one or more action horizons; all checkpoints in CHECKPOINT_PATH are
# evaluated at each. Other tunables (NUM_TRIALS_PER_HORIZON=500, NUM_ENVS=16,
# MAX_EPISODE_STEPS=200, etc.) are constants at the top of evaluation.py —
# edit them there.

if [[ $# -eq 0 ]]; then
    echo "Usage: sbatch diffusion_policy/submit_evaluation.sbatch <action_horizon> [action_horizon ...]"
    exit 1
fi
ACTION_HORIZONS=("$@")

DATE=$(date +"%Y-%m-%d")
TIME=$(date +"%H-%M-%S")
echo "DATE: $DATE"
echo "TIME: $TIME"
echo "ACTION_HORIZONS: ${ACTION_HORIZONS[*]}"

# ============================================================================
# Evaluation parameters
# ============================================================================
# Directory containing one or more .ckpt files (all will be evaluated at each horizon).
CHECKPOINT_PATH=""

# Number of trials per (checkpoint, action_horizon) to save MP4 videos for.
# 0 = off, -1 = all. Anything > 0 forces rgb_array rendering (a bit slower).
N_VIDEO_TRIALS=20
# ============================================================================

# Extract the experiment name as the directory immediately before "checkpoints".
EXPERIMENT_NAME="$(basename "$(dirname "${CHECKPOINT_PATH}")")"
echo "EXPERIMENT_NAME: $EXPERIMENT_NAME"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "SLURM Node: $HOSTNAME"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "SLURM_SUBMIT_DIR: $SLURM_SUBMIT_DIR"

# --- Environment setup ---
echo "Changing directory to $SLURM_SUBMIT_DIR"
cd "$SLURM_SUBMIT_DIR" || { echo "Failed to change directory to $SLURM_SUBMIT_DIR"; exit 1; }

conda activate PATH-TO-CONDA-ENV

mkdir -p logs

export PYTHONPATH=$PWD:$PYTHONPATH

echo "CHECKPOINT_PATH:  $CHECKPOINT_PATH"
echo "ACTION_HORIZONS:  ${ACTION_HORIZONS[*]}"
echo "N_VIDEO_TRIALS:   $N_VIDEO_TRIALS"
echo "=========================================="

export PYTHONUNBUFFERED=1

python diffusion_policy/evaluation.py \
    --checkpoints-dir "${CHECKPOINT_PATH}" \
    --action-horizons "${ACTION_HORIZONS[@]}" \
    --n-video-trials "${N_VIDEO_TRIALS}" \
    --output-dir "outputs/${EXPERIMENT_NAME}" \
    --resume
