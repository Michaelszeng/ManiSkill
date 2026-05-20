#!/bin/bash
#SBATCH --job-name=ppo_planar_pusht
#SBATCH --account=locomotion
#SBATCH --partition=vision-shared-rtx3090
#SBATCH --exclude=freeman-titanrtx-2,freeman-titanrtx-1
#SBATCH --qos=shared-if-available
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --requeue

# Usage:
#   sbatch examples/baselines/ppo/submit_ppo_train.sbatch [env_id]
#
# SLURM runner for PPO training on ManiSkill environments.
# Defaults to Planar-PushT-v1 if no env_id is provided.

ENV_ID="${1:-Planar-PushT-v1}"

DATE=$(date +"%Y-%m-%d")
TIME=$(date +"%H-%M-%S")
echo "DATE: $DATE"
echo "TIME: $TIME"
echo "ENV_ID: $ENV_ID"

# ============================================================================
# Training parameters
# ============================================================================
NUM_ENVS=4096
NUM_STEPS=16
UPDATE_EPOCHS=8
NUM_MINIBATCHES=32
GAMMA=0.99
TOTAL_TIMESTEPS=1_000_000_000
NUM_EVAL_STEPS=400
NUM_EVAL_ENVS=16
CONTROL_MODE="pd_ee_delta_pose"
ENT_COEF=0.01
# ============================================================================

echo "SLURM Job ID:         $SLURM_JOB_ID"
echo "SLURM Node:           $HOSTNAME"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "SLURM_SUBMIT_DIR:     $SLURM_SUBMIT_DIR"

# --- Environment setup ---
echo "Changing directory to $SLURM_SUBMIT_DIR"
cd "$SLURM_SUBMIT_DIR" || { echo "Failed to change directory to $SLURM_SUBMIT_DIR"; exit 1; }

# Conda setup
source /data/locomotion/michzeng/miniconda3/etc/profile.d/conda.sh
conda activate Maniskill
# Restore CUDA_VISIBLE_DEVICES in case conda activation cleared it.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${SLURM_STEP_GPUS:-${SLURM_JOB_GPUS:-0}}}"
echo "CUDA_VISIBLE_DEVICES after conda activate: ${CUDA_VISIBLE_DEVICES}"

mkdir -p logs

export PYTHONPATH=$PWD:$PYTHONPATH

echo "=========================================="
echo "ENV_ID:            $ENV_ID"
echo "NUM_ENVS:          $NUM_ENVS"
echo "NUM_STEPS:         $NUM_STEPS"
echo "UPDATE_EPOCHS:     $UPDATE_EPOCHS"
echo "NUM_MINIBATCHES:   $NUM_MINIBATCHES"
echo "GAMMA:             $GAMMA"
echo "TOTAL_TIMESTEPS:   $TOTAL_TIMESTEPS"
echo "NUM_EVAL_STEPS:    $NUM_EVAL_STEPS"
echo "NUM_EVAL_ENVS:     $NUM_EVAL_ENVS"
echo "CONTROL_MODE:      $CONTROL_MODE"
echo "ENT_COEF:          $ENT_COEF"
echo "=========================================="

python examples/baselines/ppo/ppo_fast.py \
    --env_id="${ENV_ID}" \
    --num_envs=${NUM_ENVS} \
    --num-steps=${NUM_STEPS} \
    --update_epochs=${UPDATE_EPOCHS} \
    --num_minibatches=${NUM_MINIBATCHES} \
    --gamma=${GAMMA} \
    --total_timesteps=${TOTAL_TIMESTEPS} \
    --num_eval_steps=${NUM_EVAL_STEPS} \
    --num_eval_envs=${NUM_EVAL_ENVS} \
    --control_mode=${CONTROL_MODE} \
    --ent-coef=${ENT_COEF} \
    --cudagraphs
