#!/bin/bash
#SBATCH --job-name=ppo_gen_data
#SBATCH --account=locomotion
#SBATCH --partition=vision-shared-rtx2080ti
#SBATCH --exclude=freeman-titanrtx-2,freeman-titanrtx-1
#SBATCH --qos=shared-if-available
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

# Usage:
#   sbatch examples/baselines/ppo/submit_ppo_gen_data.sbatch
#
# Pipeline:
#   1. Evaluate PPO checkpoint to collect raw trajectories
#   2. Rename trajectory files to the format expected by replay_trajectory
#   3. Replay trajectories with RGB observations
#   4. Convert .h5 dataset to .zarr format for diffusion policy training
#
# Expect this to take ~30 min to run.

# ============================================================================
# Constants
# ============================================================================
CHECKPOINT=/data/locomotion/michzeng/ManiSkill/runs/Planar-PushT-v1__ppo_fast__1__1779308025/checkpoints/ckpt_2401.pt

ENV_ID="Planar-PushT-v1"
CONTROL_MODE="pd_ee_delta_pose"
NUM_EVAL_ENVS=1
NUM_EVAL_STEPS=50000
EVAL_TEMPERATURE=0.0
REPLAY_COUNT=300
ZARR_OUTPUT=/data/locomotion/michzeng/diffusion-policy-experiments/data/diffusion_experiments/maniskill/maniskill_planar_push_t_rl_expert.zarr
# ============================================================================

DATE=$(date +"%Y-%m-%d")
TIME=$(date +"%H-%M-%S")
echo "DATE: $DATE"
echo "TIME: $TIME"

echo "SLURM Job ID:         $SLURM_JOB_ID"
echo "SLURM Node:           $HOSTNAME"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "SLURM_SUBMIT_DIR:     $SLURM_SUBMIT_DIR"

# --- Environment setup ---
echo "Changing directory to $SLURM_SUBMIT_DIR"
cd "$SLURM_SUBMIT_DIR" || { echo "Failed to change directory to $SLURM_SUBMIT_DIR"; exit 1; }

source /data/locomotion/michzeng/miniconda3/etc/profile.d/conda.sh
conda activate /data/locomotion/michzeng/conda_envs/Maniskill
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${SLURM_STEP_GPUS:-${SLURM_JOB_GPUS:-0}}}"
echo "CUDA_VISIBLE_DEVICES after conda activate: ${CUDA_VISIBLE_DEVICES}"

mkdir -p logs

export PYTHONPATH=$PWD:$PYTHONPATH

# Derived paths
CHECKPOINT_DIR=$(dirname "$CHECKPOINT")
TEST_VIDEOS_DIR="$CHECKPOINT_DIR/test_videos"

echo "=========================================="
echo "CHECKPOINT:        $CHECKPOINT"
echo "CHECKPOINT_DIR:    $CHECKPOINT_DIR"
echo "TEST_VIDEOS_DIR:   $TEST_VIDEOS_DIR"
echo "ENV_ID:            $ENV_ID"
echo "CONTROL_MODE:      $CONTROL_MODE"
echo "NUM_EVAL_ENVS:     $NUM_EVAL_ENVS"
echo "NUM_EVAL_STEPS:    $NUM_EVAL_STEPS"
echo "EVAL_TEMPERATURE:  $EVAL_TEMPERATURE"
echo "REPLAY_COUNT:      $REPLAY_COUNT"
echo "ZARR_OUTPUT:       $ZARR_OUTPUT"
echo "=========================================="

# ============================================================================
# Step 1: Evaluate PPO checkpoint to collect raw trajectories
# ============================================================================
echo ""
echo "=== Step 1: Generating trajectories from PPO checkpoint ==="
python examples/baselines/ppo/ppo_fast.py \
    --env_id="${ENV_ID}" \
    --control-mode="${CONTROL_MODE}" \
    --evaluate \
    --checkpoint="${CHECKPOINT}" \
    --num_eval_envs=${NUM_EVAL_ENVS} \
    --num-eval-steps=${NUM_EVAL_STEPS} \
    --save-trajectory \
    --no-capture-video \
    --eval-partial-reset \
    --eval-temperature=${EVAL_TEMPERATURE}

if [ $? -ne 0 ]; then
    echo "ERROR: PPO evaluation failed." >&2
    exit 1
fi

# ============================================================================
# Step 2: Rename trajectory files to the format expected by replay_trajectory
# ============================================================================
echo ""
echo "=== Step 2: Renaming trajectory files ==="
RAW_H5="$TEST_VIDEOS_DIR/trajectory.h5"
RAW_JSON="$TEST_VIDEOS_DIR/trajectory.json"
RENAMED_H5="$TEST_VIDEOS_DIR/trajectory.none.${CONTROL_MODE}.physx_cuda.h5"
RENAMED_JSON="$TEST_VIDEOS_DIR/trajectory.none.${CONTROL_MODE}.physx_cuda.json"

if [ ! -f "$RAW_H5" ]; then
    echo "ERROR: Expected trajectory file not found: $RAW_H5" >&2
    exit 1
fi

mv "$RAW_H5" "$RENAMED_H5"
mv "$RAW_JSON" "$RENAMED_JSON"
echo "Renamed: $RAW_H5 -> $RENAMED_H5"
echo "Renamed: $RAW_JSON -> $RENAMED_JSON"

# ============================================================================
# Step 3: Replay trajectories with RGB observations
# ============================================================================
echo ""
echo "=== Step 3: Replaying trajectories with RGB observations ==="
RGB_H5="$TEST_VIDEOS_DIR/trajectory.rgb.${CONTROL_MODE}.physx_cuda.h5"

python -m mani_skill.trajectory.replay_trajectory \
    --traj-path "${RENAMED_H5}" \
    -o rgb \
    -b physx_cuda \
    -n 16 \
    --save-traj \
    --use-env-states \
    --count ${REPLAY_COUNT}

if [ $? -ne 0 ]; then
    echo "ERROR: replay_trajectory failed." >&2
    exit 1
fi

# ============================================================================
# Step 4: Convert .h5 dataset to .zarr format
# ============================================================================
echo ""
echo "=== Step 4: Converting .h5 to .zarr ==="
python diffusion_policy/h5_to_zarr.py \
    --h5 "${RGB_H5}" \
    --output "${ZARR_OUTPUT}"

if [ $? -ne 0 ]; then
    echo "ERROR: h5_to_zarr conversion failed." >&2
    exit 1
fi

echo ""
echo "=== Pipeline complete ==="
echo "Zarr dataset saved to: $ZARR_OUTPUT"
