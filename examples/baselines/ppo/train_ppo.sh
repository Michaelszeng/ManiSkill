#!/bin/bash
set -euo pipefail

# Usage:
#   examples/baselines/ppo/train_ppo.sh [env_id] [checkpoint]
#
# PPO training runner for ManiSkill environments.

# Defaults to Planar-PushT-v1 if no env_id is provided.
# If a checkpoint path is provided, training warm-starts from those weights
# (note: optimizer state and global step are NOT restored).

ENV_ID="${1:-Planar-PushT-v1}"
CHECKPOINT="${2:-}"

# ============================================================================
# Training parameters
# ============================================================================
NUM_ENVS=4096
NUM_STEPS=16
UPDATE_EPOCHS=8
NUM_MINIBATCHES=32
GAMMA=0.99
TOTAL_TIMESTEPS=160_000_000
NUM_EVAL_STEPS=400
NUM_EVAL_ENVS=16
CONTROL_MODE="pd_ee_delta_pose"
ENT_COEF=0.015
# ============================================================================

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
echo "CHECKPOINT:        ${CHECKPOINT:-<none>}"
echo "=========================================="

CHECKPOINT_ARG=()
if [ -n "$CHECKPOINT" ]; then
    if [ ! -f "$CHECKPOINT" ]; then
        echo "ERROR: checkpoint file not found: $CHECKPOINT" >&2
        exit 1
    fi
    CHECKPOINT_ARG=(--checkpoint="${CHECKPOINT}")
fi

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
    "${CHECKPOINT_ARG[@]}" \
    --cudagraphs
