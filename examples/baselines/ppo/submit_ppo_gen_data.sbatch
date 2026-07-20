#!/bin/bash
#SBATCH --job-name=ppo_gen_data
#SBATCH --account=<>
#SBATCH --partition=<>
#SBATCH --exclude=<>
#SBATCH --qos=<>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --requeue

# Usage:
#   sbatch examples/baselines/ppo/submit_ppo_gen_data.sbatch
#
# SLURM wrapper around examples/baselines/ppo/gen_data.sh.
# Handles all environment setup, then hands off to the data-generation script.
# Expect this to take ~30 min to run.

echo "SLURM Job ID:         $SLURM_JOB_ID"
echo "SLURM Node:           $HOSTNAME"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "SLURM_SUBMIT_DIR:     $SLURM_SUBMIT_DIR"

# --- Working directory ---
echo "Changing directory to $SLURM_SUBMIT_DIR"
cd "$SLURM_SUBMIT_DIR" || { echo "Failed to change directory to $SLURM_SUBMIT_DIR"; exit 1; }

# --- Conda setup ---
conda activate PATH-TO-CONDA-ENV

mkdir -p logs

export PYTHONPATH=$PWD:$PYTHONPATH

bash examples/baselines/ppo/gen_data.sh
