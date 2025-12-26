"""
Run a trained diffusion policy model using https://github.com/Michaelszeng/diffusion-policy-experiments in ManiSkill.
"""

import sys
from collections import deque
from pathlib import Path

import dill
import gymnasium as gym
import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from inference_helpers import DiffusionPolicy

# Import ManiSkill to register environments
import mani_skill.envs
from mani_skill.utils.wrappers.record import RecordEpisode

# Add your diffusion policy repo to Python path
DIFFUSION_POLICY_PATH = Path("~/diffusion-policy").expanduser()
sys.path.insert(0, str(DIFFUSION_POLICY_PATH))
from diffusion_policy.workspace.base_workspace import BaseWorkspace

# Configuration
ENV_ID = "Planar-PushT-v1"
NUM_FAILURES_TO_STOP = 10
CHECKPOINT = "/home/michzeng/diffusion-policy/data/outputs/maniskill/2_obs/checkpoints/epoch=075-val_loss=0.1215-val_ddim_mse=0.747608.ckpt"
CONTROL_MODE = "pd_ee_delta_pose"
# OBS_MODE = "state"
OBS_MODE = "rgbd"
# Must match training configuration
STATE_MODE = "qpos_qvel"  # "qpos", "qpos_qvel", "tcp_pose"
N_ACTION_STEPS = 8  # Action horizon

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    # Base seed for reproducibility (None to disable)
    # Episode N uses seed = SEED + N
    # To reproduce episode 2, set SEED=42 and run episodes 0,1,2 or set SEED=44 and run 1 episode
    SEED = 1
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)

    # Enable interactive plotting mode
    plt.ion()

    # Create environment
    print(f"Creating environment: {ENV_ID}")
    env = gym.make(
        ENV_ID,
        num_envs=1,
        obs_mode=OBS_MODE,
        render_mode="human",  # This opens a visualization window
        control_mode=CONTROL_MODE,
        sim_backend="physx_cuda",
        max_episode_steps=200,
        intersection_thresh=0.75,
    )
    if env.render_mode != "human":
        env = RecordEpisode(env, output_dir="./videos", save_video=True)

    policy = DiffusionPolicy(env, CHECKPOINT, CONTROL_MODE, OBS_MODE, STATE_MODE, N_ACTION_STEPS, DEVICE)

    # Run episodes
    failed_episodes = []
    episode = 0
    while len(failed_episodes) < NUM_FAILURES_TO_STOP:
        # Seed each episode deterministically (episode 0 uses SEED, episode 1 uses SEED+1, etc.)
        episode_seed = SEED + episode
        print(f"\n=== Episode {episode + 1} (seed={episode_seed}) ===")

        episode_reward, success, intersection_ratio, steps = policy.run_one_episode(episode_seed)

        print(
            f"Episode finished: reward={episode_reward:.3f}, steps={steps}, "
            f"success={success}, final_intersection={intersection_ratio:.4f} "
            f"(need ≥0.90), seed={episode_seed}"
        )

        if not success:
            failed_episodes.append(episode_seed)

        episode += 1

    env.close()
    print("\nDone!")
    print(f"Failed episodes: {failed_episodes}")


if __name__ == "__main__":
    main()
