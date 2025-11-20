"""
DAGGER Step 3: Collect RL expert demonstrations from intervention points

This script:
- Loads failed episodes with marked intervention points
- Runs an RL policy from each intervention point
- If successful, saves the expert trajectory segment (state-action pairs)
"""

import argparse
import json
import math
import sys
from pathlib import Path

import gymnasium as gym
import h5py
import numpy as np
import torch
from ppo_fast import Agent

import mani_skill.envs
from mani_skill.trajectory.dataset import load_h5_data
from mani_skill.trajectory.utils import dict_to_list_of_dicts
from mani_skill.utils.wrappers.record import RecordEpisode

OUTPUT_DIR = "~/diffusion-policy/dagger_data/maniskill/failures"
EXPERT_OUTPUT_DIR = "~/diffusion-policy/dagger_data/maniskill/expert_demos"

RL_CHECKPOINT = "~/.maniskill/demos/PushT-v1/rl/ppo_pd_joint_delta_pos_ckpt.pt"


def parse_args():
    parser = argparse.ArgumentParser(description="Collect expert demonstrations from intervention points")
    parser.add_argument(
        "--traj-path",
        type=str,
        default=Path(OUTPUT_DIR).expanduser() / "trajectory.h5",
        help="Path to trajectory .h5 file with intervention points",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=Path(EXPERT_OUTPUT_DIR).expanduser(),
        help="Directory to save expert demonstrations",
    )
    parser.add_argument(
        "--rl-checkpoint",
        type=str,
        default=Path(RL_CHECKPOINT).expanduser(),
        help="Path to RL policy checkpoint (e.g., PPO checkpoint)",
    )
    args = parser.parse_args()
    return args


def load_rl_policy(checkpoint_path: str, env, device: str = "cuda"):
    """
    Load RL policy from checkpoint.

    Args:
        checkpoint_path: Path to RL checkpoint (e.g., PPO checkpoint)
        env: Environment instance (to get obs/action dimensions)
        device: Device to load model on

    Returns:
        agent: Loaded RL agent
    """
    # Get observation and action dimensions
    n_obs = math.prod(env.single_observation_space.shape)
    n_act = math.prod(env.single_action_space.shape)

    print(f"RL Policy: obs_dim={n_obs}, act_dim={n_act}")

    # Load agent
    agent = Agent(n_obs, n_act, device=torch.device(device))
    checkpoint = torch.load(checkpoint_path, map_location=device)
    agent.load_state_dict(checkpoint)
    agent.eval()

    print(f"Loaded RL policy from: {checkpoint_path}")

    return agent


class DAGGERRLExpertCollector:
    def __init__(
        self,
        traj_path: str,
        rl_checkpoint: str,
        output_dir: str,
    ):
        self.traj_path = Path(traj_path).expanduser()

        # If traj_path is a directory, assume trajectory.h5 is inside
        if self.traj_path.is_dir():
            self.traj_path = self.traj_path / "trajectory.h5"

        self.json_path = self.traj_path.with_suffix(".json")
        self.rl_checkpoint = Path(rl_checkpoint).expanduser()
        self.output_dir = Path(output_dir).expanduser()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load trajectory metadata
        with open(self.json_path, "r") as f:
            self.metadata = json.load(f)

        self.h5_file = h5py.File(self.traj_path, "r")
        self.episodes = self.metadata["episodes"]

        # Load intervention points
        self.intervention_file = self.traj_path.parent / "intervention_points.json"
        if not self.intervention_file.exists():
            raise FileNotFoundError(
                f"Intervention points file not found: {self.intervention_file}\n"
                f"Please run dagger_review_failures.py first to mark intervention points."
            )

        with open(self.intervention_file, "r") as f:
            intervention_data = json.load(f)
            self.intervention_points = {int(k): v for k, v in intervention_data["intervention_points"].items()}

        print(f"Loaded {len(self.intervention_points)} episodes with intervention points")

        # Create environment for RL policy (needs obs_mode="state")
        env_id = self.metadata["env_info"]["env_id"]
        env_kwargs = self.metadata["env_info"]["env_kwargs"].copy()

        # Override to state mode for RL policy
        env_kwargs["obs_mode"] = "state"
        env_kwargs["render_mode"] = "rgb_array"  # For recording

        base_env = gym.make(env_id, **env_kwargs)

        # Wrap with RecordEpisode to save expert demonstrations
        self.env = RecordEpisode(
            base_env,
            output_dir=str(self.output_dir),
            save_trajectory=True,
            trajectory_name="trajectory",
            save_video=True,
            save_on_reset=False,  # Manually save only successful segments
            record_reward=True,
            record_env_state=True,
            source_type="rl_expert",
            source_desc="Expert demonstrations from RL policy via DAGGER from marked intervention points",
        )

        # Load RL policy
        print("Loading RL policy...")
        self.rl_agent = load_rl_policy(self.rl_checkpoint, self.env, device=self.device)

    def collect_from_intervention(self, episode_id: int, intervention_timestep: int):
        """
        Collect expert demonstration from a specific intervention point.

        Args:
            episode_id: Episode ID from trajectory file
            intervention_timestep: Timestep where intervention starts

        Returns:
            success: Whether the expert successfully completed the task
        """
        # Find episode metadata
        episode = next(e for e in self.episodes if e["episode_id"] == episode_id)

        print(f"\n{'=' * 80}")
        print(f"Episode {episode_id}, Intervention at timestep {intervention_timestep}")
        print(f"{'=' * 80}")

        # Reset environment
        self.env.reset(**episode["reset_kwargs"])

        # Load trajectory and set to intervention point
        traj_key = f"traj_{episode_id}"
        traj = self.h5_file[traj_key]

        # Convert env_states
        env_states_dict = load_h5_data(traj["env_states"])
        env_states = dict_to_list_of_dicts(env_states_dict)

        # Set environment to intervention timestep
        self.env.base_env.set_state_dict(env_states[intervention_timestep])

        # Get initial observation after setting state
        # We need to trigger the environment to return the observation
        # This is a bit tricky - we'll step with a zero action to get obs
        obs = self.env.base_env.get_obs()

        print(f"Starting RL expert from timestep {intervention_timestep}...")

        # Run RL policy
        episode_reward = 0
        step = 0
        max_steps = 500  # Safety limit

        terminated = False
        truncated = False

        while not (terminated or truncated) and step < max_steps:
            # Get action from RL policy
            with torch.no_grad():
                action = self.rl_agent.actor_mean(obs)

            # Execute action
            obs, reward, terminated, truncated, info = self.env.step(action)

            episode_reward += reward[0].item() if isinstance(reward, torch.Tensor) else reward
            step += 1

            # Check for success
            if info.get("success", [False])[0] if isinstance(info.get("success"), list) else info.get("success", False):
                terminated = True
                break

        success = (
            info.get("success", [False])[0] if isinstance(info.get("success"), list) else info.get("success", False)
        )

        if success:
            print(f"✅ SUCCESS! Steps: {step}, Reward: {episode_reward:.3f}")
            # Save this trajectory segment
            self.env.flush_trajectory(verbose=True)
            self.env.flush_video(verbose=True)
        else:
            print(f"❌ Failed. Steps: {step}, Reward: {episode_reward:.3f}")
            # Don't save failed attempts

        return success

    def run(self):
        """Run expert collection for all intervention points"""
        total_interventions = sum(len(v) for v in self.intervention_points.values())
        successful = 0
        failed = 0

        print(f"\n{'=' * 80}")
        print("DAGGER RL Expert Collection")
        print(f"{'=' * 80}")
        print(f"Total interventions to collect: {total_interventions}")
        print(f"RL checkpoint: {self.rl_checkpoint}")
        print(f"Output directory: {self.output_dir}")
        print(f"{'=' * 80}\n")

        for episode_id, timesteps in self.intervention_points.items():
            for timestep in timesteps:
                print(f"\nProgress: {successful + failed + 1}/{total_interventions}")

                success = self.collect_from_intervention(episode_id, timestep)

                if success:
                    successful += 1
                else:
                    failed += 1

        # Cleanup
        self.h5_file.close()
        self.env.close()

        print("\n" + "=" * 80)
        print("RL Expert Collection Complete!")
        print("=" * 80)
        print(f"Successful: {successful}/{total_interventions}")
        print(f"Failed: {failed}/{total_interventions}")
        print(f"Success rate: {successful / total_interventions * 100:.1f}%")
        print(f"\nExpert demonstrations saved to: {self.output_dir}")

        if successful > 0:
            print("\nNext steps:")
            print("1. Merge expert demonstrations with original training data:")
            print("   python -m mani_skill.trajectory.merge_trajectory \\")
            print("     --traj-paths \\")
            print("       ~/diffusion-policy/data/original_demos/trajectory.h5 \\")
            print(f"       {self.output_dir}/trajectory.h5 \\")
            print("     --output-path ~/diffusion-policy/data/dagger_iteration_1/trajectory.h5")
            print("2. Retrain diffusion policy on augmented dataset")


def main():
    args = parse_args()
    collector = DAGGERRLExpertCollector(
        traj_path=args.traj_path,
        rl_checkpoint=args.rl_checkpoint,
        output_dir=args.output_dir,
    )
    collector.run()


if __name__ == "__main__":
    main()
