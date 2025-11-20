"""
DAGGER Step 2: Interactive trajectory review with time scrubbing

This script allows you to:
- Load recorded failed episodes
- Scrub through time using keyboard controls
- Mark intervention points where expert should take over
- Save intervention points for expert data collection
- Skip already-reviewed episodes (--unreviewed-only, default: on)

Controls:
  - Space: Play/Pause
  - Right Arrow / D: Step forward 1 timestep
  - Left Arrow / A: Step backward 1 timestep
  - Up Arrow / W: Skip forward 10 timesteps
  - Down Arrow / S: Skip backward 10 timesteps
  - M: Mark current timestep as intervention point
  - U: Unmark current timestep
  - N: Next episode
  - P: Previous episode
  - R: Restart current episode from beginning
  - H: Show help
  - Q: Quit and save intervention points
"""

import argparse
import json
from pathlib import Path

import gymnasium as gym
import h5py
import numpy as np

import mani_skill.envs
from mani_skill.trajectory.dataset import load_h5_data
from mani_skill.trajectory.utils import dict_to_list_of_dicts

OUTPUT_DIR = "~/diffusion-policy/dagger_data/maniskill/failures"


def parse_args():
    parser = argparse.ArgumentParser(description="Interactive trajectory review for DAGGER")
    parser.add_argument(
        "--traj-path",
        type=str,
        default=Path(OUTPUT_DIR).expanduser() / "trajectory.h5",
        help="Path to trajectory .h5 file or directory containing trajectory.h5 (default: %(default)s)",
    )
    parser.add_argument(
        "--unreviewed-only",
        action="store_true",
        default=True,
        help="Only show episodes without intervention points (default: True)",
    )
    parser.add_argument(
        "--no-unreviewed-only",
        dest="unreviewed_only",
        action="store_false",
        help="Show all episodes, including those already reviewed",
    )
    args = parser.parse_args()
    return args


class InteractiveTrajectoryReviewer:
    def __init__(self, traj_path: str, unreviewed_only: bool = True):
        traj_path = Path(traj_path).expanduser()

        # If traj_path is a directory, assume trajectory.h5 is inside
        if traj_path.is_dir():
            traj_path = traj_path / "trajectory.h5"

        self.traj_path = traj_path
        self.json_path = self.traj_path.with_suffix(".json")

        # Load trajectory data
        print(f"Loading trajectory from: {self.traj_path}")
        with open(self.json_path, "r") as f:
            self.metadata = json.load(f)

        self.h5_file = h5py.File(self.traj_path, "r")
        all_episodes = self.metadata["episodes"]

        print(f"Loaded {len(all_episodes)} episodes")

        # Create environment
        env_id = self.metadata["env_info"]["env_id"]
        env_kwargs = self.metadata["env_info"]["env_kwargs"].copy()
        env_kwargs["render_mode"] = "human"  # Override render_mode for interactive viewing
        self.env = gym.make(env_id, **env_kwargs)
        self.viewer = None

        # State
        self.current_episode_idx = 0
        self.current_timestep = 0
        self.playing = False
        self.intervention_points = {}  # {episode_id: [timesteps]}

        # Load existing intervention points if they exist
        self.intervention_file = self.traj_path.parent / "intervention_points.json"
        if self.intervention_file.exists():
            with open(self.intervention_file, "r") as f:
                loaded = json.load(f)
                self.intervention_points = {int(k): v for k, v in loaded["intervention_points"].items()}
            print(f"Loaded existing intervention points from: {self.intervention_file}")

        # Print overview of labeled vs unlabeled episodes
        print("\n" + "=" * 80)
        print("EPISODE OVERVIEW")
        print("=" * 80)

        labeled_episodes = {ep["episode_id"]: ep for ep in all_episodes if ep["episode_id"] in self.intervention_points}
        unlabeled_episodes = {
            ep["episode_id"]: ep for ep in all_episodes if ep["episode_id"] not in self.intervention_points
        }

        if labeled_episodes:
            print(f"\n✅ Labeled Episodes ({len(labeled_episodes)}):")
            for ep_id, ep in sorted(labeled_episodes.items()):
                num_interventions = len(self.intervention_points[ep_id])
                print(f"   - Episode {ep_id} (seed {ep['episode_seed']}): {num_interventions} intervention point(s)")

        if unlabeled_episodes:
            print(f"\n❌ Unlabeled Episodes ({len(unlabeled_episodes)}):")
            for ep_id, ep in sorted(unlabeled_episodes.items()):
                print(f"   - Episode {ep_id} (seed {ep['episode_seed']})")

        print("\n" + "=" * 80 + "\n")

        # Filter episodes based on unreviewed_only flag
        if unreviewed_only and self.intervention_points:
            # Only show episodes that don't have intervention points yet
            self.episodes = [ep for ep in all_episodes if ep["episode_id"] not in self.intervention_points]
            if len(self.episodes) == 0:
                print("✅ All episodes have been reviewed!")
                print(f"   Total episodes: {len(all_episodes)}")
                print(f"   Reviewed: {len(self.intervention_points)}")
                print("\nTo review all episodes again, use: --no-unreviewed-only")
            else:
                print(f"Showing {len(self.episodes)} unreviewed episodes (out of {len(all_episodes)} total)")
                print(f"Already reviewed: {len(self.intervention_points)} episodes")
                print("To review all episodes, use: --no-unreviewed-only")
        else:
            self.episodes = all_episodes
            if self.intervention_points:
                print(f"Showing all {len(all_episodes)} episodes ({len(self.intervention_points)} already reviewed)")

    def load_episode(self, episode_idx):
        """Load an episode and reset environment"""
        self.current_episode_idx = episode_idx
        self.current_timestep = 0

        episode = self.episodes[episode_idx]
        episode_id = episode["episode_id"]

        print(f"\n{'=' * 80}")
        print(f"Episode {episode_idx + 1}/{len(self.episodes)} (ID: {episode_id})")
        print(f"Steps: {episode['elapsed_steps']}, Control mode: {episode['control_mode']}")
        print(f"{'=' * 80}")

        # Load trajectory data
        traj_key = f"traj_{episode_id}"
        self.current_traj = self.h5_file[traj_key]
        self.current_episode_id = episode_id

        # Convert env_states from dict-of-arrays to list-of-dicts for indexing
        env_states_dict = load_h5_data(self.current_traj["env_states"])
        self.env_states = dict_to_list_of_dicts(env_states_dict)

        self.max_timestep = len(self.env_states) - 1

        # Get intervention points for this episode
        self.current_interventions = self.intervention_points.get(episode_id, [])

        # Reset environment with the episode seed
        self.env.reset(**episode["reset_kwargs"])

        # Set to first state
        self.set_timestep(0)

    def set_timestep(self, timestep):
        """Set environment to a specific timestep"""
        timestep = max(0, min(timestep, self.max_timestep))
        self.current_timestep = timestep

        # Set environment state
        self.env.base_env.set_state_dict(self.env_states[timestep])

        # Render
        self.env.render()

        # Print status - pad with spaces to clear any leftover characters
        intervention_marker = " [INTERVENTION MARKED]" if timestep in self.current_interventions else ""
        episode_info = f"Episode: {self.current_episode_idx + 1}/{len(self.episodes)}"
        status_line = f"Timestep: {timestep:4d}/{self.max_timestep} | {episode_info}{intervention_marker}"
        # Pad to 100 characters to clear any previous text
        print(f"\r{status_line:<100}", end="", flush=True)

        # Get viewer if not already
        if self.viewer is None:
            self.viewer = self.env.base_env.render_human()

    def mark_intervention(self):
        """Mark current timestep as intervention point"""
        if self.current_timestep not in self.current_interventions:
            self.current_interventions.append(self.current_timestep)
            self.current_interventions.sort()
            self.intervention_points[self.current_episode_id] = self.current_interventions
            print(f"\n✓ Marked timestep {self.current_timestep} as intervention point")
            print(f"  Total intervention points for this episode: {len(self.current_interventions)}")
        else:
            print(f"\n⚠ Timestep {self.current_timestep} already marked")

    def unmark_intervention(self):
        """Unmark current timestep"""
        if self.current_timestep in self.current_interventions:
            self.current_interventions.remove(self.current_timestep)
            self.intervention_points[self.current_episode_id] = self.current_interventions
            if len(self.current_interventions) == 0:
                del self.intervention_points[self.current_episode_id]
            print(f"\n✓ Unmarked timestep {self.current_timestep}")
        else:
            print(f"\n⚠ Timestep {self.current_timestep} not marked")

    def save_intervention_points(self):
        """Save intervention points to JSON file"""
        # Convert to serializable format
        save_data = {
            "trajectory_path": str(self.traj_path),
            "intervention_points": self.intervention_points,
            "total_episodes": len(self.episodes),
            "total_interventions": sum(len(v) for v in self.intervention_points.values()),
        }

        with open(self.intervention_file, "w") as f:
            json.dump(save_data, f, indent=2)

        print(f"\n✓ Saved intervention points to: {self.intervention_file}")
        print(f"  Total episodes with interventions: {len(self.intervention_points)}")
        print(f"  Total intervention points: {save_data['total_interventions']}")

    def show_help(self):
        """Print help message"""
        print("\n" + "=" * 80)
        print("CONTROLS:")
        print("=" * 80)
        print("  Space       : Play/Pause automatic playback")
        print("  Right / D   : Step forward 1 timestep")
        print("  Left / A    : Step backward 1 timestep")
        print("  Up / W      : Skip forward 10 timesteps")
        print("  Down / S    : Skip backward 10 timesteps")
        print("  M           : Mark current timestep as intervention point")
        print("  U           : Unmark current timestep")
        print("  L           : List all intervention points for current episode")
        print("  N           : Next episode")
        print("  P           : Previous episode")
        print("  R           : Restart current episode from beginning")
        print("  H           : Show this help")
        print("  Q           : Quit and save intervention points")
        print("=" * 80 + "\n")

    def list_interventions(self):
        """List all intervention points for current episode"""
        print("\n" + "-" * 80)
        if len(self.current_interventions) == 0:
            print(f"No intervention points marked for episode {self.current_episode_id}")
        else:
            print(f"Intervention points for episode {self.current_episode_id}:")
            for i, t in enumerate(self.current_interventions, 1):
                print(f"  {i}. Timestep {t}")
        print("-" * 80 + "\n")

    def run(self):
        """Run interactive review loop"""
        # Check if there are episodes to review
        if len(self.episodes) == 0:
            print("\nNo episodes to review. Exiting.")
            self.h5_file.close()
            self.env.close()
            return

        self.load_episode(0)
        self.show_help()

        print("\nReviewing trajectories. Press H for help.")

        running = True
        while running:
            # Handle viewer input
            if self.viewer is not None:
                # Step forward/backward
                if self.viewer.window.key_press("right") or self.viewer.window.key_press("d"):
                    self.set_timestep(self.current_timestep + 1)
                elif self.viewer.window.key_press("left") or self.viewer.window.key_press("a"):
                    self.set_timestep(self.current_timestep - 1)

                # Skip forward/backward
                elif self.viewer.window.key_press("up") or self.viewer.window.key_press("w"):
                    self.set_timestep(self.current_timestep + 10)
                elif self.viewer.window.key_press("down") or self.viewer.window.key_press("s"):
                    self.set_timestep(self.current_timestep - 10)

                # Mark/unmark intervention
                elif self.viewer.window.key_press("m"):
                    self.mark_intervention()
                elif self.viewer.window.key_press("u"):
                    self.unmark_intervention()
                elif self.viewer.window.key_press("l"):
                    self.list_interventions()

                # Episode navigation
                elif self.viewer.window.key_press("n"):
                    if self.current_episode_idx < len(self.episodes) - 1:
                        self.load_episode(self.current_episode_idx + 1)
                    else:
                        print("\n⚠ Already at last episode")
                elif self.viewer.window.key_press("p"):
                    if self.current_episode_idx > 0:
                        self.load_episode(self.current_episode_idx - 1)
                    else:
                        print("\n⚠ Already at first episode")
                elif self.viewer.window.key_press("r"):
                    self.set_timestep(0)

                # Play/pause
                elif self.viewer.window.key_press("space"):
                    self.playing = not self.playing
                    status = "Playing" if self.playing else "Paused"
                    print(f"\n{status}")

                # Help
                elif self.viewer.window.key_press("h"):
                    self.show_help()

                # Quit
                elif self.viewer.window.key_press("q"):
                    print("\n\nQuitting...")
                    running = False

                # Auto-play
                if self.playing:
                    self.set_timestep(self.current_timestep + 1)
                    if self.current_timestep >= self.max_timestep:
                        self.playing = False
                        print("\n⚠ Reached end of episode")

                # Must call render to process events
                self.env.render()

        # Save intervention points
        self.save_intervention_points()

        # Cleanup
        self.h5_file.close()
        self.env.close()

        print("\n" + "=" * 80)
        print("Review Complete!")
        print("=" * 80)
        if len(self.intervention_points) > 0:
            print("\nNext step: Collect expert demonstrations")
            print("  python diffusion_policy/dagger_collect_expert.py \\")
            print(f"    --traj-path {self.traj_path}")
        else:
            print("\nNo intervention points marked.")


def main():
    args = parse_args()
    reviewer = InteractiveTrajectoryReviewer(args.traj_path, unreviewed_only=args.unreviewed_only)
    reviewer.run()


if __name__ == "__main__":
    main()
