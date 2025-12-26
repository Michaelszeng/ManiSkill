"""
Evaluate a trained diffusion policy model using
https://github.com/Michaelszeng/diffusion-policy-experiments in ManiSkill.

Currently, this runs the policy for 100 trials at each action horizons and logs the
success rates for each. Shows a success rate vs action horizon plot at the end.
"""

import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import dill
import gymnasium as gym
import hydra
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.stats.proportion as smp
import torch
from inference_helpers import DiffusionPolicy
from matplotlib.ticker import ScalarFormatter

# Import ManiSkill to register environments
import mani_skill.envs
from mani_skill.utils.wrappers.record import RecordEpisode

# Add your diffusion policy repo to Python path
DIFFUSION_POLICY_PATH = Path("~/diffusion-policy").expanduser()
sys.path.insert(0, str(DIFFUSION_POLICY_PATH))
from diffusion_policy.workspace.base_workspace import BaseWorkspace


# -----------------------------------------------------------------------------
# Data structures and plotting constants
# -----------------------------------------------------------------------------
@dataclass
class HorizonResult:
    """Results for a single action horizon with a specific checkpoint."""

    horizon: int
    success_rate: float
    num_trials: int
    num_successes: int
    checkpoint_path: str
    checkpoint_name: str


# Configuration
ENV_ID = "Planar-PushT-v1"
NUM_TRIALS_PER_HORIZON = 1000
CONTROL_MODE = "pd_ee_delta_pose"
OBS_MODE = "rgbd"
# Must match training configuration
STATE_MODE = "qpos_qvel"  # "qpos", "qpos_qvel", "tcp_pose"

CHECKPOINTS = [
    "/home/michzeng/diffusion-policy/data/outputs/maniskill/2_obs/checkpoints/epoch=075-val_loss=0.1215-val_ddim_mse=0.747608.ckpt",
]
ACTION_HORIZONS = [1, 2, 3, 4, 5, 6, 7, 8]

# Set to True for fast evaluation (no rendering/video), False for human visualization
FAST_MODE = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def make_action_horizon_plot(
    results: List[HorizonResult], plot_name: str = "Action Horizon Comparison", dpi: int = 150
) -> plt.Figure:
    """Create plot showing success rate vs action horizon with checkpoint labels."""
    # Plotting style constants
    NAVY = "#1f3a68"
    GRID_COLOR = "#e0e0e0"

    def compute_wilson_ci(success_rates: np.ndarray, num_trials: np.ndarray) -> np.ndarray:
        """Compute Wilson 95% confidence intervals and return as yerr format."""
        ci_bounds = np.array(
            [
                smp.proportion_confint(int(p * n), n, alpha=0.05, method="wilson")
                for p, n in zip(success_rates, num_trials)
            ],
            dtype=float,
        )
        ci_lo = ci_bounds[:, 0]
        ci_hi = ci_bounds[:, 1]
        # Return distances from central value for error bars
        return np.vstack([np.clip(success_rates - ci_lo, 0, 1), np.clip(ci_hi - success_rates, 0, 1)])

    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    ax.set_facecolor("white")

    if not results:
        print("Warning: No results to plot.")
        return fig

    # Sort by horizon
    results = sorted(results, key=lambda r: r.horizon)

    horizons = np.array([res.horizon for res in results], dtype=float)
    success_rates = np.array([res.success_rate for res in results], dtype=float)
    num_trials = np.array([res.num_trials for res in results], dtype=int)

    # Compute error bars
    yerr = compute_wilson_ci(success_rates, num_trials)

    # Main line with markers
    ax.plot(
        horizons,
        success_rates,
        color=NAVY,
        linewidth=1.5,
        marker="o",
        markersize=4,
        markeredgecolor="white",
        markeredgewidth=0.8,
        zorder=3,
    )

    # Error bars with 95% Wilson CI
    ax.errorbar(
        horizons,
        success_rates,
        yerr=yerr,
        fmt="none",
        ecolor=NAVY,
        elinewidth=1.0,
        capsize=4.0,
        capthick=1.0,
        alpha=0.9,
        zorder=2,
    )

    # Add checkpoint labels for each point
    for res in results:
        # Extract checkpoint name from path (e.g., "epoch=075-val_loss=0.1215...")
        ckpt_path = Path(res.checkpoint_path)
        ckpt_name = ckpt_path.stem if ckpt_path.suffix == ".ckpt" else ckpt_path.name

        # Truncate checkpoint name to first 10 chars for readability
        label_text = ckpt_name[:10] + "..." if len(ckpt_name) > 10 else ckpt_name

        # Position label slightly above the data point
        ax.annotate(
            label_text,
            xy=(res.horizon, res.success_rate),
            xytext=(0, 5),  # 5 points above
            textcoords="offset points",
            fontsize=6,
            color=NAVY,
            ha="center",
            va="bottom",
            alpha=0.8,
        )

    # Configure x-axis
    ax.set_xticks(horizons)
    ax.set_xticklabels([str(int(val)) if float(val).is_integer() else f"{val:g}" for val in horizons])

    # Configure y-axis
    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.linspace(0, 1.0, 6))

    # Labels and title
    ax.set_title(plot_name, fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Action Horizon (steps)", fontsize=12)
    ax.set_ylabel("Success Rate", fontsize=12)

    # Grid styling
    ax.grid(True, which="major", color=GRID_COLOR, linestyle="-", linewidth=0.8, alpha=0.6)

    # Spine styling
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color("#4f4f4f")

    # Tick styling
    ax.tick_params(axis="both", which="major", labelsize=8, length=6, width=1)

    fig.tight_layout()
    fig.set_dpi(dpi)

    return fig


def main():
    # Base seed for reproducibility
    SEED = 1
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)

    # Enable interactive plotting mode
    plt.ion()

    # Create environment
    print(f"Creating environment: {ENV_ID}")
    render_mode = None if FAST_MODE else "human"
    print(f"Mode: {'FAST (no rendering)' if FAST_MODE else 'HUMAN (with visualization)'}")

    env = gym.make(
        ENV_ID,
        num_envs=16,
        obs_mode=OBS_MODE,
        render_mode=render_mode,
        control_mode=CONTROL_MODE,
        sim_backend="physx_cuda",
        max_episode_steps=200,
        intersection_thresh=0.75,
    )

    # Only record videos in non-fast mode if rendering is disabled
    if not FAST_MODE and env.render_mode != "human":
        env = RecordEpisode(env, output_dir="./videos", save_video=True)

    # Get number of parallel environments
    num_envs = getattr(env, "num_envs", 1)
    print(f"Running with {num_envs} parallel environment(s)")

    # Store results: checkpoint -> action_horizon -> metrics dict
    results = {checkpoint: {} for checkpoint in CHECKPOINTS}

    # Evaluate each checkpoint at each action horizon
    for checkpoint in CHECKPOINTS:
        for action_horizon in ACTION_HORIZONS:
            print(f"\n{'=' * 60}")
            print(f"Evaluating Action Horizon: {action_horizon}")
            print(f"{'=' * 60}")

            # Create policy with current action horizon
            policy = DiffusionPolicy(env, checkpoint, CONTROL_MODE, OBS_MODE, STATE_MODE, action_horizon, DEVICE)

            successes = 0
            total_rewards = []
            total_steps = []
            intersection_ratios = []

            # Run trials for this action horizon
            trials_completed = 0
            while trials_completed < NUM_TRIALS_PER_HORIZON:
                # Seed each episode deterministically
                episode_seed = SEED + trials_completed

                if (trials_completed + 1) % 10 == 0:
                    print(f"  Trial {trials_completed + 1}/{NUM_TRIALS_PER_HORIZON}...")

                result = policy.run_one_episode(episode_seed)

                # Handle both single env and vectorized env results
                if num_envs == 1:
                    # Single environment: results are scalars
                    episode_reward, success, intersection_ratio, steps = result
                    results_list = [(episode_reward, success, intersection_ratio, steps)]
                else:
                    # Vectorized environments: results are lists
                    episode_rewards, successes_batch, intersection_ratios_batch, steps_batch = result
                    results_list = list(zip(episode_rewards, successes_batch, intersection_ratios_batch, steps_batch))

                # Process results from this batch
                for episode_reward, success, intersection_ratio, steps_val in results_list:
                    if trials_completed >= NUM_TRIALS_PER_HORIZON:
                        break

                    if success:
                        successes += 1

                    total_rewards.append(episode_reward)
                    total_steps.append(steps_val)
                    intersection_ratios.append(intersection_ratio)

                    trials_completed += 1

            # Calculate statistics
            success_rate = successes / NUM_TRIALS_PER_HORIZON
            avg_reward = np.mean(total_rewards)
            avg_steps = np.mean(total_steps)
            avg_intersection = np.mean(intersection_ratios)

            results[checkpoint][action_horizon] = {
                "success_rate": success_rate,
                "avg_reward": avg_reward,
                "avg_steps": avg_steps,
                "avg_intersection": avg_intersection,
                "num_successes": successes,
            }

            print(f"\nResults for Action Horizon {action_horizon}:")
            print(f"  Success Rate: {success_rate:.2%} ({successes}/{NUM_TRIALS_PER_HORIZON})")
            print(f"  Average Reward: {avg_reward:.3f}")
            print(f"  Average Steps: {avg_steps:.1f}")
            print(f"  Average Final Intersection: {avg_intersection:.4f}")

    env.close()

    # Select best checkpoint for each action horizon
    best_results: List[HorizonResult] = []
    for horizon in ACTION_HORIZONS:
        best_checkpoint = None
        best_success_rate = -1
        best_metrics = None

        # Find checkpoint with highest success rate for this horizon
        for checkpoint in CHECKPOINTS:
            metrics = results[checkpoint].get(horizon)
            if metrics and metrics["success_rate"] > best_success_rate:
                best_success_rate = metrics["success_rate"]
                best_checkpoint = checkpoint
                best_metrics = metrics

        if best_checkpoint is not None and best_metrics is not None:
            checkpoint_name = Path(best_checkpoint).stem
            horizon_result = HorizonResult(
                horizon=horizon,
                success_rate=best_metrics["success_rate"],
                num_trials=NUM_TRIALS_PER_HORIZON,
                num_successes=best_metrics["num_successes"],
                checkpoint_path=best_checkpoint,
                checkpoint_name=checkpoint_name,
            )
            best_results.append(horizon_result)

    # Print summary
    print(f"\n{'=' * 60}")
    print("SUMMARY - Best Checkpoint per Action Horizon")
    print(f"{'=' * 60}")
    print(f"{'Horizon':<10} {'Success Rate':<15} {'Successes':<15} {'Checkpoint'}")
    print("-" * 60)
    for res in best_results:
        print(
            f"{res.horizon:<10} {res.success_rate:.2%}{'':<10} "
            f"{res.num_successes}/{res.num_trials}{'':<8} {res.checkpoint_name[:40]}"
        )
    print(f"{'=' * 60}")

    # Create and save plot
    fig = make_action_horizon_plot(best_results, plot_name="Success Rate vs Action Horizon (Best Checkpoints)", dpi=150)
    output_path = "success_rate_vs_action_horizon.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to: {output_path}")
    plt.show()

    print("\nDone!")


if __name__ == "__main__":
    main()
