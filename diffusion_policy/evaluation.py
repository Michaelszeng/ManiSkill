"""
Evaluate a trained diffusion policy model using
https://github.com/Michaelszeng/diffusion-policy-experiments in ManiSkill.

For each (checkpoint, action_horizon) pair this script runs
NUM_TRIALS_PER_HORIZON trials and writes a results.csv / results.pkl / summary.txt directory.
At the end of the grid an overall_summary.txt and overall_results.pkl are written.

Example:
    python diffusion_policy/evaluation.py \
        --checkpoints-dir /path/to/checkpoints \
        --action-horizons 1 2 3 4 5 6 8 10 12 15 \
        --n-video-trials 20 \
        --output-dir outputs/maniskill/2_obs/eval
"""

import argparse
import csv
import datetime
import math
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

import dill
import gymnasium as gym
import imageio
import numpy as np
import torch
from inference_helpers import DiffusionPolicy

# Import ManiSkill to register environments
import mani_skill.envs
from mani_skill.utils.wrappers.record import RecordEpisode

# Add your diffusion policy repo to Python path
DIFFUSION_POLICY_PATH = Path("~/diffusion-policy").expanduser()
sys.path.insert(0, str(DIFFUSION_POLICY_PATH))
from diffusion_policy.workspace.base_workspace import BaseWorkspace  # noqa: F401


# -----------------------------------------------------------------------------
# Data structures
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


# -----------------------------------------------------------------------------
# Fixed configuration (kept as constants — change here, not via CLI)
# -----------------------------------------------------------------------------
ENV_ID = "Planar-PushT-v1"
NUM_TRIALS_PER_HORIZON = 500
CONTROL_MODE = "pd_ee_delta_pose"
OBS_MODE = "rgbd"
# Must match training configuration
STATE_MODE = "qpos_qvel"  # "qpos", "qpos_qvel", "tcp_pose"
MAX_EPISODE_STEPS = 200
NUM_ENVS = 16
INTERSECTION_THRESH = 0.75

CSV_FIELDS = ["trial", "result", "reward", "trial_time"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--checkpoints-dir",
        type=str,
        default="/home/michzeng/diffusion-policy/data/outputs/maniskill/2_obs/checkpoints",
        help="directory containing .ckpt files (all will be evaluated)",
    )
    p.add_argument("--action-horizons", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 7, 8])
    p.add_argument("--seed", type=int, default=1)
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="directory to write results (default: <checkpoints_dir>.parent/eval/<date>/<time>/)",
    )
    p.add_argument(
        "--fast-mode", dest="fast_mode", action="store_true", default=True, help="skip rendering for speed (default)"
    )
    p.add_argument(
        "--no-fast-mode",
        dest="fast_mode",
        action="store_false",
        help="open the SAPIEN human viewer (incompatible with video recording)",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="skip (checkpoint, horizon) pairs whose results.pkl already has all trials done",
    )
    p.add_argument(
        "--n-video-trials",
        type=int,
        default=0,
        help="save MP4 videos for the first N trials per (ckpt, horizon). 0=off, -1=all",
    )
    p.add_argument("--video-fps", type=int, default=20)
    return p.parse_args()


# -----------------------------------------------------------------------------
# IO helpers
# -----------------------------------------------------------------------------
def _classify(success: bool, steps: int, max_episode_steps: int) -> str:
    if success:
        return "success"
    return "timeout" if steps >= max_episode_steps else "failure"


def _write_mp4(frames: list, path: Path, fps: int) -> None:
    with imageio.get_writer(path, fps=fps, codec="libx264", pixelformat="yuv420p") as w:
        for frame in frames:
            w.append_data(frame)


def _write_summary(
    path: Path,
    *,
    checkpoint: str,
    action_horizon: int,
    n_success: int,
    n_total: int,
    n_target: int,
    trial_records: list,
) -> None:
    n_failure = sum(1 for r in trial_records if r["result"] == "failure")
    n_timeout = sum(1 for r in trial_records if r["result"] == "timeout")
    rate = n_success / n_total if n_total > 0 else 0.0
    avg_steps = sum(r["trial_time"] for r in trial_records) / len(trial_records) if trial_records else 0.0
    with open(path, "w") as f:
        f.write(f"Checkpoint       : {checkpoint}\n")
        f.write(f"Action horizon   : {action_horizon}\n")
        f.write(f"Trials completed : {n_total} / {n_target}\n")
        f.write(f"Successes        : {n_success}\n")
        f.write(f"Failures         : {n_failure}\n")
        f.write(f"Timeouts         : {n_timeout}\n")
        f.write(f"Success rate     : {rate:.1%}\n")
        f.write(f"Avg trial steps  : {avg_steps:.1f}\n")


def _write_pkl(path: Path, *, trials, n_success, n_total, checkpoint, action_horizon) -> None:
    with open(path, "wb") as f:
        pickle.dump(
            {
                "trials": trials,
                "n_success": n_success,
                "n_total": n_total,
                "success_rate": n_success / n_total if n_total > 0 else 0.0,
                "checkpoint": checkpoint,
                "action_horizon": action_horizon,
                "env_id": ENV_ID,
                "control_mode": CONTROL_MODE,
                "obs_mode": OBS_MODE,
                "state_mode": STATE_MODE,
            },
            f,
        )


def _load_existing_run(run_dir: Path):
    """Return (trials, n_success, n_total) from a prior results.pkl, or None."""
    pkl_path = run_dir / "results.pkl"
    if not pkl_path.exists():
        return None
    try:
        with open(pkl_path, "rb") as f:
            saved = pickle.load(f)
    except Exception:
        return None
    return saved.get("trials", []), saved.get("n_success", 0), saved.get("n_total", 0)


def evaluate_one(
    policy,
    run_dir: Path,
    checkpoint: str,
    action_horizon: int,
    base_seed: int,
    n_video_trials: int,
    video_fps: int,
    resume: bool,
) -> dict:
    """Run NUM_TRIALS_PER_HORIZON trials for one (checkpoint, action_horizon)
    pair, streaming output files into ``run_dir`` after each round. Saves MP4
    videos for the first ``n_video_trials`` trials (-1 = all, 0 = off)."""
    num_trials = NUM_TRIALS_PER_HORIZON
    num_envs = NUM_ENVS
    run_dir.mkdir(parents=True, exist_ok=True)
    csv_path = run_dir / "results.csv"
    pkl_path = run_dir / "results.pkl"
    summary_path = run_dir / "summary.txt"
    videos_dir = run_dir / "videos"

    record_video = n_video_trials != 0
    video_budget = n_video_trials if n_video_trials >= 0 else num_trials
    if record_video:
        videos_dir.mkdir(exist_ok=True)

    # --- resume or start fresh ---
    trial_records: list = []
    n_success = n_total = 0
    if resume:
        prior = _load_existing_run(run_dir)
        if prior is not None and prior[2] >= num_trials:
            print(f"  [resume] {run_dir.name}: already has {prior[2]}/{num_trials} trials, skipping")
            trial_records, n_success, n_total = prior
            return {
                "success_rate": n_success / n_total,
                "num_successes": n_success,
                "num_trials": n_total,
                "trials": trial_records,
            }

    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
    csv_writer.writeheader()
    csv_file.flush()

    n_rounds = max(1, math.ceil(num_trials / num_envs))
    for i in range(n_rounds):
        episode_seed = base_seed + n_total
        # Record this round if any trial in it falls within the video budget.
        record_this_round = record_video and (n_total < video_budget)
        result = policy.run_one_episode(episode_seed, record_video=record_this_round)
        frame_buffers = policy.last_frames if record_this_round else None

        # Normalize to per-env lists
        if num_envs == 1:
            rewards, succs, _inter, steps_l = result
            batch = [(rewards, succs, steps_l)]
        else:
            rewards, succs, _inter, steps_l = result
            batch = list(zip(rewards, succs, steps_l))

        for env_idx, (reward, success, steps) in enumerate(batch):
            if n_total >= num_trials:
                break
            n_total += 1
            result_str = _classify(bool(success), int(steps), MAX_EPISODE_STEPS)
            if success:
                n_success += 1
            record = {
                "trial": n_total,
                "result": result_str,
                "reward": float(reward),
                "trial_time": int(steps),
            }
            trial_records.append(record)
            csv_writer.writerow(record)
            csv_file.flush()

            if record_this_round and n_total <= video_budget and frame_buffers is not None:
                video_path = videos_dir / f"trial_{n_total:04d}_{result_str}.mp4"
                _write_mp4(frame_buffers[env_idx], video_path, fps=video_fps)
                print(f"    saved {video_path.name}")

        # Durable intermediate writes
        _write_summary(
            summary_path,
            checkpoint=checkpoint,
            action_horizon=action_horizon,
            n_success=n_success,
            n_total=n_total,
            n_target=num_trials,
            trial_records=trial_records,
        )
        _write_pkl(
            pkl_path,
            trials=trial_records,
            n_success=n_success,
            n_total=n_total,
            checkpoint=checkpoint,
            action_horizon=action_horizon,
        )

        rate = n_success / n_total if n_total > 0 else 0.0
        print(f"  Round {i + 1}/{n_rounds}: running {n_success}/{n_total} ({rate:.1%})")

    csv_file.close()
    return {
        "success_rate": n_success / num_trials,
        "num_successes": n_success,
        "num_trials": n_total,
        "trials": trial_records,
    }


def main():
    args = parse_args()

    if args.n_video_trials != 0 and not args.fast_mode:
        raise SystemExit(
            "--n-video-trials is incompatible with --no-fast-mode (human viewer): "
            "video recording needs render_mode='rgb_array'."
        )

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # --- discover checkpoints ---
    checkpoints_dir = Path(args.checkpoints_dir)
    if not checkpoints_dir.exists():
        raise ValueError(f"Checkpoints directory does not exist: {args.checkpoints_dir}")
    checkpoints = sorted(checkpoints_dir.glob("*.ckpt"))
    if not checkpoints:
        raise ValueError(f"No .ckpt files found in directory: {args.checkpoints_dir}")
    CHECKPOINTS = [str(ckpt) for ckpt in checkpoints]
    print(f"\nFound {len(CHECKPOINTS)} checkpoint(s) in {args.checkpoints_dir}:")
    for i, ckpt in enumerate(CHECKPOINTS, 1):
        print(f"  {i}. {Path(ckpt).name}")
    print()

    # --- output directory ---
    # Default: a timestamped folder next to checkpoints/, inside the training
    # run dir (so eval results live alongside the run that produced them).
    if args.output_dir is not None:
        out_dir = Path(args.output_dir)
    else:
        now = datetime.datetime.now()
        out_dir = checkpoints_dir.parent / "eval" / now.strftime("%Y-%m-%d") / now.strftime("%H-%M-%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing results to {out_dir}/\n")

    # --- env ---
    if args.n_video_trials != 0:
        render_mode = "rgb_array"
        mode_desc = "VIDEO (rgb_array rendering)"
    elif not args.fast_mode:
        render_mode = "human"
        mode_desc = "HUMAN (with visualization)"
    else:
        render_mode = None
        mode_desc = "FAST (no rendering)"
    print(f"Creating environment: {ENV_ID}")
    print(f"Mode: {mode_desc}")
    env = gym.make(
        ENV_ID,
        num_envs=NUM_ENVS,
        obs_mode=OBS_MODE,
        render_mode=render_mode,
        control_mode=CONTROL_MODE,
        sim_backend="physx_cuda",
        max_episode_steps=MAX_EPISODE_STEPS,
        intersection_thresh=INTERSECTION_THRESH,
    )
    num_envs = getattr(env, "num_envs", 1)
    print(f"Running with {num_envs} parallel environment(s)\n")

    # --- evaluate grid ---
    results = {ckpt: {} for ckpt in CHECKPOINTS}
    for checkpoint in CHECKPOINTS:
        ckpt_stem = Path(checkpoint).stem
        for action_horizon in args.action_horizons:
            print(f"{'=' * 60}")
            print(f"Checkpoint: {ckpt_stem}")
            print(f"Action horizon: {action_horizon}")
            print(f"{'=' * 60}")

            policy = DiffusionPolicy(env, checkpoint, CONTROL_MODE, OBS_MODE, STATE_MODE, action_horizon, DEVICE)
            run_dir = out_dir / f"T_a_{action_horizon}" / ckpt_stem
            metrics = evaluate_one(
                policy=policy,
                run_dir=run_dir,
                checkpoint=checkpoint,
                action_horizon=action_horizon,
                base_seed=args.seed,
                n_video_trials=args.n_video_trials,
                video_fps=args.video_fps,
                resume=args.resume,
            )
            results[checkpoint][action_horizon] = metrics
            print(
                f"  → success rate: {metrics['success_rate']:.2%} "
                f"({metrics['num_successes']}/{metrics['num_trials']})\n"
            )

    env.close()

    # --- best-per-horizon summary ---
    best_results: List[HorizonResult] = []
    for horizon in args.action_horizons:
        best_ckpt, best_metrics, best_rate = None, None, -1.0
        for ckpt in CHECKPOINTS:
            m = results[ckpt].get(horizon)
            if m and m["success_rate"] > best_rate:
                best_rate, best_ckpt, best_metrics = m["success_rate"], ckpt, m
        if best_ckpt is not None:
            best_results.append(
                HorizonResult(
                    horizon=horizon,
                    success_rate=best_metrics["success_rate"],
                    num_trials=best_metrics["num_trials"],
                    num_successes=best_metrics["num_successes"],
                    checkpoint_path=best_ckpt,
                    checkpoint_name=Path(best_ckpt).stem,
                )
            )

    overall_summary_path = out_dir / "overall_summary.txt"
    with open(overall_summary_path, "w") as f:
        f.write("SUMMARY - Best Checkpoint per Action Horizon\n")
        f.write(f"{'Horizon':<10} {'Success Rate':<15} {'Successes':<15} Checkpoint\n")
        f.write("-" * 60 + "\n")
        for r in best_results:
            f.write(
                f"{r.horizon:<10} {r.success_rate:.2%}{'':<10} "
                f"{r.num_successes}/{r.num_trials}{'':<8} {r.checkpoint_name}\n"
            )

    overall_pkl_path = out_dir / "overall_results.pkl"
    with open(overall_pkl_path, "wb") as f:
        dill.dump(
            {
                "best_results": best_results,
                "full_results": results,
                "config": {
                    "env_id": ENV_ID,
                    "num_trials_per_horizon": NUM_TRIALS_PER_HORIZON,
                    "control_mode": CONTROL_MODE,
                    "obs_mode": OBS_MODE,
                    "state_mode": STATE_MODE,
                    "max_episode_steps": MAX_EPISODE_STEPS,
                    "num_envs": NUM_ENVS,
                    "intersection_thresh": INTERSECTION_THRESH,
                    **vars(args),
                },
            },
            f,
        )

    print(f"\n{'=' * 60}")
    print("SUMMARY - Best Checkpoint per Action Horizon")
    print(f"{'=' * 60}")
    print(f"{'Horizon':<10} {'Success Rate':<15} {'Successes':<15} Checkpoint")
    print("-" * 60)
    for r in best_results:
        print(
            f"{r.horizon:<10} {r.success_rate:.2%}{'':<10} "
            f"{r.num_successes}/{r.num_trials}{'':<8} {r.checkpoint_name[:40]}"
        )
    print(f"{'=' * 60}")
    print(f"\nResults written to {out_dir}/")


if __name__ == "__main__":
    main()
