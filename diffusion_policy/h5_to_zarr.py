"""
Convert a ManiSkill Planar-PushT RGB trajectory.h5 to the Drake-style zarr
format used by ~/diffusion-policy (matches sim_sim_tee_data_carbon_large.zarr).

Usage:
    python diffusion_policy/h5_to_zarr.py \
        --h5 /path/to/trajectory.rgb.pd_ee_delta_pose.physx_cuda.h5 \
        --output maniskill_pusht.zarr

Output zarr layout:
    data/state          (N, 3)  float64  pusher (x, y, theta) per step
    data/slider_state   (N, 3)  float64  T-block (x, y, theta) per step
    data/action         (N, 2)  float32  2D pd_ee delta actions in env-normalized
                                         units (the raw 2D actions executed by the
                                         env at each step -- matches the deltas
                                         recorded by diffusion_policy/gamepad_teleop.py)
    data/target         (N, 3)  float64  goal T-block pose, repeated per step
    data/overhead_camera (N, 128, 128, 3) uint8  (mapped from h5 base_camera)
    data/wrist_camera   (N, 128, 128, 3) uint8  (only if present in h5)
    meta/episode_ends   (E,)    int64    cumulative step counts
"""
import argparse
from pathlib import Path

import h5py
import numpy as np
import zarr
from tqdm import tqdm


def _xytheta(pose7: np.ndarray) -> np.ndarray:
    """(..., 7) [x, y, z, qw, qx, qy, qz] -> (..., 3) [x, y, yaw]."""
    x, y = pose7[..., 0], pose7[..., 1]
    qw, qx, qy, qz = pose7[..., 3], pose7[..., 4], pose7[..., 5], pose7[..., 6]
    yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    return np.stack([x, y, yaw], axis=-1)


def convert(h5_path: str, zarr_path: str, success_only: bool = False):
    states, sliders, actions, targets = [], [], [], []
    overheads, wrists = [], []
    episode_ends, n_skipped = [], 0
    end = 0
    has_wrist = None

    with h5py.File(h5_path, "r") as f:
        traj_keys = sorted(
            (k for k in f.keys() if k.startswith("traj_")),
            key=lambda s: int(s.split("_")[1]),
        )
        for tk in tqdm(traj_keys, desc="episodes"):
            t = f[tk]
            if success_only and not bool(t["success"][:].any()):
                n_skipped += 1
                continue

            T = t["actions"].shape[0]  # T actions, T+1 observations
            tcp = t["obs/extra/tcp_pose"][:T]              # (T, 7)
            tee = t["env_states/actors/Tee"][:T, :7]       # (T, 7)
            goal = t["env_states/actors/goal_Tee"][0, :7]  # (7,) constant

            state = _xytheta(tcp).astype(np.float64)
            slider = _xytheta(tee).astype(np.float64)
            target = np.tile(_xytheta(goal), (T, 1)).astype(np.float64)
            # PlanarPushTEnv overrides single_action_space to shape (2,), so
            # RecordEpisode stores the raw 2D deltas the env received. Use
            # them directly so the zarr action semantics match
            # diffusion_policy/gamepad_teleop.py (delta, not absolute target).
            actions_raw = np.asarray(t["actions"][:T])
            if actions_raw.ndim != 2 or actions_raw.shape[-1] != 2:
                raise ValueError(
                    f"Expected 2D actions of shape (T, 2) from PlanarPushTEnv, "
                    f"got shape {actions_raw.shape}. If this h5 was collected "
                    f"with a non-PlanarPushT env, this conversion script does "
                    f"not apply."
                )
            action = actions_raw.astype(np.float32)

            overheads.append(t["obs/sensor_data/base_camera/rgb"][:T])

            wrist_present = "wrist_camera" in t["obs/sensor_data"]
            if has_wrist is None:
                has_wrist = wrist_present
            elif has_wrist != wrist_present:
                raise ValueError(f"{tk}: wrist_camera presence inconsistent across episodes")
            if wrist_present:
                wrists.append(t["obs/sensor_data/wrist_camera/rgb"][:T])

            states.append(state); sliders.append(slider)
            actions.append(action); targets.append(target)
            end += T
            episode_ends.append(end)

    if not episode_ends:
        raise RuntimeError("No episodes converted (all skipped?).")

    states = np.concatenate(states); sliders = np.concatenate(sliders)
    actions = np.concatenate(actions); targets = np.concatenate(targets)
    overheads = np.concatenate(overheads)
    episode_ends = np.asarray(episode_ends, dtype=np.int64)

    root = zarr.open_group(zarr_path, mode="w")
    d = root.create_group("data"); m = root.create_group("meta")
    d.create_dataset("state", data=states, chunks=(1024, 3))
    d.create_dataset("slider_state", data=sliders, chunks=(1024, 3))
    d.create_dataset("action", data=actions, chunks=(2048, 2))
    d.create_dataset("target", data=targets, chunks=(1024, 3))
    d.create_dataset("overhead_camera", data=overheads, chunks=(128, 128, 128, 3))
    if has_wrist:
        wrists = np.concatenate(wrists)
        d.create_dataset("wrist_camera", data=wrists, chunks=(128, 128, 128, 3))
    m.create_dataset("episode_ends", data=episode_ends)

    msg = f"Wrote {len(episode_ends)} episodes, {end} steps → {zarr_path}"
    if success_only:
        msg += f" (skipped {n_skipped} failed)"
    if not has_wrist:
        msg += " [no wrist_camera in source h5]"
    print(msg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True, help="path to trajectory.rgb.*.h5")
    ap.add_argument("--output", default="maniskill_pusht.zarr")
    ap.add_argument("--success-only", action="store_true",
                    help="skip episodes where success was never reached")
    args = ap.parse_args()
    convert(args.h5, args.output, success_only=args.success_only)


if __name__ == "__main__":
    main()
