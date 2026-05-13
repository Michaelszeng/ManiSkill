"""
Gamepad teleop data collection for Planar-PushT-v1 (ManiSkill).

Controls:
    A:  start recording / save trajectory
    B:  discard recording / reset environment
    X:  quit and convert collected trajectories to zarr
    RT: half-speed (slow control)
    LT: triple-speed (fast control)

Usage:
    python diffusion_policy/gamepad_teleop.py --output planar_pusht.zarr
"""
import argparse
import os
import shutil
import time
from enum import Enum
from pathlib import Path

import gymnasium as gym
import numpy as np
import pygame
import torch
import zarr
from transforms3d.euler import quat2euler

import mani_skill.envs  # noqa: F401  (registers envs)
from mani_skill.envs.tasks.tabletop.planar_push_t import PlanarPushTEnv
from mani_skill.utils.registration import register_env


# Same task as Planar-PushT-v1 (inherits its base + wrist cameras), but with a
# long episode horizon so teleop sessions aren't truncated mid-trajectory.
@register_env("Planar-PushT-Teleop-v1", max_episode_steps=10_000)
class PlanarPushTTeleopEnv(PlanarPushTEnv):
    pass


class FSMState(Enum):
    REGULAR = "regular"
    DATA_COLLECTION = "data collection"
    TERMINATE = "terminate"


def _print_blue(text):
    print(f"\033[94m{text}\033[0m")


def _to_numpy(x):
    return x.cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def _quat_to_z(q):
    q = _to_numpy(q).reshape(-1)
    return quat2euler([q[0], q[1], q[2], q[3]], axes="sxyz")[2]


def _img(rgb):
    arr = _to_numpy(rgb)
    if arr.ndim == 4:
        arr = arr[0]
    return arr.astype(np.uint8)


def _pusher_xytheta(obs):
    p = _to_numpy(obs["extra"]["tcp_pose"]).reshape(-1)
    return np.array([p[0], p[1], _quat_to_z(p[3:7])])


def _slider_xytheta(base_env):
    pos = _to_numpy(base_env.tee.pose.p).reshape(-1)
    quat = _to_numpy(base_env.tee.pose.q).reshape(-1)
    return np.array([pos[0], pos[1], _quat_to_z(quat)])


def _read_gamepad(joy, deadzone=0.15):
    pygame.event.pump()
    lx, ly = joy.get_axis(0), joy.get_axis(1)
    lx = 0.0 if abs(lx) < deadzone else lx
    ly = 0.0 if abs(ly) < deadzone else ly
    n_axes = joy.get_numaxes()
    buttons = {
        "A": bool(joy.get_button(0)),
        "B": bool(joy.get_button(1)),
        "X": bool(joy.get_button(2)),
        # Trigger axes go from -1 (released) to +1 (pressed) on most pads.
        "LT": (joy.get_axis(4) > 0.0) if n_axes > 4 else False,
        "RT": (joy.get_axis(5) > 0.0) if n_axes > 5 else False,
    }
    # World +x is "forward" in the env; map left-stick (lx, ly) → (-ly, -lx)
    # so pushing the stick up moves the pusher in +x.
    return np.array([-ly, -lx], dtype=np.float32), buttons


def _pressed(prev, curr):
    return {k: (curr[k] and not prev[k]) for k in curr}


def _save_trajectory(traj_dir, idx, buf):
    np.savez_compressed(
        traj_dir / f"{idx:05d}.npz",
        state=np.stack(buf["state"]),
        slider_state=np.stack(buf["slider_state"]),
        base_camera=np.stack(buf["base_camera"]),
        wrist_camera=np.stack(buf["wrist_camera"]),
    )


def convert_to_zarr(traj_dir: Path, zarr_path: str):
    print(f"\nConverting {traj_dir} → {zarr_path}")
    states, sliders, actions, targets, bases, wrists, episode_ends = (
        [], [], [], [], [], [], []
    )
    end = 0
    for f in sorted(traj_dir.glob("*.npz")):
        d = np.load(f)
        state, slider = d["state"], d["slider_state"]
        action = np.concatenate([state[1:, :2], state[-1:, :2]], axis=0)
        target = np.tile(slider[-1], (len(state), 1))
        states.append(state); sliders.append(slider)
        actions.append(action); targets.append(target)
        bases.append(d["base_camera"]); wrists.append(d["wrist_camera"])
        end += len(state)
        episode_ends.append(end)

    states = np.concatenate(states); sliders = np.concatenate(sliders)
    actions = np.concatenate(actions); targets = np.concatenate(targets)
    bases = np.concatenate(bases); wrists = np.concatenate(wrists)

    root = zarr.open_group(zarr_path, mode="w")
    data = root.create_group("data"); meta = root.create_group("meta")
    data.create_dataset("state", data=states, chunks=(1024, 3))
    data.create_dataset("slider_state", data=sliders, chunks=(1024, 3))
    data.create_dataset("action", data=actions, chunks=(2048, 2))
    data.create_dataset("target", data=targets, chunks=(1024, 3))
    data.create_dataset("base_camera", data=bases, chunks=(128, 128, 128, 3))
    data.create_dataset("wrist_camera", data=wrists, chunks=(128, 128, 128, 3))
    meta.create_dataset("episode_ends", data=np.array(episode_ends))
    print(f"Saved {len(episode_ends)} trajectories, {end} timesteps total.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="planar_pusht_teleop.zarr")
    ap.add_argument("--traj-dir", default="planar_pusht_trajs")
    ap.add_argument("--scale", type=float, default=0.4,
                    help="base action scale applied to joystick output")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() == 0:
        raise RuntimeError("No gamepad detected. Plug in a controller.")
    joy = pygame.joystick.Joystick(0); joy.init()
    _print_blue(f"Gamepad: {joy.get_name()}")

    traj_dir = Path(args.traj_dir)
    if traj_dir.exists():
        shutil.rmtree(traj_dir)
    traj_dir.mkdir(parents=True)

    env = gym.make("Planar-PushT-Teleop-v1", num_envs=1,
                   obs_mode="rgb", render_mode="human")
    seed = args.seed if args.seed is not None else int(time.time()) % 10**6
    obs, _ = env.reset(seed=seed)
    base_env = env.unwrapped

    fsm = FSMState.REGULAR
    prev = {"A": False, "B": False, "X": False, "LT": False, "RT": False}
    buf = {k: [] for k in ("state", "slider_state", "base_camera", "wrist_camera")}
    n_saved = 0

    def record():
        buf["state"].append(_pusher_xytheta(obs))
        buf["slider_state"].append(_slider_xytheta(base_env))
        buf["base_camera"].append(_img(obs["sensor_data"]["base_camera"]["rgb"]))
        buf["wrist_camera"].append(_img(obs["sensor_data"]["wrist_camera"]["rgb"]))

    def clear():
        for v in buf.values():
            v.clear()

    _print_blue("Ready. Press A to start recording, B to reset, X to quit.")
    while fsm != FSMState.TERMINATE:
        joy_xy, btn = _read_gamepad(joy)
        pressed = _pressed(prev, btn); prev = btn

        scale = args.scale
        if btn["RT"]: scale *= 0.5
        if btn["LT"]: scale *= 3.0

        if pressed["X"]:
            fsm = FSMState.TERMINATE
            break

        if fsm == FSMState.REGULAR:
            if pressed["A"]:
                clear(); record()
                fsm = FSMState.DATA_COLLECTION
                _print_blue("Recording started.")
            elif pressed["B"]:
                obs, _ = env.reset()
                _print_blue("Environment reset.")
        elif fsm == FSMState.DATA_COLLECTION:
            if pressed["A"]:
                _save_trajectory(traj_dir, n_saved, buf)
                n_saved += 1
                clear(); obs, _ = env.reset()
                fsm = FSMState.REGULAR
                _print_blue(f"Trajectory saved ({n_saved} total).")
                continue
            elif pressed["B"]:
                clear(); obs, _ = env.reset()
                fsm = FSMState.REGULAR
                _print_blue("Recording discarded.")
                continue

        action = (joy_xy * scale).reshape(1, 2)
        obs, _, _, _, _ = env.step(action)
        if fsm == FSMState.DATA_COLLECTION:
            record()
        env.render()

    env.close()
    pygame.quit()
    _print_blue(f"Done. Collected {n_saved} trajectories.")
    if n_saved > 0:
        convert_to_zarr(traj_dir, args.output)


if __name__ == "__main__":
    main()
