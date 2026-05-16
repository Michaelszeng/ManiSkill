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

Re-running with the same --output path appends new trajectories to the existing
zarr rather than overwriting it, so data can be collected across multiple sessions.
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


def _read_gamepad(joy, deadzone=0.05):
    pygame.event.pump()
    lx, ly = joy.get_axis(0), joy.get_axis(1)
    lx = 0.0 if abs(lx) < deadzone else lx
    ly = 0.0 if abs(ly) < deadzone else ly
    buttons = {
        "X": bool(joy.get_button(0)),  # b0
        "A": bool(joy.get_button(1)),  # b1
        "B": bool(joy.get_button(2)),  # b2
        "LT": bool(joy.get_button(6)),  # L2 on Logitech Dual Action
        "RT": bool(joy.get_button(7)),  # R2 on Logitech Dual Action
    }
    # World +x is "forward" in the env; map left-stick (lx, ly) → (ly, lx)
    # (inverted: pushing the stick up moves the pusher in -x).
    return np.array([ly, lx], dtype=np.float32), buttons


def _interactive_debug(joy):
    """Live display of all axes and buttons. Press Ctrl-C or Esc to exit."""
    n_axes = joy.get_numaxes()
    n_btns = joy.get_numbuttons()
    _print_blue(f"Gamepad: {joy.get_name()}  |  {n_axes} axes, {n_btns} buttons")
    _print_blue("Live gamepad debug — press Ctrl-C or Esc to exit.\n")
    try:
        while True:
            pygame.event.pump()
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    raise KeyboardInterrupt

            axes = "  ".join(f"a{i}:{joy.get_axis(i):+.2f}" for i in range(n_axes))
            btns = "  ".join(f"b{i}:{'ON ' if joy.get_button(i) else 'off'}" for i in range(n_btns))
            line = f"\r{axes}    {btns}   "
            print(line, end="", flush=True)
            time.sleep(0.05)
    except KeyboardInterrupt:
        print()
        _print_blue("Debug mode exited.")


def _pressed(prev, curr):
    return {k: (curr[k] and not prev[k]) for k in curr}


def _save_trajectory(traj_dir, idx, buf):
    np.savez_compressed(
        traj_dir / f"{idx:05d}.npz",
        state=np.stack(buf["state"]),
        slider_state=np.stack(buf["slider_state"]),
        action=np.stack(buf["action"]),
        base_camera=np.stack(buf["base_camera"]),
        wrist_camera=np.stack(buf["wrist_camera"]),
    )


def convert_to_zarr(traj_dir: Path, zarr_path: str):
    print(f"\nConverting {traj_dir} → {zarr_path}")
    states, sliders, actions, targets, bases, wrists, episode_ends = ([], [], [], [], [], [], [])
    session_len = 0
    for f in sorted(traj_dir.glob("*.npz")):
        d = np.load(f)
        state, slider = d["state"], d["slider_state"]
        action = d["action"]  # actual commanded 2D actions
        target = np.tile(slider[-1], (len(state), 1))
        states.append(state)
        sliders.append(slider)
        actions.append(action)
        targets.append(target)
        bases.append(d["base_camera"])
        wrists.append(d["wrist_camera"])
        session_len += len(state)
        episode_ends.append(session_len)

    states = np.concatenate(states)
    sliders = np.concatenate(sliders)
    actions = np.concatenate(actions)
    targets = np.concatenate(targets)
    bases = np.concatenate(bases)
    wrists = np.concatenate(wrists)
    episode_ends = np.array(episode_ends)

    root = zarr.open_group(zarr_path, mode="a")

    if "data" in root:
        # Append to existing zarr; offset episode_ends by the current total length.
        existing_len = root["data"]["state"].shape[0]
        root["data"]["state"].append(states)
        root["data"]["slider_state"].append(sliders)
        root["data"]["action"].append(actions)
        root["data"]["target"].append(targets)
        root["data"]["base_camera"].append(bases)
        root["data"]["wrist_camera"].append(wrists)
        root["meta"]["episode_ends"].append(episode_ends + existing_len)
        total_trajs = root["meta"]["episode_ends"].shape[0]
        total_steps = root["data"]["state"].shape[0]
        print(f"Appended {len(episode_ends)} trajectories ({total_trajs} total, {total_steps} timesteps).")
    else:
        data = root.create_group("data")
        meta = root.create_group("meta")
        data.create_dataset("state", data=states, chunks=(1024, 3))
        data.create_dataset("slider_state", data=sliders, chunks=(1024, 3))
        data.create_dataset("action", data=actions, chunks=(2048, 2))
        data.create_dataset("target", data=targets, chunks=(1024, 3))
        data.create_dataset("base_camera", data=bases, chunks=(128, 128, 128, 3))
        data.create_dataset("wrist_camera", data=wrists, chunks=(128, 128, 128, 3))
        meta.create_dataset("episode_ends", data=episode_ends)
        print(f"Saved {len(episode_ends)} trajectories, {session_len} timesteps total.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="planar_pusht_teleop.zarr")
    ap.add_argument("--traj-dir", default="planar_pusht_trajs")
    ap.add_argument("--scale", type=float, default=0.15, help="base action scale applied to joystick output")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--debug-axes", action="store_true", help="print gamepad axes/button info at startup and exit")
    args = ap.parse_args()

    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() == 0:
        raise RuntimeError("No gamepad detected. Plug in a controller.")
    joy = pygame.joystick.Joystick(0)
    joy.init()
    _print_blue(f"Gamepad: {joy.get_name()}")

    if args.debug_axes:
        time.sleep(0.1)  # let axes settle
        _interactive_debug(joy)
        pygame.quit()
        return

    zarr_path = Path(args.output)
    if zarr_path.exists():
        _existing = zarr.open_group(str(zarr_path), mode="r")
        n_existing = int(_existing["meta"]["episode_ends"].shape[0]) if "meta" in _existing else 0
    else:
        n_existing = 0
    _print_blue(f"Zarr: {zarr_path} ({n_existing} existing episodes)")

    traj_dir = Path(args.traj_dir)
    if traj_dir.exists():
        shutil.rmtree(traj_dir)
    traj_dir.mkdir(parents=True)

    env = gym.make(
        "Planar-PushT-Teleop-v1",
        num_envs=1,
        obs_mode="rgb",
        render_mode="human",
        require_pusher_at_start=True,
    )
    seed = args.seed if args.seed is not None else int(time.time()) % 10**6
    obs, _ = env.reset(seed=seed)
    base_env = env.unwrapped

    fsm = FSMState.REGULAR
    prev = {"A": False, "B": False, "X": False, "LT": False, "RT": False}
    buf = {k: [] for k in ("state", "slider_state", "action", "base_camera", "wrist_camera")}
    n_saved = 0

    def record(action_2d):
        buf["state"].append(_pusher_xytheta(obs))
        buf["slider_state"].append(_slider_xytheta(base_env))
        buf["action"].append(action_2d.reshape(2))
        buf["base_camera"].append(_img(obs["sensor_data"]["base_camera"]["rgb"]))
        buf["wrist_camera"].append(_img(obs["sensor_data"]["wrist_camera"]["rgb"]))

    def clear():
        for v in buf.values():
            v.clear()

    _print_blue("Ready. Press A to start recording, B to reset, X to quit.")
    while fsm != FSMState.TERMINATE:
        joy_xy, btn = _read_gamepad(joy)
        pressed = _pressed(prev, btn)
        prev = btn

        scale = args.scale
        if btn["RT"]:
            scale *= 0.5
        if btn["LT"]:
            scale *= 3.0

        if pressed["X"]:
            fsm = FSMState.TERMINATE
            break

        if fsm == FSMState.REGULAR:
            if pressed["A"]:
                clear()
                fsm = FSMState.DATA_COLLECTION
                _print_blue("Recording started.")
            elif pressed["B"]:
                obs, _ = env.reset()
                _print_blue("Environment reset.")
        elif fsm == FSMState.DATA_COLLECTION:
            if pressed["A"]:
                ep_len = len(buf["state"])
                _save_trajectory(traj_dir, n_saved, buf)
                n_saved += 1
                clear()
                obs, _ = env.reset()
                fsm = FSMState.REGULAR
                _print_blue(f"Trajectory saved ({ep_len} steps) — {n_saved} this session, {n_existing + n_saved} total.")
                continue
            elif pressed["B"]:
                clear()
                obs, _ = env.reset()
                fsm = FSMState.REGULAR
                _print_blue("Recording discarded.")
                continue

        action = (joy_xy * scale).reshape(1, 2)
        if fsm == FSMState.DATA_COLLECTION:
            record(action)  # record (obs[t], action[t]) before stepping
        obs, _, terminated, _, _ = env.step(action)
        if fsm == FSMState.DATA_COLLECTION:
            if terminated.any():
                ep_len = len(buf["state"])
                _save_trajectory(traj_dir, n_saved, buf)
                n_saved += 1
                clear()
                obs, _ = env.reset()
                fsm = FSMState.REGULAR
                _print_blue(f"Success! Trajectory saved ({ep_len} steps) — {n_saved} this session, {n_existing + n_saved} total.")
        elif terminated.any():
            obs, _ = env.reset()
            _print_blue("Success! Environment reset.")
        env.render()

    env.close()
    pygame.quit()
    _print_blue(f"Done. Collected {n_saved} trajectories.")
    if n_saved > 0:
        convert_to_zarr(traj_dir, args.output)


if __name__ == "__main__":
    main()
