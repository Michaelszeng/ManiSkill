"""Interactively view episodes from a processed zarr dataset."""

import argparse
import importlib
import io
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")  # off-screen rendering; must be set before pyplot import
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Per-episode state/action data loader
# ---------------------------------------------------------------------------


def load_episode_state_action(dataset, episode_ends, ep):
    start = 0 if ep == 0 else episode_ends[ep - 1]
    end = episode_ends[ep]

    state = dataset["data"]["state"][start:end]  # (T, 3) x, y, theta
    slider_state = dataset["data"]["slider_state"][start:end]  # (T, 3) x, y, theta
    action = dataset["data"]["action"][start:end]  # (T, 2) dx, dy
    target = dataset["data"]["target"][start:end]  # (T, 3) x, y, theta

    return state, slider_state, action, target


# ---------------------------------------------------------------------------
# Matplotlib panel renderer
# ---------------------------------------------------------------------------


def render_state_panel(
    state: np.ndarray,
    slider_state: np.ndarray,
    action: np.ndarray,
    target: np.ndarray,
    frame_idx: int,
    panel_h: int = 480,
    panel_w: int = 480,
) -> np.ndarray:
    """Render a matplotlib panel for the given episode state/action data.

    Returns an (panel_h, panel_w, 3) uint8 RGB image.
    """
    T = len(state)
    fig = plt.figure(figsize=(panel_w / 100, panel_h / 100), dpi=100)

    # ---- 2D trajectory plot (top, larger) ----
    ax2d = fig.add_axes([0.05, 0.38, 0.90, 0.58])

    # Full trajectory as a faint gray line
    ax2d.plot(state[:, 0], state[:, 1], color="lightgray", linewidth=0.8, zorder=1, label="Pusher Path")
    ax2d.plot(slider_state[:, 0], slider_state[:, 1], color="darkgray", linewidth=0.8, zorder=1, label="Slider Path")

    # Colour the trajectory from blue (start) to red (end) to show time
    for t in range(0, T - 1, max(1, T // 60)):
        frac = t / max(T - 1, 1)
        color = (frac, 0.0, 1.0 - frac)
        ax2d.plot(
            state[t : t + 2, 0], state[t : t + 2, 1], color=color, linewidth=1.5, zorder=2
        )
        ax2d.plot(
            slider_state[t : t + 2, 0], slider_state[t : t + 2, 1], color=color, linewidth=1.5, zorder=2, linestyle="--"
        )

    # Current position
    cx, cy, ctheta = state[frame_idx]
    sx, sy, stheta = slider_state[frame_idx]
    tx, ty, ttheta = target[frame_idx]
    
    ax2d.scatter([cx], [cy], color="yellow", s=60, zorder=5, edgecolors="black", linewidths=0.5, label="Pusher")
    ax2d.scatter([sx], [sy], color="cyan", s=60, zorder=5, edgecolors="black", linewidths=0.5, label="Slider")
    ax2d.scatter([tx], [ty], color="green", s=60, zorder=5, edgecolors="black", linewidths=0.5, marker="x", label="Target")

    # Current orientation (as a line segment)
    scale = 0.05
    ax2d.plot([cx, cx + scale * np.cos(ctheta)], [cy, cy + scale * np.sin(ctheta)], color="yellow", linewidth=2, zorder=6)
    ax2d.plot([sx, sx + scale * np.cos(stheta)], [sy, sy + scale * np.sin(stheta)], color="cyan", linewidth=2, zorder=6)
    ax2d.plot([tx, tx + scale * np.cos(ttheta)], [ty, ty + scale * np.sin(ttheta)], color="green", linewidth=2, zorder=6)

    # Current action delta arrow
    if action is not None:
        dp = action[frame_idx]
        mag = np.linalg.norm(dp)
        if mag > 1e-6:
            dp_scaled = dp / mag * min(mag * 5, scale * 1.5)
            ax2d.quiver(
                cx,
                cy,
                dp_scaled[0],
                dp_scaled[1],
                color="orange",
                angles="xy",
                scale_units="xy",
                scale=1,
                width=0.005,
                zorder=7,
                label="Action"
            )

    ax2d.set_xlabel("X", fontsize=7, labelpad=0)
    ax2d.set_ylabel("Y", fontsize=7, labelpad=0)
    ax2d.tick_params(labelsize=6, pad=0)
    ax2d.set_title(f"Trajectory  (frame {frame_idx}/{T - 1})", fontsize=8, pad=2)
    ax2d.legend(fontsize=6, loc="upper right")
    ax2d.axis("equal")

    # ---- Time-series plot (bottom) ----
    ax_ts = fig.add_axes([0.10, 0.04, 0.85, 0.28])
    t_axis = np.arange(T)

    ax_ts.plot(t_axis, state[:, 0], color="red", linewidth=0.8, label="Pusher X")
    ax_ts.plot(t_axis, state[:, 1], color="green", linewidth=0.8, label="Pusher Y")
    ax_ts.plot(t_axis, state[:, 2], color="blue", linewidth=0.8, label="Pusher Theta")

    ax_ts.axvline(frame_idx, color="yellow", linewidth=1.2, zorder=5)
    ax_ts.set_xlim(0, T - 1)
    ax_ts.tick_params(labelsize=6)
    ax_ts.set_ylabel("State", fontsize=7)
    ax_ts.legend(fontsize=6, loc="upper right", ncol=3)
    ax_ts.set_facecolor("#1a1a1a")
    ax_ts.grid(color="gray", linewidth=0.3)

    fig.patch.set_facecolor("#1a1a1a")
    ax2d.set_facecolor("#1a1a1a")
    ax2d.grid(True, linewidth=0.3)

    # Render to numpy array via PNG encode/decode
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor="#1a1a1a", dpi=100)
    plt.close(fig)
    buf.seek(0)
    import cv2 as _cv2

    arr = np.frombuffer(buf.getvalue(), np.uint8)
    panel_bgr = _cv2.imdecode(arr, _cv2.IMREAD_COLOR)
    panel_rgb = _cv2.cvtColor(panel_bgr, _cv2.COLOR_BGR2RGB)

    # Resize to exact panel dimensions
    if panel_rgb.shape[:2] != (panel_h, panel_w):
        panel_rgb = _cv2.resize(panel_rgb, (panel_w, panel_h))

    return panel_rgb


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Interactively view episodes from a processed zarr dataset.")
    parser.add_argument("zarr_path", help="Path to the .zarr dataset file")
    parser.add_argument(
        "--episode",
        "-e",
        type=int,
        default=0,
        help="Episode index to start at (default: 0)",
    )
    parser.add_argument("--fps", type=int, default=10, help="Playback speed in frames per second (default: 10)")
    parser.add_argument(
        "--state",
        action="store_true",
        help="Show the 2D state/action matplotlib panel below the camera feeds.",
    )
    args = parser.parse_args()

    # src/dataset/zarr.py shadows the installed zarr package when this file is
    # run as a script (Python inserts the script's directory first on sys.path).
    # We must evict the cached local module and re-import with sys.path pruned.
    _script_dir = str(Path(__file__).parent.resolve())
    _saved_path = sys.path[:]
    sys.path = [p for p in sys.path if Path(p).resolve() != Path(_script_dir).resolve()]
    if "zarr" in sys.modules:
        del sys.modules["zarr"]
    zarr_lib = importlib.import_module("zarr")
    sys.path = _saved_path

    dataset = zarr_lib.open(args.zarr_path, mode="r")
    episode_ends = dataset["meta"]["episode_ends"][:]
    n_episodes = len(episode_ends)

    # Detect whether state/action data is available
    has_state = args.state and "data" in dataset and "state" in dataset["data"]

    print(f"Dataset: {args.zarr_path}")
    print(f"Episodes: {n_episodes}  |  Total frames: {episode_ends[-1]}")
    print()
    print(f"Structure:\n{zarr_lib.open_group(args.zarr_path, mode='r').tree()}")
    print()
    print("Controls:")
    print("  k / l     step 1 / 10 frames forward")
    print("  j / h     step 1 / 10 frames backward")
    print("  n / p     next / previous episode")
    print("  Space     toggle play/pause")
    print("  q         quit")
    print()

    ep_idx = max(0, min(args.episode, n_episodes - 1))
    frame_delay_ms = max(1, 1000 // args.fps)
    playing = False

    # Camera images are 128x128 each; concatenated strip is 128x256.
    # Scale 4x -> 512x1024. State panel (if shown) is stacked below at full width.
    cam_h = 512
    cam_w = 1024  # 256 * 4 (4x uniform scale)
    state_h = 720  # height of the state/action panel when shown
    win_w = cam_w
    win_h = cam_h + state_h if has_state else cam_h
    cv2.namedWindow("Dataset Viewer", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Dataset Viewer", win_w, win_h)

    def load_episode(ep):
        start = 0 if ep == 0 else episode_ends[ep - 1]
        end = episode_ends[ep]
        imgs1 = dataset["data"]["base_camera"][start:end]
        imgs2 = dataset["data"]["wrist_camera"][start:end]
        
        # Check if success is in meta
        success = False
        if "meta" in dataset and "success" in dataset["meta"]:
            success = bool(dataset["meta"]["success"][ep])
            
        camera_frames = np.concatenate([imgs1, imgs2], axis=2)  # (T, 128, 256, 3)

        state_data = None
        if has_state:
            state_data = load_episode_state_action(dataset, episode_ends, ep)

        return camera_frames, success, state_data

    camera_frames, success, state_data = load_episode(ep_idx)
    frame_idx = 0

    # Cache rendered matplotlib panels per episode to avoid re-rendering each frame.
    # Panels are rendered lazily and cached in a list indexed by frame.
    panel_cache: list = []

    def get_panel(fi):
        nonlocal panel_cache
        if not panel_cache:
            panel_cache = [None] * len(camera_frames)
        if panel_cache[fi] is None:
            state, slider_state, action, target = state_data
            panel_cache[fi] = render_state_panel(
                state,
                slider_state,
                action,
                target,
                fi,
                panel_h=state_h,
                panel_w=cam_w,
            )
        return panel_cache[fi]

    def on_episode_change(ep):
        nonlocal camera_frames, success, state_data, panel_cache
        camera_frames, success, state_data = load_episode(ep)
        panel_cache = []

    while True:
        cam = camera_frames[frame_idx]
        # Camera images are stored RGB; convert to BGR for OpenCV display
        cam_bgr = cv2.cvtColor(cam, cv2.COLOR_RGB2BGR)
        # Scale uniformly: 128x256 -> 512x1024 (4x in both dimensions)
        cam_bgr = cv2.resize(cam_bgr, (cam_w, cam_h))

        if has_state:
            panel_rgb = get_panel(frame_idx)
            panel_bgr = cv2.cvtColor(panel_rgb, cv2.COLOR_RGB2BGR)
            display = np.concatenate([cam_bgr, panel_bgr], axis=0)
        else:
            display = cam_bgr

        label = (
            f"Ep {ep_idx + 1}/{n_episodes}  |  "
            f"Frame {frame_idx}/{len(camera_frames) - 1}"
        )
        if "meta" in dataset and "success" in dataset["meta"]:
            label += f"  |  {'SUCCESS' if success else 'FAILURE'}"
            
        cv2.putText(display, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Dataset Viewer", display)

        wait_ms = frame_delay_ms if playing else 0
        key = cv2.waitKey(wait_ms) & 0xFF

        if key == ord("q"):
            break
        elif key == ord(" "):
            playing = not playing
        elif key == ord("k") or (playing and key == 255):
            if frame_idx < len(camera_frames) - 1:
                frame_idx += 1
            else:
                playing = False
        elif key == ord("l"):
            frame_idx = min(frame_idx + 10, len(camera_frames) - 1)
        elif key == ord("j"):
            frame_idx = max(frame_idx - 1, 0)
        elif key == ord("h"):
            frame_idx = max(frame_idx - 10, 0)
        elif key == ord("n"):
            ep_idx = min(ep_idx + 1, n_episodes - 1)
            on_episode_change(ep_idx)
            frame_idx = 0
            playing = False
        elif key == ord("p"):
            ep_idx = max(ep_idx - 1, 0)
            on_episode_change(ep_idx)
            frame_idx = 0
            playing = False

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
