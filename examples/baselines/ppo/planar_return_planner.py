"""Markovian return-to-start planner for Planar-PushT-v1.

Re-plans every step from current state:
  1. Rasterize the T's current footprint onto a 2D occupancy grid,
     dilated by the pusher tip radius + safety margin.
  2. Run 8-connected Dijkstra from the start pose over free cells,
     producing a distance-to-start field.
  3. Step in the direction of steepest descent on that field.

The action is a pure function of the environment state, so episodes
remain Markovian.
"""
import heapq

import numpy as np

# Workspace bounds for PlanarPushT (xy plane), padded around spawn region.
_X_MIN, _X_MAX = -0.55, 0.15
_Y_MIN, _Y_MAX = -0.40, 0.50
_CELL = 0.005  # 5 mm

# T box geometry from PlanarPushTEnv._load_scene, in the T's actor frame.
_COM_Y = 0.0375
_BOX1_HW, _BOX1_HH = 0.10, 0.025
_BOX2_HW, _BOX2_HH = 0.025, 0.075
_BOX1_CY = -_COM_Y
_BOX2_CY = 4 * _BOX1_HH - _COM_Y  # 0.0625

_PUSHER_R = 0.012  # panda_stick tip; small but nonzero to inflate the T.
_MAX_STEP_M = 0.05  # controller scales action [-1, 1] → up to 5 cm per step.

_SQRT2 = float(np.sqrt(2.0))
_NEIGHBORS = (
    (-1, -1, _SQRT2), (-1, 0, 1.0), (-1, 1, _SQRT2),
    (0, -1, 1.0),                   (0, 1, 1.0),
    (1, -1, _SQRT2),  (1, 0, 1.0),  (1, 1, _SQRT2),
)

_NX = int(round((_X_MAX - _X_MIN) / _CELL))
_NY = int(round((_Y_MAX - _Y_MIN) / _CELL))
_CELL_X = _X_MIN + (np.arange(_NX) + 0.5) * _CELL
_CELL_Y = _Y_MIN + (np.arange(_NY) + 0.5) * _CELL
_CELL_XX, _CELL_YY = np.meshgrid(_CELL_X, _CELL_Y)


def _quat_to_z(q):
    # q = [qw, qx, qy, qz]; assumes rotation is about z only.
    return 2.0 * float(np.arctan2(q[3], q[0]))


def _xy_to_cell(x, y):
    j = int(np.clip(round((x - _X_MIN) / _CELL - 0.5), 0, _NX - 1))
    i = int(np.clip(round((y - _Y_MIN) / _CELL - 0.5), 0, _NY - 1))
    return i, j


def _build_occupancy(tee_xy, tee_theta, margin):
    cos_t, sin_t = np.cos(tee_theta), np.sin(tee_theta)
    dx = _CELL_XX - tee_xy[0]
    dy = _CELL_YY - tee_xy[1]
    # World → T local frame (rotate by -theta).
    lx = cos_t * dx + sin_t * dy
    ly = -sin_t * dx + cos_t * dy
    box1 = (np.abs(lx) <= _BOX1_HW + margin) & (np.abs(ly - _BOX1_CY) <= _BOX1_HH + margin)
    box2 = (np.abs(lx) <= _BOX2_HW + margin) & (np.abs(ly - _BOX2_CY) <= _BOX2_HH + margin)
    return box1 | box2


def _dijkstra(occ, src):
    # Pure-Python loop with numpy scalar access is ~100x slower per op than
    # native list access, so we flatten to Python lists for the hot path.
    ny, nx = occ.shape
    occ_flat = occ.ravel().tolist()
    inf = float("inf")
    D = [inf] * (ny * nx)
    si, sj = src
    src_idx = si * nx + sj
    if occ_flat[src_idx]:
        return np.array(D, dtype=np.float32).reshape(ny, nx)
    D[src_idx] = 0.0
    heap = [(0.0, si, sj)]
    while heap:
        d, i, j = heapq.heappop(heap)
        idx = i * nx + j
        if d > D[idx]:
            continue
        for di, dj, w in _NEIGHBORS:
            ni, nj = i + di, j + dj
            if 0 <= ni < ny and 0 <= nj < nx:
                nidx = ni * nx + nj
                if not occ_flat[nidx]:
                    nd = d + w
                    if nd < D[nidx]:
                        D[nidx] = nd
                        heapq.heappush(heap, (nd, ni, nj))
    return np.array(D, dtype=np.float32).reshape(ny, nx)


def _nearest_free(occ, i, j, max_r=8):
    if not occ[i, j]:
        return i, j
    for r in range(1, max_r + 1):
        for di in range(-r, r + 1):
            for dj in range(-r, r + 1):
                if max(abs(di), abs(dj)) != r:
                    continue
                ni, nj = i + di, j + dj
                if 0 <= ni < _NY and 0 <= nj < _NX and not occ[ni, nj]:
                    return ni, nj
    return i, j


def _los_clear(occ_flat, i0, j0, i1, j1):
    di, dj = i1 - i0, j1 - j0
    n = max(abs(di), abs(dj))
    if n == 0:
        return True
    for k in range(1, n + 1):
        ci = int(round(i0 + di * k / n))
        cj = int(round(j0 + dj * k / n))
        if occ_flat[ci * _NX + cj]:
            return False
    return True


def compute_return_action(base_env, env_idx=0, safety_margin=0.01):
    """Return a (2,) float32 action [dx, dy] in the env's [-1, 1] action space.

    `base_env` should be the unwrapped PlanarPushTEnv (exposes .tee, .agent.tcp,
    .ee_starting_pos2D, .pusher_start_tol).
    """
    pusher_xy = base_env.agent.tcp.pose.p[env_idx, :2].detach().cpu().numpy()
    tee_xy = base_env.tee.pose.p[env_idx, :2].detach().cpu().numpy()
    tee_q = base_env.tee.pose.q[env_idx].detach().cpu().numpy()
    start_xy = base_env.ee_starting_pos2D[:2].detach().cpu().numpy()
    tol = float(base_env.pusher_start_tol)

    diff = start_xy - pusher_xy
    dist = float(np.linalg.norm(diff))
    if dist <= tol * 0.25:
        return np.zeros(2, dtype=np.float32)

    occ = _build_occupancy(tee_xy, _quat_to_z(tee_q), _PUSHER_R + safety_margin)
    occ_flat = occ.ravel().tolist()
    si, sj = _xy_to_cell(float(start_xy[0]), float(start_xy[1]))
    if occ_flat[si * _NX + sj]:
        # Goal lies inside the dilated T (shouldn't happen) — fall back to straight line.
        return (diff / max(dist, _MAX_STEP_M)).astype(np.float32)

    D = _dijkstra(occ, (si, sj))

    pi, pj = _xy_to_cell(float(pusher_xy[0]), float(pusher_xy[1]))
    pi, pj = _nearest_free(occ, pi, pj)

    # Walk the gradient cell-by-cell. The 5cm action chord would otherwise cut
    # corners around the T, so we pick the furthest cell that is both reached
    # by descent on D and has clear line-of-sight from the pusher.
    max_walk = int(round(_MAX_STEP_M / _CELL)) * 2 + 4
    ci, cj = pi, pj
    carrot = None
    for _ in range(max_walk):
        best_nd = D[ci, cj]
        nxt = None
        for di, dj, _w in _NEIGHBORS:
            ni, nj = ci + di, cj + dj
            if 0 <= ni < _NY and 0 <= nj < _NX:
                nidx = ni * _NX + nj
                if not occ_flat[nidx] and D[ni, nj] < best_nd:
                    best_nd = D[ni, nj]
                    nxt = (ni, nj)
        if nxt is None:
            break
        ni, nj = nxt
        # Stop extending the carrot once the chord exceeds one controller step.
        chord_m = float(np.hypot(_CELL_X[nj] - pusher_xy[0], _CELL_Y[ni] - pusher_xy[1]))
        if chord_m > _MAX_STEP_M:
            break
        if _los_clear(occ_flat, pi, pj, ni, nj):
            carrot = (ni, nj)
        ci, cj = ni, nj

    if carrot is None:
        # No collision-free descent direction at full step — back off to a
        # half-magnitude straight-line action toward the start.
        action = 0.5 * diff / max(dist, _MAX_STEP_M)
    else:
        cy = _CELL_Y[carrot[0]]
        cx = _CELL_X[carrot[1]]
        direction = np.array([cx - pusher_xy[0], cy - pusher_xy[1]], dtype=np.float32)
        norm = float(np.linalg.norm(direction))
        if norm > 0:
            direction = direction / norm
        # Slow down within one step of the goal so we don't overshoot.
        speed = min(1.0, dist / _MAX_STEP_M)
        action = direction * speed

    return action.astype(np.float32)
