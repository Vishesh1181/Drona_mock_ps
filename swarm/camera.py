"""Overhead camera = the ONLY position sensor the controller gets.

It renders a synthetic greyscale frame by projecting each drone's marker
corners (pinhole model) and warping the ArUco image there
(cv2.warpPerspective), then adds pixel noise, drops 5% of frames and
delivers each frame 80 ms after it was captured.

Camera frame (OpenCV convention), camera at (0,0,3) looking straight down:
    x_c = +x_world,  y_c = -y_world,  z_c = -z_world (depth Z = 3 - z)
    u = fx * x / (3 - z) + cx,   v = -fy * y / (3 - z) + cy
"""
from collections import deque

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


class OverheadCamera:
    def __init__(self, cfg, n, rng):
        c = cfg["camera"]
        self.c = c
        self.n = n
        self.rng = rng
        self.W, self.H = c["width"], c["height"]
        self.K = np.array([[c["fx"], 0, c["cx"]], [0, c["fy"], c["cy"]], [0, 0, 1.0]])
        self.cam_pos = np.array(c["position"], float)
        # rotation camera -> world (columns = camera axes expressed in world)
        self.R_wc = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], float)
        self.period = 1.0 / c["fps"]
        self.next_capture = 0.0
        self.queue = deque()
        self.stats = {"captured": 0, "dropped": 0}

        # marker tile: ArUco (6x6 cells incl. black border) + 1 white cell margin
        d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, c["aruco_dict"]))
        cell = 20
        self.tiles = []
        for i in range(n):
            mk = cv2.aruco.generateImageMarker(d, i, 6 * cell)
            self.tiles.append(cv2.copyMakeBorder(mk, cell, cell, cell, cell,
                                                 cv2.BORDER_CONSTANT, value=255))
        self.tile_px = 8 * cell
        half = c["plate_size"] / 2
        # plate corners in BODY frame, same order as tile corners TL, TR, BR, BL
        # tile x (right) = body +x, tile y (down) = body -y, marker faces up (+z)
        self.plate_body = np.array([[-half, half, c["marker_height"]], [half, half, c["marker_height"]],
                                    [half, -half, c["marker_height"]], [-half, -half, c["marker_height"]]])
        t = self.tile_px
        self.tile_src = np.float32([[-0.5, -0.5], [t - 0.5, -0.5], [t - 0.5, t - 0.5], [-0.5, t - 0.5]])
        self.faults = None

    # ---------------- geometry ----------------
    def project(self, pts_world):
        """World points (k,3) -> pixels (k,2) and depth Z (k,)."""
        pc = (pts_world - self.cam_pos) @ self.R_wc      # = R_wc^T (p - c) for each row
        Z = pc[:, 2]
        uv = np.c_[self.K[0, 0] * pc[:, 0] / Z + self.K[0, 2], self.K[1, 1] * pc[:, 1] / Z + self.K[1, 2]]
        return uv, Z

    # ---------------- timing ----------------
    def due(self, t):
        return t >= self.next_capture - 1e-9

    def capture(self, t, positions, quats, t_mission=None):
        """Called by the runner at the physics step where a frame is due."""
        self.next_capture += self.period
        self.stats["captured"] += 1
        if self.rng.random() < self.c["drop_prob"]:
            self.stats["dropped"] += 1
            return
        hidden, override = set(), {}
        if self.faults is not None and t_mission is not None:
            hidden, override = self.faults.camera_effects(t_mission, positions)
        frame = self.render(positions, quats, hidden, override)
        self.queue.append((t + self.c["latency"], t, frame))

    def deliver(self, t):
        """Frames whose 80 ms latency has elapsed: list of (t_capture, image)."""
        out = []
        while self.queue and self.queue[0][0] <= t + 1e-9:
            _, t_cap, img = self.queue.popleft()
            out.append((t_cap, img))
        return out

    # ---------------- rendering ----------------
    def render(self, positions, quats, hidden=(), override=None):
        override = override or {}
        img = np.full((self.H, self.W), self.c["background_gray"], np.float32)
        for i in range(self.n):
            if i in hidden:
                continue
            p = override.get(i, positions[i])
            R = Rotation.from_quat(quats[i]).as_matrix()            # body -> world
            corners_w = p + self.plate_body @ R.T
            uv, Z = self.project(corners_w)
            if np.any(Z <= 0.05):
                continue
            if self.c["corner_jitter_px"] > 0:
                uv = uv + self.rng.normal(0, self.c["corner_jitter_px"], uv.shape)
            x0, y0 = np.floor(uv.min(0)).astype(int) - 2
            x1, y1 = np.ceil(uv.max(0)).astype(int) + 2
            x0c, y0c, x1c, y1c = max(x0, 0), max(y0, 0), min(x1, self.W), min(y1, self.H)
            if x1c <= x0c or y1c <= y0c:
                continue                                              # outside the image
            dst = np.float32(uv - [x0c, y0c])
            Hm = cv2.getPerspectiveTransform(self.tile_src, dst)
            size = (x1c - x0c, y1c - y0c)
            warped = cv2.warpPerspective(self.tiles[i].astype(np.float32), Hm, size, flags=cv2.INTER_LINEAR)
            mask = cv2.warpPerspective(np.ones_like(self.tiles[i], np.float32), Hm, size, flags=cv2.INTER_LINEAR)
            roi = img[y0c:y1c, x0c:x1c]
            img[y0c:y1c, x0c:x1c] = roi * (1 - mask) + warped * mask
        img += self.rng.normal(0, self.c["pixel_noise_sigma"], img.shape)
        return np.clip(img, 0, 255).astype(np.uint8)
