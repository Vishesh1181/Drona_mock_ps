"""C. Vision localisation + D. smoothing of the estimate.

frame -> ArUco detect (sub-pixel corners) -> level-marker pinhole model
      -> world pose (x, y, z, yaw) -> outlier gate -> alpha-beta filter.
Status per drone from the age of the last ACCEPTED measurement:
      valid (<=150 ms) / stale (>150 ms) / lost (>300 ms).
Latency compensation: estimate(t_now) predicts p + v * (t_now - t_capture).
"""
import cv2
import numpy as np

VALID, STALE, LOST = "valid", "stale", "lost"


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class Track:
    """Alpha-beta filter on position (3) + low-pass on yaw, timestamped at capture time."""

    def __init__(self, lc):
        self.lc = lc
        self.init = False
        self.p = np.zeros(3)
        self.v = np.zeros(3)
        self.yaw = 0.0
        self.t = -np.inf
        self.rejects = 0

    def update(self, t, z, yaw):
        """Returns 'init' | 'ok' | 'outlier' | 'reacquired'."""
        lc = self.lc
        if not self.init:
            self.p, self.v, self.yaw, self.t, self.init = z.copy(), np.zeros(3), yaw, t, True
            return "init"
        dt = max(t - self.t, 1e-3)
        p_pred = self.p + self.v * dt
        r = z - p_pred
        gate = lc["max_speed"] * dt + lc["gate_margin"]          # physically possible motion
        if np.linalg.norm(r) > gate:
            self.rejects += 1
            if self.rejects >= lc["reacquire_after"]:               # camera keeps insisting -> trust it
                self.p, self.v, self.yaw, self.t, self.rejects = z.copy(), np.zeros(3), yaw, t, 0
                return "reacquired"
            return "outlier"
        self.rejects = 0
        a = np.array([lc["alpha_xy"], lc["alpha_xy"], lc["alpha_z"]])   # z is ~5x noisier
        b = np.array([lc["beta_xy"], lc["beta_xy"], lc["beta_z"]])     # than x,y -> softer gains
        self.p = p_pred + a * r                                     # position correction
        self.v = self.v + (b / dt) * r                              # velocity correction
        self.yaw = wrap(self.yaw + lc["alpha_yaw"] * wrap(yaw - self.yaw))
        self.t = t
        return "ok"


class Localizer:
    def __init__(self, cfg, n, camera):
        c = cfg["camera"]
        self.lc = cfg["localization"]
        self.n = n
        self.K = camera.K
        self.cam_pos = camera.cam_pos
        self.R_wc = camera.R_wc
        d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, c["aruco_dict"]))
        prm = cv2.aruco.DetectorParameters()
        prm.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector = cv2.aruco.ArucoDetector(d, prm)
        self.marker_size = c["marker_size"]
        self.size_bias = self.lc["size_bias_px"]                    # detector shrinks markers ~0.35 px
        self.h_marker = c["marker_height"]
        self.tracks = [Track(self.lc) for _ in range(n)]
        self.last_raw = [None] * n                                  # for metrics / plots

    def measure(self, img):
        """Raw (unfiltered) measurements in this frame: {id: (pos, yaw)}.

        Pixel -> world (camera at height Hc looking down, level marker):
          depth  Z = fx * s / side_px          (s = 8 cm marker, side_px = apparent size)
          x = (u - cx) * Z / fx,   y = -(v - cy) * Z / fy,   z = Hc - Z - h_marker
        (u, v) = intersection of the marker diagonals (exact image of the centre
        under perspective). We use this 'level-marker' model instead of full
        6-DoF PnP: an 8 cm marker is only ~36 px wide, so PnP's tilt estimate is
        ill-conditioned and leaks into position (tested: 2-3x worse). The drone
        is within a few degrees of level, so the model error is tiny."""
        corners, ids, _ = self.detector.detectMarkers(img)
        out = {}
        if ids is None:
            return out
        fx, fy, cx, cy = self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]
        for c, i in zip(corners, ids.flatten()):
            i = int(i)
            if i >= self.n:
                continue
            uv = c.reshape(4, 2).astype(float)                       # TL, TR, BR, BL
            side = np.mean(np.linalg.norm(uv - np.roll(uv, -1, axis=0), axis=1)) + self.size_bias
            Z = fx * self.marker_size / side
            p1, p2, p3, p4 = uv
            A = np.c_[p3 - p1, -(p4 - p2)]
            if abs(np.linalg.det(A)) < 1e-6:
                continue
            lam = np.linalg.solve(A, p2 - p1)[0]
            u, v = p1 + lam * (p3 - p1)                              # diagonal intersection
            pos = np.array([(u - cx) * Z / fx, -(v - cy) * Z / fy, self.cam_pos[2] - Z - self.h_marker])
            e = (uv[1] - uv[0]) + (uv[2] - uv[3])                    # marker +x edge direction in image
            yaw = np.arctan2(-e[1], e[0])                            # image v points along -y_world
            out[i] = (pos, yaw)
        return out

    def process(self, img, t_capture):
        """Update tracks with one delivered frame. Returns list of (id, result) events."""
        events = []
        for i, (pos, yaw) in self.measure(img).items():
            self.last_raw[i] = (t_capture, pos)
            res = self.tracks[i].update(t_capture, pos, yaw)
            if res in ("outlier", "reacquired"):
                events.append((i, res, pos))
        return events

    def estimate(self, t_now):
        """Latency-compensated estimates at t_now: pos (n,3), vel (n,3), yaw (n,), status list, age."""
        P, V, Y, S, AGE = np.zeros((self.n, 3)), np.zeros((self.n, 3)), np.zeros(self.n), [], np.zeros(self.n)
        for i, tr in enumerate(self.tracks):
            age = t_now - tr.t if tr.init else np.inf
            AGE[i] = age
            h = min(max(age, 0.0), self.lc["max_extrapolation"])
            P[i] = tr.p + tr.v * h
            V[i] = tr.v
            Y[i] = tr.yaw
            S.append(VALID if age <= self.lc["stale_s"] else STALE if age <= self.lc["lost_s"] else LOST)
        return P, V, Y, S, AGE
