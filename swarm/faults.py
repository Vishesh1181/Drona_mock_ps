"""Fault drills - injected at the CAMERA (what the controller sees), never in the controller.
F1: drone 2's marker hidden for 0.4 s.
F3: one single frame shows drone 1 displaced by 1 m (an outlier).
Times are seconds after arming."""
import numpy as np


class FaultInjector:
    def __init__(self, cfg, names, camera):
        self.cfg = cfg
        self.f = cfg["faults"]
        self.names = set(names)
        self.cam = camera
        self.f3_done = False
        self.log = []

    def camera_effects(self, tm, positions):
        hidden, override = set(), {}
        if "F1" in self.names:
            f = self.f["F1"]
            if f["start"] <= tm < f["start"] + f["duration"]:
                hidden.add(f["drone"])
        if "F3" in self.names and not self.f3_done and tm >= self.f["F3"]["start"]:
            f = self.f["F3"]
            i = f["drone"]
            fake = self._jump(positions, i, f["jump"])
            if fake is not None:
                override[i] = fake
                self.f3_done = True
                self.log.append((tm, i, fake))
                print(f"[fault F3 t={tm:.2f}s] frame shows drone {i} at {np.round(fake, 2)} "
                      f"(true {np.round(positions[i], 2)})")
        return hidden, override

    def _jump(self, P, i, dist):
        """1 m jump in a direction that stays inside the image and off other markers."""
        for ang in np.radians(np.arange(0, 360, 30)):
            q = P[i] + dist * np.array([np.cos(ang), np.sin(ang), 0.0])
            uv, Z = self.cam.project(q[None, :])
            u, v = uv[0]
            inside = 40 < u < self.cam.W - 40 and 40 < v < self.cam.H - 40
            clear = all(np.linalg.norm(q[:2] - P[j, :2]) > 0.2 for j in range(len(P)) if j != i)
            if inside and clear:
                return q
        return None
