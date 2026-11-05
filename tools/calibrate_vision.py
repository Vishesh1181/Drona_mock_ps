"""Offline vision calibration + accuracy check (NOT used in the control loop).
Renders frames at known poses, runs the real Localizer and reports bias/std
per height band. The mean apparent-size bias (px) goes into
localization.size_bias_px. Run:  python tools/calibrate_vision.py"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scipy.spatial.transform import Rotation
from swarm.config import load_config
from swarm.camera import OverheadCamera
from swarm.localization import Localizer

cfg = load_config()
rng = np.random.default_rng(3)
n = 4
cam = OverheadCamera(cfg, n, rng)
loc = Localizer(cfg, n, cam)
fx, s = cfg["camera"]["fx"], cfg["camera"]["marker_size"]
for zr in ((0.6, 1.0), (1.0, 1.4)):
    E, B = [], []
    for _ in range(100):
        P = np.c_[rng.uniform(-0.9, 0.9, n), rng.uniform(-0.55, 0.55, n), rng.uniform(*zr, n)]
        eul = np.c_[rng.uniform(-0.3, 0.3, n), rng.uniform(-0.03, 0.03, (n, 2))]
        for i, (p, y) in loc.measure(cam.render(P, [Rotation.from_euler("ZYX", e).as_quat() for e in eul])).items():
            E.append(p - P[i])
            Zt, Zm = 3 - P[i, 2] - 0.015, 3 - p[2] - 0.015
            B.append(fx * s / Zt - fx * s / Zm)          # remaining size bias in px
    E = np.array(E)
    print(f"z in {zr}: bias mm {np.round(E.mean(0)*1000, 1)}  std mm {np.round(E.std(0)*1000, 1)}"
          f"  residual size bias {np.mean(B):+.3f} px (add to size_bias_px)")
