"""1-minute plumbing test - RUN THIS FIRST after installing.
Uses ground truth ON PURPOSE (it is a test of PlutoInterface, not the controller).
Checks: hover throttle near 1444, pitch>1500 -> +x, roll>1500 -> -y (right),
yaw>1500 -> yaw increases.   python sanity_check.py [--gui]"""
import argparse
import numpy as np
from swarm.config import load_config
from swarm.sim_world import SimWorld
from swarm.pluto_interface import PlutoInterface

ap = argparse.ArgumentParser()
ap.add_argument("--gui", action="store_true")
a = ap.parse_args()
cfg = load_config()
cfg["disturbances"]["drift_accel_max"] = 0.0          # clean test
w = SimWorld(cfg, np.array([[0.0, 0.0, 0.05]]), np.random.default_rng(0), gui=a.gui)
info = w.model_info()
print("model:", {k: round(v, 6) for k, v in info.items()})
pl = PlutoInterface(0, w, cfg)
pl.arm()
hover = info["hover_throttle_theory"]


def fly(sec, roll=1500, pitch=1500, yaw=1500, thr=None, zref=None):
    for _ in range(int(sec * w.hz)):
        p, _, v, _ = w.ground_truth()
        t = thr if thr is not None else hover + 300 * (zref - p[0, 2]) - 150 * v[0, 2]   # test-only z hold
        pl.set_rc(roll, pitch, yaw, t)
        w.step()
    return w.ground_truth()


fly(3.0, zref=1.0)
p0, _, _, r0 = w.ground_truth()
print(f"climb to 1 m with hover~{hover:.0f}: z = {p0[0, 2]:.2f} m (expect ~1.0)")
p1, _, v1, _ = fly(0.6, pitch=1600, zref=1.0)
print(f"pitch 1600 for 0.6 s: vx = {v1[0, 0]:+.2f} m/s  -> {'OK (+x)' if v1[0, 0] > 0.1 else 'WRONG SIGN!'}")
fly(1.5, zref=1.0)
p2, _, v2, _ = fly(0.6, roll=1600, zref=1.0)
print(f"roll 1600 for 0.6 s:  vy = {v2[0, 1]:+.2f} m/s  -> {'OK (-y, right)' if v2[0, 1] < -0.1 else 'WRONG SIGN!'}")
fly(1.5, zref=1.0)
_, _, _, ra = w.ground_truth()
_, _, _, rb = fly(0.5, yaw=1600, zref=1.0)
print(f"yaw 1600 for 0.5 s: dyaw = {np.degrees(rb[0, 2] - ra[0, 2]):+.1f} deg (expect about +9)")
w.close()
