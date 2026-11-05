"""Math 5 check: predicted vs simulated step response of the x-axis loop.

Linearised hover model + our cascade (ignoring delays):
   x'' = Kv (Kp (x_ref - x) - x')   ->   x'' + Kv x' + Kv Kp x = Kv Kp x_ref
   wn = sqrt(Kv Kp),  zeta = 0.5 sqrt(Kv / Kp)
We also predict a 'realistic' version: the velocity integrator (Ki), ~0.10 s feedback delay (camera 80 ms
+ half a 40 ms control period) and a 50 ms first-order tilt (attitude) lag,
by integrating the same linear equations numerically. Then we fly the real thing in PyBullet: hover at (x0, y0, 1), step
x_ref by +0.3 m, and plot all three.        python step_test.py
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from swarm.config import load_config
from swarm.planner import min_jerk
from swarm.runner import run


class StepPlan:
    """Takeoff 3 s -> hold -> at t_step jump x_ref by `step` -> hold -> land."""
    FENCE = ("hold", "step")

    def __init__(self, start, t_step=8.0, step=0.3, h=1.0):
        self.start = start.copy()
        self.n = len(start)
        self.top = start.copy()
        self.top[:, 2] = h
        self.t_step, self.step = t_step, step
        self.T_total = t_step + 8.0 + 4.0
        self.slot_index = np.arange(self.n)
        self.phase = np.zeros(self.n)

    def phase_at(self, tau):
        if tau < 3:
            return "takeoff", tau
        if tau < self.t_step:
            return "hold", tau - 3
        if tau < self.t_step + 8:
            return "step", tau - self.t_step
        if tau < self.T_total:
            return "land", tau - self.t_step - 8
        return "done", 0.0

    def fence_active(self, tau):
        return self.phase_at(tau)[0] in self.FENCE

    def sample(self, tau):
        ph, t = self.phase_at(tau)
        z = np.zeros((self.n, 3))
        if ph == "takeoff":
            s, ds, dds = min_jerk(t / 3)
            d = self.top - self.start
            return self.start + d * s, d * ds / 3, d * dds / 9
        p = self.top.copy()
        if ph in ("step", "land", "done"):
            p[:, 0] += self.step
        if ph in ("land", "done"):
            g = p.copy()
            g[:, 2] = self.start[:, 2]
            s, ds, dds = min_jerk(min(t / 4, 1))
            return p + (g - p) * s, (g - p) * ds / 4, (g - p) * dds / 16
        return p, z, z


def predict(c, step, t, delay=0.0, tau_att=0.0):
    kp, kv = c["pos_kp_xy"], c["vel_kp_xy"]
    wn, zeta = np.sqrt(kv * kp), 0.5 * np.sqrt(kv / kp)
    wd = wn * np.sqrt(1 - zeta**2)
    ideal = step * (1 - np.exp(-zeta * wn * t) * (np.cos(wd * t) + zeta / np.sqrt(1 - zeta**2) * np.sin(wd * t)))
    dt = t[1] - t[0]
    ki = c["vel_ki_xy"]
    x, v, acc, I, buf = 0.0, 0.0, 0.0, 0.0, []
    nd = int(round(delay / dt))
    out = []
    for _ in t:                                   # same linear ODE, delayed feedback
        buf.append((x, v))
        xd, vd = buf[-1 - nd] if len(buf) > nd else (0.0, 0.0)
        e_v = kp * (step - xd) - vd
        I += ki * e_v * dt                                    # velocity-loop integrator
        a_cmd = kv * e_v + I
        acc += (a_cmd - acc) * dt / tau_att if tau_att > 0 else a_cmd - acc   # attitude (tilt) lag
        v += acc * dt
        x += v * dt
        out.append(x)
    return ideal, np.array(out), wn, zeta


ap = argparse.ArgumentParser()
ap.add_argument("--config", default="config.yaml")
ap.add_argument("--gui", action="store_true")
a = ap.parse_args()
cfg = load_config(a.config)
c = cfg["control"]
holder = {}


def factory(start):
    holder["plan"] = StepPlan(start)
    return holder["plan"]


df = run(cfg, 1, "step", gui=a.gui, plan_factory=factory, tag="step_test")
plan = holder["plan"]
g = df[(df.drone == 0) & (df.t >= plan.t_step - 1) & (df.t <= plan.t_step + 6)]
t = g.t.values - plan.t_step
x = g.x.values - plan.top[0, 0]
tt = np.linspace(0, 6, 3001)
ideal, delayed, wn, zeta = predict(c, plan.step, tt, delay=0.10, tau_att=0.05)
os_sim = (x[t > 0].max() - plan.step) / plan.step * 100
os_id = (ideal.max() - plan.step) / plan.step * 100


def t_settle(tv, xv):
    bad = np.where(np.abs(xv - plan.step) > 0.05 * plan.step)[0]
    return tv[bad[-1]] if len(bad) else 0.0


print(f"wn={wn:.2f} rad/s zeta={zeta:.2f} | overshoot predicted {os_id:.1f}% "
      f"(realistic {(delayed.max() - plan.step) / plan.step * 100:.1f}%) simulated {os_sim:.1f}% | "
      f"5% settling predicted {t_settle(tt, ideal):.2f}s simulated {t_settle(t[t > 0], x[t > 0]):.2f}s")
plt.figure(figsize=(8, 4.5))
plt.plot(tt, ideal, "k--", label=f"predicted 2nd order (wn={wn:.2f}, zeta={zeta:.2f})")
plt.plot(tt, delayed, "C1:", label="predicted + integrator + 0.10 s delay + 50 ms tilt lag")
plt.plot(t, x, "C0", lw=2, label="PyBullet (true x)")
plt.axhline(plan.step, color="gray", lw=.5)
plt.xlabel("time after step [s]"), plt.ylabel("x - x0 [m]"), plt.grid(alpha=.3), plt.legend()
plt.title("x-axis step response: predicted vs simulated")
out = os.path.join(cfg["logging"]["out_dir"], "step_test", "step_response.png")
plt.tight_layout()
plt.savefig(out, dpi=130)
print("saved", out)
