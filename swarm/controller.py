"""D. Cascaded controller: position (P) -> velocity (PI) -> acceleration -> RC.

    v_cmd = Kp_pos (p_ref - p) + v_ff                      (clamped to vmax)
    a_cmd = Kp_vel (v_cmd - v) + I + a_ff                  (clamped to amax)
    I    += Ki_vel (v_cmd - v) dt   (only if not pushing a saturated output; |I| <= Imax)

The integrator I (units m/s^2) is what beats the unknowns without being told:
  * x,y : a constant drift push d needs a constant tilt  ->  I_xy -> -d
  * z   : the hover throttle is unknown and battery sag raises it over time
          ->  I_z learns it.   learned hover = guess + k_thr * I_z
Acceleration -> RC (small-angle hover model, see report Math 5):
  world -> body: a_b = Rz(-yaw_est) a_xy     (yaw from VISION, not the IMU)
  pitch = atan(a_bx / g),   roll = atan(-a_by cos(pitch) / g)
  throttle = 1000 + (guess - 1000 + k_thr a_z) / (cos(roll) cos(pitch))
  rc = 1500 + angle / 15deg * 500 ; yaw_rate = Kpsi * yaw_err
"""
import numpy as np

G = 9.81


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class SwarmController:
    def __init__(self, cfg, n):
        self.c = cfg["control"]
        self.p = cfg["pluto"]
        self.n = n
        self.dt = 1.0 / cfg["sim"]["control_hz"]
        self.I = np.zeros((n, 3))
        self.tilt_max = np.radians(self.p["max_tilt_deg"])
        self.rate_max = np.radians(self.p["max_yaw_rate_deg"])

    def reset(self, i):
        self.I[i] = 0.0

    def hover_throttle(self, i):
        return self.c["hover_throttle_guess"] + self.c["k_throttle"] * self.I[i, 2]

    # --------------------------------------------------------------
    def track(self, i, p, v, yaw, p_ref, v_ff, a_ff, yaw_ref=0.0, integrate=True):
        c = self.c
        kp_pos = np.array([c["pos_kp_xy"], c["pos_kp_xy"], c["pos_kp_z"]])
        kp_vel = np.array([c["vel_kp_xy"], c["vel_kp_xy"], c["vel_kp_z"]])
        ki_vel = np.array([c["vel_ki_xy"], c["vel_ki_xy"], c["vel_ki_z"]])

        # ---- outer loop: position -> velocity command
        v_cmd = kp_pos * (p_ref - p) + v_ff
        sp = np.linalg.norm(v_cmd[:2])
        if sp > c["vmax_xy"]:
            v_cmd[:2] *= c["vmax_xy"] / sp
        v_cmd[2] = np.clip(v_cmd[2], -c["vmax_z"], c["vmax_z"])

        # ---- inner loop: velocity -> acceleration command (PI + feed-forward)
        e_v = v_cmd - v
        a_unsat = kp_vel * e_v + self.I[i] + a_ff
        a = self._clamp_acc(a_unsat)

        # ---- anti-windup: conditional integration + hard clamp
        if integrate:
            sat = np.abs(a - a_unsat) > 1e-9
            pushing = np.sign(e_v) == np.sign(a_unsat - a)
            dI = ki_vel * e_v * self.dt
            dI[sat & pushing] = 0.0
            self.I[i] += dI
            self.I[i, :2] = np.clip(self.I[i, :2], -c["imax_xy"], c["imax_xy"])
            self.I[i, 2] = np.clip(self.I[i, 2], -c["imax_z"], c["imax_z"])
        return self.acc_to_rc(a, yaw, yaw_ref), a

    def trim_only(self, i, yaw, yaw_ref=0.0, extra_az=0.0):
        """No position info (stale / lost): fly level on the learned trims.
        I_xy still cancels the drift, I_z still holds altitude (open loop)."""
        a = self.I[i].copy()
        a[2] += extra_az
        return self.acc_to_rc(self._clamp_acc(a), yaw, yaw_ref), a

    def _clamp_acc(self, a):
        c = self.c
        a = a.copy()
        n = np.linalg.norm(a[:2])
        if n > c["amax_xy"]:
            a[:2] *= c["amax_xy"] / n
        a[2] = np.clip(a[2], -c["amax_z_down"], c["amax_z_up"])
        return a

    def acc_to_rc(self, a, yaw, yaw_ref):
        c, P = self.c, self.p
        cy, sy = np.cos(yaw), np.sin(yaw)
        a_bx = cy * a[0] + sy * a[1]                 # world -> body (heading) frame
        a_by = -sy * a[0] + cy * a[1]
        pitch = np.clip(np.arctan2(a_bx, G), -self.tilt_max, self.tilt_max)
        roll = np.clip(np.arctan2(-a_by * np.cos(pitch), G), -self.tilt_max, self.tilt_max)
        thr = 1000 + (c["hover_throttle_guess"] - 1000 + c["k_throttle"] * a[2]) / (np.cos(roll) * np.cos(pitch))
        yaw_rate = np.clip(c["yaw_kp"] * wrap(yaw_ref - yaw), -self.rate_max, self.rate_max)
        half = (P["rc_max"] - P["rc_min"]) / 2
        mid = P["rc_mid"]
        rc = {"roll": mid + roll / self.tilt_max * half,
              "pitch": mid + pitch / self.tilt_max * half,
              "yaw": mid + yaw_rate / self.rate_max * half,
              "throttle": thr}
        return {k: int(np.clip(round(v), P["rc_min"], P["rc_max"])) for k, v in rc.items()}
