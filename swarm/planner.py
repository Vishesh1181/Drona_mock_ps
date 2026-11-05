"""A. Formation + wave planner (generic in N, one shared mission clock tau).

Mission: takeoff -> transit to slots -> hold -> wave x cycles -> hold -> land.
Every segment is C2-continuous (position, velocity, acceleration), so the
controller gets smooth feed-forward and nothing teleports.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment


def min_jerk(u):
    """Minimum-jerk (quintic) profile s(u), u in [0,1].
    s = 10u^3 - 15u^4 + 6u^5  ->  s(0)=0, s(1)=1 and s', s'' are 0 at both ends.
    Returns s, ds/du, d2s/du2."""
    u = np.clip(u, 0.0, 1.0)
    s = 10 * u**3 - 15 * u**4 + 6 * u**5
    ds = 30 * u**2 - 60 * u**3 + 30 * u**4
    dds = 60 * u - 180 * u**2 + 120 * u**3
    return s, ds, dds


def formation_slots(kind, n, m):
    """Slot positions (n,3) and wave phase offsets (n,) for slot index i."""
    h = m["height"]
    i = np.arange(n)
    if kind == "line":
        d = m["line_spacing"]
        x = (i - (n - 1) / 2.0) * d
        slots = np.c_[x, np.zeros(n), np.full(n, h)]
        phases = i * np.pi / 2.0                       # PS: wave phase i*pi/2
    elif kind == "circle":
        R = m["circle_radius"]
        th = 2 * np.pi * i / n
        slots = np.c_[R * np.cos(th), R * np.sin(th), np.full(n, h)]
        phases = th                                    # PS: wave phase theta_i
    else:
        raise ValueError(f"unknown formation {kind}")
    return slots, phases


def assign_slots(start_xyz, slots):
    """Hungarian assignment minimising the sum of SQUARED distances.
    Squared cost gives a safety guarantee for synchronised straight-line
    transits (CAPT, Turpin et al. 2014): min distance >= D/sqrt(2) where D is
    the smaller of the start / goal separations. Proof is in the report."""
    diff = start_xyz[:, None, :] - slots[None, :, :]
    cost = np.sum(diff**2, axis=2)
    rows, cols = linear_sum_assignment(cost)
    return cols[np.argsort(rows)]                      # drone i -> slot cols[i]


class MissionPlan:
    """Pure function of the mission clock tau: sample(tau) -> pos, vel, acc."""

    FENCE_PHASES = ("transit", "hold", "wave", "hold2")

    def __init__(self, cfg, start_xyz, formation):
        m = cfg["mission"]
        self.cfg = cfg
        self.n = len(start_xyz)
        self.formation = formation
        self.start = np.asarray(start_xyz, float).copy()
        self.h = m["height"]
        self.A = m["wave_amplitude"]
        self.omega = 2 * np.pi / m["wave_period"]
        self.ramp = m["wave_ramp"]
        self.yaw = m["yaw"]

        slots_raw, phases_raw = formation_slots(formation, self.n, m)
        self.lift = self.start.copy()
        self.lift[:, 2] = self.h                       # straight up first
        self.assignment = assign_slots(self.lift, slots_raw)
        self.slot = slots_raw[self.assignment]
        self.phase = phases_raw[self.assignment]       # phase follows the slot
        self.slot_index = self.assignment

        dmax = np.max(np.linalg.norm(self.slot - self.lift, axis=1))
        # min-jerk peak speed = 1.875 * D / T  ->  T = 1.875 * D / vmax
        t_transit = max(m["transit_min_time"], 1.875 * dmax / m["transit_vmax"])
        t_wave = m["wave_cycles"] * m["wave_period"]
        self.segments = [("takeoff", m["takeoff_time"]), ("transit", t_transit),
                         ("hold", m["hold_before"]), ("wave", t_wave),
                         ("hold2", m["hold_after"]), ("land", m["land_time"])]
        self.t_start = {}
        t = 0.0
        for name, dur in self.segments:
            self.t_start[name] = t
            t += dur
        self.T_total = t
        self.durations = dict(self.segments)

    # ------------------------------------------------------------------
    def phase_at(self, tau):
        for name, dur in self.segments:
            if tau < self.t_start[name] + dur:
                return name, tau - self.t_start[name]
        return "done", tau - self.T_total

    def fence_active(self, tau):
        return self.phase_at(tau)[0] in self.FENCE_PHASES

    def _blend(self, a, b, t, T):
        s, ds, dds = min_jerk(t / T)
        return a + (b - a) * s, (b - a) * ds / T, (b - a) * dds / T**2

    def _envelope(self, tw):
        """C2 amplitude envelope e(t): 0 -> 1 over `ramp`, 1 -> 0 at the end."""
        Tw = self.durations["wave"]
        if tw < self.ramp:
            s, ds, dds = min_jerk(tw / self.ramp)
            return s, ds / self.ramp, dds / self.ramp**2
        if tw > Tw - self.ramp:
            s, ds, dds = min_jerk((Tw - tw) / self.ramp)
            return s, -ds / self.ramp, dds / self.ramp**2
        return 1.0, 0.0, 0.0

    def wave_z(self, tw):
        """z_i = h + A e(t) sin(w t - phi_i) and its 1st/2nd derivatives."""
        e, de, dde = self._envelope(tw)
        S = np.sin(self.omega * tw - self.phase)
        C = np.cos(self.omega * tw - self.phase)
        w = self.omega
        z = self.h + self.A * e * S
        dz = self.A * (de * S + e * w * C)
        ddz = self.A * (dde * S + 2 * de * w * C - e * w**2 * S)
        return z, dz, ddz

    def sample(self, tau):
        """Targets for all drones at mission time tau: pos, vel, acc (n,3)."""
        name, t = self.phase_at(tau)
        n = self.n
        vel = np.zeros((n, 3))
        acc = np.zeros((n, 3))
        if name == "takeoff":
            pos, vel, acc = self._blend(self.start, self.lift, t, self.durations["takeoff"])
        elif name == "transit":
            pos, vel, acc = self._blend(self.lift, self.slot, t, self.durations["transit"])
        elif name in ("hold", "hold2"):
            pos = self.slot.copy()
        elif name == "wave":
            pos = self.slot.copy()
            pos[:, 2], vel[:, 2], acc[:, 2] = self.wave_z(t)
        elif name == "land":
            ground = self.slot.copy()
            ground[:, 2] = self.start[:, 2]
            pos, vel, acc = self._blend(self.slot, ground, t, self.durations["land"])
        else:  # done
            pos = self.slot.copy()
            pos[:, 2] = self.start[:, 2]
        return pos, vel, acc

    def summary(self):
        lines = [f"Formation={self.formation} N={self.n}  total={self.T_total:.1f}s"]
        lines += [f"  {k:8s} starts {self.t_start[k]:6.2f}s  lasts {d:5.2f}s" for k, d in self.segments]
        for i in range(self.n):
            lines.append(f"  drone {i} -> slot {self.slot_index[i]} at {np.round(self.slot[i], 3)}"
                         f" phase {np.degrees(self.phase[i]):.0f} deg")
        return "\n".join(lines)


def spawn_positions(cfg, n, rng):
    """Ground grid (spacing from config), centred, IDs optionally shuffled."""
    sp = cfg["sim"]["spawn"]
    s = sp["spacing"]
    ncols = int(np.ceil(np.sqrt(n)))
    nrows = int(np.ceil(n / ncols))
    pts = []
    for r in range(nrows):
        for c in range(ncols):
            if len(pts) < n:
                pts.append([(c - (ncols - 1) / 2) * s, (r - (nrows - 1) / 2) * s, sp["z"]])
    pts = np.array(pts)
    if sp.get("shuffle_ids", False):
        pts = pts[rng.permutation(n)]
    return pts
