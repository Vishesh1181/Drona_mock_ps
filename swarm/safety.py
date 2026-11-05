"""E. Safety supervisor - decides, every control tick, what each drone may do.

Rules (PS):
  stale (>150 ms)          -> that drone HOLDs (flies level on learned trims);
                              shared clock pauses so the wave stays in sync
  lost  (>300 ms)          -> that drone LANDS, the swarm PAUSES until it is down
  separation < 0.35 m      -> EVERYONE holds (emergency_hold_time), then everyone lands
  geofence breach          -> that drone lands
Every transition is logged with a timestamp.

Directives returned per drone:
  ("track",)            follow the mission plan
  ("freeze", pos)       hold this position, no feed-forward
  ("trim", extra_az)    no vision: level flight on integrator trims (+ optional sink)
  ("land", pos, vel)    vision-guided landing target
  ("off",)              disarmed
"""
import numpy as np

from .localization import VALID, STALE, LOST

IDLE, ACTIVE, HOLD, LANDING, LANDED = "IDLE", "ACTIVE", "HOLD", "LANDING", "LANDED"


class SafetySupervisor:
    def __init__(self, cfg, n):
        self.cfg = cfg
        self.s = cfg["safety"]
        self.n = n
        self.dmin = cfg["world"]["min_separation"]
        f = cfg["world"]["geofence"]
        tol = self.s["fence_tolerance"]
        self.lo = np.array([f["x"][0], f["y"][0], f["z"][0]]) - tol
        self.hi = np.array([f["x"][1], f["y"][1], f["z"][1]]) + tol
        self.state = [IDLE] * n
        self.land = [None] * n
        self.swarm = "RUN"
        self.t_emergency = None
        self.freeze_pos = None
        self.z_ground = np.zeros(n)
        self.events = []
        self.fault_landing = set()          # drones whose fault-landing pauses the swarm

    def log(self, t, drone, kind, detail=""):
        self.events.append({"t": round(t, 3), "drone": drone, "event": kind, "detail": detail})
        print(f"[safety t={t:6.2f}s] drone {drone}: {kind} {detail}")

    def arm_all(self, t, z_ground):
        self.z_ground = np.asarray(z_ground, float).copy()
        for i in range(self.n):
            self.state[i] = ACTIVE
        self.log(t, -1, "ARM", "all drones armed, mission start")

    def airborne(self, i):
        return self.state[i] in (ACTIVE, HOLD, LANDING)

    def _start_landing(self, t, i, p, status, reason):
        self.state[i] = LANDING
        mode = "vision" if status == VALID else "blind"
        self.land[i] = {"t0": t, "z0": p[2], "xy": p[:2].copy(), "mode": mode, "tb": t, "z_last": p[2]}
        self.log(t, i, "LAND", f"({reason}, {mode})")

    # ------------------------------------------------------------------
    def step(self, t, plan_phase, fence_active, P, status):
        """P = estimated positions (n,3), status = list of valid/stale/lost."""
        # ---- per-drone vision health ----
        for i in range(self.n):
            if self.state[i] not in (ACTIVE, HOLD):
                continue
            st = status[i]
            if st == LOST:
                self.fault_landing.add(i)
                self._start_landing(t, i, P[i], st, "marker LOST >300 ms")
                self.log(t, -1, "SWARM_PAUSE", f"until drone {i} has landed")
            elif st == STALE and self.state[i] == ACTIVE:
                self.state[i] = HOLD
                self.log(t, i, "STALE_HOLD", "vision >150 ms old")
            elif st == VALID and self.state[i] == HOLD:
                self.state[i] = ACTIVE
                self.log(t, i, "RECOVERED", "fresh vision again")
            # ---- geofence ----
            if (self.state[i] in (ACTIVE, HOLD) and fence_active and st == VALID
                    and np.any((P[i] < self.lo) | (P[i] > self.hi))):
                self._start_landing(t, i, P[i], st, f"GEOFENCE at {np.round(P[i], 2)}")

        # ---- separation (only drones we can actually see, in the air) ----
        if self.swarm == "RUN":
            ids = [i for i in range(self.n) if self.airborne(i) and status[i] == VALID
                   and P[i, 2] > self.z_ground[i] + 0.25]
            for a in range(len(ids)):
                for b in range(a + 1, len(ids)):
                    i, j = ids[a], ids[b]
                    d = np.linalg.norm(P[i] - P[j])
                    if d < self.dmin:
                        self.swarm = "EMERGENCY_HOLD"
                        self.t_emergency = t
                        self.freeze_pos = P.copy()
                        self.log(t, -1, "SEPARATION_BREACH", f"drones {i},{j} at {d:.3f} m -> all hold")
                        break
                if self.swarm != "RUN":
                    break
        if self.swarm == "EMERGENCY_HOLD" and t - self.t_emergency >= self.s["emergency_hold_time"]:
            self.swarm = "EMERGENCY_LAND"
            for i in range(self.n):
                if self.state[i] in (ACTIVE, HOLD):
                    self._start_landing(t, i, P[i], status[i], "after separation breach")

        # ---- directives ----
        directives = []
        for i in range(self.n):
            directives.append(self._directive(t, i, P[i], status[i], plan_phase))

        # mission clock runs only if nothing is wrong
        self.fault_landing = {i for i in self.fault_landing if self.state[i] == LANDING}
        any_hold = any(s == HOLD for s in self.state) and self.s["stale_pauses_swarm"]
        clock = self.swarm == "RUN" and not any_hold and not self.fault_landing
        return directives, clock

    def _directive(self, t, i, p, st, plan_phase):
        s = self.s
        state = self.state[i]
        zg = self.z_ground[i]
        if state in (IDLE, LANDED):
            return ("off",)
        if state == HOLD:
            return ("trim", 0.0)
        if state == ACTIVE:
            if self.swarm == "EMERGENCY_HOLD":
                return ("freeze", self.freeze_pos[i].copy())
            if plan_phase == "done" or (plan_phase == "land" and st == VALID and p[2] < zg + s["disarm_height"]):
                self.state[i] = LANDED
                self.log(t, i, "DISARM", "mission landing complete")
                return ("off",)
            return ("track",)
        # ---- LANDING ----
        L = self.land[i]
        if st == VALID or (st == STALE and L["mode"] == "vision"):   # short gaps: keep guiding
            if L["mode"] == "blind":
                L.update(mode="vision", t0=t, z0=p[2], xy=p[:2].copy())
                self.log(t, i, "LAND_VISION", "vision back, guided landing")
            if p[2] < zg + s["disarm_height"]:
                self.state[i] = LANDED
                self.log(t, i, "DISARM", "landed")
                return ("off",)
            L["z_last"] = p[2]
            z = max(L["z0"] - s["land_speed"] * (t - L["t0"]), zg - 0.02)
            vz = -s["land_speed"] if z > zg else 0.0
            # target has reached the floor: if we are still "above" it for 1.5 s we are
            # sitting on the ground and vision z is just biased/noisy -> touchdown
            if z <= zg - 0.02:
                L.setdefault("t_floor", t)
                if t - L["t_floor"] > s["touchdown_wait"]:
                    self.state[i] = LANDED
                    self.log(t, i, "DISARM", "touchdown (target at floor, no further descent)")
                    return ("off",)
            return ("land", np.r_[L["xy"], z], np.array([0.0, 0.0, vz]))
        # blind: sink 0.3 m/s^2 for 1 s (-> 0.3 m/s), then coast at that speed, then cut
        if L["mode"] == "vision":
            L.update(mode="blind", tb=t)
        tb = t - L["tb"]
        a, T = s["blind_descent_accel"], s["blind_descent_time"]
        v = a * T
        t_cut = T + max(L["z_last"] - zg - 0.5 * a * T**2 - 0.05, 0.0) / v
        if tb > t_cut:
            self.state[i] = LANDED
            self.log(t, i, "DISARM", "blind landing timer")
            return ("off",)
        return ("trim", -a if tb < T else 0.0)
