"""The PLANT: real physics (PyBullet via gym-pybullet-drones, Crazyflie cf2x)
with ground effect + drag + downwash (Physics.PYB_GND_DRAG_DW), plus the PS's
imperfections:  drift push, battery sag.  Only PlutoInterface (the drone's
firmware), the camera model and the metrics code are allowed to read it.
"""
import numpy as np
import pybullet as p


class SimWorld:
    def __init__(self, cfg, spawn_xyz, rng, gui=False):
        from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
        from gym_pybullet_drones.utils.enums import DroneModel, Physics

        s = cfg["sim"]
        self.cfg = cfg
        self.n = len(spawn_xyz)
        self.hz = s["physics_hz"]
        self.dt = 1.0 / self.hz
        kw = dict(drone_model=DroneModel.CF2X, num_drones=self.n,
                  initial_xyzs=np.asarray(spawn_xyz), initial_rpys=np.zeros((self.n, 3)),
                  physics=getattr(Physics, s["physics"]), gui=gui, record=False,
                  obstacles=False, user_debug_gui=False)
        try:                                   # gym-pybullet-drones >= 2.0
            self.env = CtrlAviary(pyb_freq=self.hz, ctrl_freq=self.hz, **kw)
        except TypeError:                      # older API
            self.env = CtrlAviary(freq=self.hz, aggregate_phy_steps=1, **kw)
        self._guard_downwash(s.get("dw_min_dz", 0.05))
        out = self.env.reset()
        self.obs = out[0] if isinstance(out, tuple) else out
        e = self.env
        self.mass = float(e.M)
        self.max_rpm = float(e.MAX_RPM)
        self.max_thrust = 4 * e.KF * self.max_rpm**2
        self.client = e.CLIENT
        self.t = 0.0
        self.plutos = [None] * self.n

        d = cfg["disturbances"]
        self.drift = rng.uniform(-d["drift_accel_max"], d["drift_accel_max"], (self.n, 3))
        if gui:
            self._draw_fence()

    def _guard_downwash(self, min_dz):
        """Same downwash model as gym-pybullet-drones, with one guard.

        Upstream force ~ (R / 4dz)^2 blows up as dz -> 0, so drones resting on
        the ground (sub-mm height differences) get thrown metres away. The
        model only describes a drone flying in another's wake, so we skip pairs
        with dz < min_dz or whose upper drone has its motors off."""
        e = self.env

        def downwash(nth):
            for i in range(e.NUM_DRONES):
                dz = e.pos[i, 2] - e.pos[nth, 2]
                dxy = np.linalg.norm(e.pos[i, 0:2] - e.pos[nth, 0:2])
                if dz < min_dz or dxy >= 10 or not np.any(e.last_clipped_action[i] > 0):
                    continue
                alpha = e.DW_COEFF_1 * (e.PROP_RADIUS / (4 * dz)) ** 2
                beta = e.DW_COEFF_2 * dz + e.DW_COEFF_3
                p.applyExternalForce(e.DRONE_IDS[nth], 4,
                                     forceObj=[0, 0, -alpha * np.exp(-.5 * (dxy / beta) ** 2)],
                                     posObj=[0, 0, 0], flags=p.LINK_FRAME, physicsClientId=e.CLIENT)

        e._downwash = downwash

    def model_info(self):
        e = self.env
        return {"mass_kg": self.mass, "kf": float(e.KF), "km": float(e.KM), "arm_m": float(e.L),
                "thrust2weight": float(e.THRUST2WEIGHT_RATIO), "max_rpm": self.max_rpm,
                "max_thrust_N": self.max_thrust, "hover_rpm": float(e.HOVER_RPM),
                "Ixx": float(e.J[0, 0]), "Iyy": float(e.J[1, 1]), "Izz": float(e.J[2, 2]),
                "hover_throttle_theory": 1000 + 1000 * self.mass * 9.81 / self.max_thrust}

    def attach(self, pluto):
        self.plutos[pluto.id] = pluto

    def battery_scale(self, t=None):
        d = self.cfg["disturbances"]
        t = self.t if t is None else t
        return 1.0 - (1.0 - d["battery_final"]) * min(t / d["battery_sag_time"], 1.0)

    # ---- on-board IMU (used ONLY inside PlutoInterface firmware) ----
    def imu_quat(self, i):
        return self.obs[i][3:7]

    def imu_yaw(self, i):
        return float(self.obs[i][9])

    # ---- one physics step (all drones) ----
    def step(self):
        sag = self.battery_scale()
        rpms = np.zeros((self.n, 4))
        for i, pl in enumerate(self.plutos):
            if pl is not None:
                rpms[i] = pl.firmware_step(self.dt, self.imu_quat(i), sag)
        for i in range(self.n):                    # constant drift push (world frame)
            pos = self.obs[i][0:3]
            p.applyExternalForce(self.env.DRONE_IDS[i], -1, forceObj=(self.mass * self.drift[i]).tolist(),
                                 posObj=pos.tolist(), flags=p.WORLD_FRAME, physicsClientId=self.client)
        out = self.env.step(rpms)
        self.obs = out[0]
        self.t += self.dt

    # ---- ground truth (camera rendering + metrics ONLY) ----
    def ground_truth(self):
        o = np.asarray(self.obs)
        return o[:, 0:3].copy(), o[:, 3:7].copy(), o[:, 10:13].copy(), o[:, 7:10].copy()

    def _draw_fence(self):
        fx, fy, fz = (self.cfg["world"]["geofence"][k] for k in ("x", "y", "z"))
        cs = [(x, y, z) for x in fx for y in fy for z in fz]
        for a in cs:
            for b in cs:
                if sum(u != v for u, v in zip(a, b)) == 1 and a < b:
                    p.addUserDebugLine(a, b, [1, 0, 0], 1.5, physicsClientId=self.client)

    def close(self):
        self.env.close()
