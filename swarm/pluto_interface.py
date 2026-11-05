"""PlutoInterface - the ONLY door between our code and a drone.

Public API (identical to what a real Pluto accepts over Wi-Fi/MSP):
    set_rc(roll, pitch, yaw, throttle)   each 1000..2000, 1500 = neutral
    arm(), disarm()

Everything below `firmware_step` emulates the drone's FIRMWARE ("angle mode"):
    rc_roll / rc_pitch -> target roll / pitch angle, +-15 deg (body frame)
    rc_yaw             -> target yaw RATE, +-90 deg/s (integrated to a yaw setpoint)
    rc_throttle        -> collective thrust: 1000 = 0, 2000 = max thrust (which sags)
The attitude loop itself is gym-pybullet-drones' DSLPIDControl attitude part
(the "sim's built-in low-level attitude controller" the PS allows). It uses the
drone's on-board IMU attitude, exactly like the real flight controller does.
Our planner/controller never see that IMU and never touch the simulator.
"""
import numpy as np
from scipy.spatial.transform import Rotation


class PlutoInterface:
    def __init__(self, drone_id, world, cfg):
        from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
        from gym_pybullet_drones.utils.enums import DroneModel

        self.id = drone_id
        self.world = world
        self.p = cfg["pluto"]
        self.att = DSLPIDControl(drone_model=DroneModel.CF2X)   # used ONLY for its attitude loop
        self.armed = False
        self.rc = {"roll": 1500, "pitch": 1500, "yaw": 1500, "throttle": 1000}
        self.yaw_sp = 0.0
        world.attach(self)

    # ----------------------------- public API -----------------------------
    def set_rc(self, roll, pitch, yaw, throttle):
        lo, hi = self.p["rc_min"], self.p["rc_max"]
        self.rc = {"roll": int(np.clip(roll, lo, hi)), "pitch": int(np.clip(pitch, lo, hi)),
                   "yaw": int(np.clip(yaw, lo, hi)), "throttle": int(np.clip(throttle, lo, hi))}

    def arm(self):
        self.armed = True
        self.att.reset()
        self.yaw_sp = self.world.imu_yaw(self.id)
        self.rc = {"roll": 1500, "pitch": 1500, "yaw": 1500, "throttle": 1000}

    def disarm(self):
        self.armed = False

    # ----------------------------- firmware -------------------------------
    def firmware_step(self, dt, imu_quat, max_thrust_scale):
        """Called by the simulated drone at the physics rate. Returns 4 motor RPMs."""
        if not self.armed:
            return np.zeros(4)
        w = self.world
        mid, half = self.p["rc_mid"], (self.p["rc_max"] - self.p["rc_min"]) / 2
        tilt = np.radians(self.p["max_tilt_deg"])
        roll = (self.rc["roll"] - mid) / half * tilt          # + = roll right
        pitch = (self.rc["pitch"] - mid) / half * tilt        # + = nose down = forward
        yaw_rate = (self.rc["yaw"] - mid) / half * np.radians(self.p["max_yaw_rate_deg"])
        self.yaw_sp = (self.yaw_sp + yaw_rate * dt + np.pi) % (2 * np.pi) - np.pi

        # body-frame angle mode: R_des = Rz(yaw_sp) Ry(pitch) Rx(roll)
        # (DSL expects intrinsic 'XYZ' Euler angles of the desired rotation)
        target_euler = Rotation.from_euler("ZYX", [self.yaw_sp, pitch, roll]).as_euler("XYZ")

        # throttle -> thrust; a sagging battery gives less thrust for the same stick
        frac = (self.rc["throttle"] - self.p["rc_min"]) / (self.p["rc_max"] - self.p["rc_min"])
        thrust = frac * w.max_thrust * max_thrust_scale                     # N, total
        a = self.att
        pwm = (np.sqrt(thrust / (4 * a.KF)) - a.PWM2RPM_CONST) / a.PWM2RPM_SCALE
        rpm = a._dslPIDAttitudeControl(dt, pwm, imu_quat, target_euler, np.array([0.0, 0.0, yaw_rate]))
        return np.clip(rpm, 0.0, w.max_rpm * np.sqrt(max_thrust_scale))   # T ~ rpm^2
