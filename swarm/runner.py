"""Main loop. One control tick (25 Hz):

   camera frames that have arrived (80 ms late) -> Localizer
   estimate(t) -> SafetySupervisor -> directives + "may the clock run?"
   MissionPlan.sample(tau) -> SwarmController -> PlutoInterface.set_rc()
   then 10 physics steps (250 Hz); the camera captures at 30 fps meanwhile.

Ground truth is read ONLY by the camera model (to render) and the logger
(to compute errors). The controller only ever sees Localizer estimates.
"""
import json
import os
import time

import numpy as np
import pandas as pd

from .camera import OverheadCamera
from .controller import SwarmController
from .faults import FaultInjector
from .localization import Localizer, VALID
from .planner import MissionPlan, spawn_positions
from .safety import SafetySupervisor, LANDED, IDLE
from .validator import validate


def run(cfg, n, formation, faults=(), gui=False, live=False, show_cam=False, out_dir=None,
        plan_factory=None, tag=None, world_cls=None, pluto_cls=None):
    if world_cls is None:
        from .sim_world import SimWorld as world_cls
    if pluto_cls is None:
        from .pluto_interface import PlutoInterface as pluto_cls

    rng = np.random.default_rng(cfg["seed"])
    tag = tag or f"{formation}_N{n}" + (f"_{'_'.join(faults)}" if faults else "")
    out_dir = out_dir or os.path.join(cfg["logging"]["out_dir"], tag)
    os.makedirs(out_dir, exist_ok=True)

    spawn = spawn_positions(cfg, n, rng)
    world = world_cls(cfg, spawn, rng, gui=gui)
    plutos = [pluto_cls(i, world, cfg) for i in range(n)]
    try:
        info = world.model_info()
        print("[model]", json.dumps({k: round(v, 9) for k, v in info.items()}))
    except Exception as ex:                                   # noqa
        info = {}
        print("[model] could not read model info:", ex)
    cam = OverheadCamera(cfg, n, rng)
    cam.faults = FaultInjector(cfg, faults, cam) if faults else None
    loc = Localizer(cfg, n, cam)
    ctrl = SwarmController(cfg, n)
    sup = SafetySupervisor(cfg, n)
    dash = None
    if live or show_cam:
        from .dashboard import LiveDashboard
        dash = LiveDashboard(cfg, n, show_plot=live, show_cam=show_cam)

    hz_c = cfg["sim"]["control_hz"]
    n_sub = int(round(cfg["sim"]["physics_hz"] / hz_c))
    dt_c = 1.0 / hz_c
    stage, plan, tau, t_arm = "init", None, 0.0, None
    rows = []
    wall0 = time.time()
    last_frame = None
    ground_samples = []

    while True:
        t = world.t
        # ---------------- sensing ----------------
        for t_cap, img in cam.deliver(t):
            last_frame = img
            for i, kind, pos in loc.process(img, t_cap):
                if t_arm is not None:
                    sup.log(t - t_arm, i, "OUTLIER_REJECTED" if kind == "outlier" else "REACQUIRED",
                            f"measured {np.round(pos, 2)}")
        P, V, Y, S, AGE = loc.estimate(t)

        # ---------------- planning (once, on the ground) ----------------
        if stage == "init" and t >= 0.3 and all(s == VALID for s in S):
            ground_samples.append(P.copy())          # average the noisy on-ground estimate
        if stage == "init" and t >= cfg["sim"]["init_time"]:
            if not all(s == VALID for s in S) or len(ground_samples) < 5:
                print("[init] waiting for all markers...", S)
            else:
                start = np.mean(ground_samples, axis=0)
                plan = plan_factory(start) if plan_factory else MissionPlan(cfg, start, formation)
                print(plan.summary() if hasattr(plan, "summary") else "")
                rep = validate(plan, cfg) if isinstance(plan, MissionPlan) else {"ok": True}
                with open(os.path.join(out_dir, "validator.json"), "w") as f:
                    json.dump({k: v for k, v in rep.items() if k != "limits"}, f, indent=1, default=str)
                if not rep["ok"]:
                    print("[init] plan REJECTED - not arming.")
                    break
                for i in range(n):
                    plutos[i].arm()
                    ctrl.reset(i)
                t_arm = t
                sup.arm_all(0.0, start[:, 2])
                stage = "mission"
        if stage == "init" and t > cfg["sim"]["init_time"] + 5:
            print("[init] markers never all seen - abort")
            break

        # ---------------- control ----------------
        if stage == "mission":
            tm = t - t_arm
            phase, _ = plan.phase_at(tau)
            directives, clock = sup.step(tm, phase, plan.fence_active(tau), P, S)
            tgt_p, tgt_v, tgt_a = plan.sample(tau)
            if not clock:
                tgt_v, tgt_a = np.zeros_like(tgt_v), np.zeros_like(tgt_a)
            gt_p, gt_q, gt_v, gt_rpy = world.ground_truth()
            for i in range(n):
                d = directives[i]
                cmd_p, mode = tgt_p[i].copy(), d[0]
                if mode == "off":
                    plutos[i].disarm()
                    rc = {"roll": 1500, "pitch": 1500, "yaw": 1500, "throttle": 1000}
                elif mode == "track":
                    rc, _ = ctrl.track(i, P[i], V[i], Y[i], tgt_p[i], tgt_v[i], tgt_a[i])
                elif mode == "freeze":
                    cmd_p = d[1]
                    rc, _ = ctrl.track(i, P[i], V[i], Y[i], cmd_p, np.zeros(3), np.zeros(3))
                elif mode == "land":
                    cmd_p = d[1]
                    rc, _ = ctrl.track(i, P[i], V[i], Y[i], cmd_p, d[2], np.zeros(3))
                else:  # trim (no usable vision)
                    cmd_p = np.full(3, np.nan)
                    rc, _ = ctrl.trim_only(i, Y[i], extra_az=d[1])
                if mode != "off":
                    plutos[i].set_rc(rc["roll"], rc["pitch"], rc["yaw"], rc["throttle"])
                rows.append({"t": tm, "tau": tau, "phase": phase, "clock": clock, "drone": i,
                             "state": sup.state[i], "mode": mode, "status": S[i], "age": AGE[i],
                             "tx": cmd_p[0], "ty": cmd_p[1], "tz": cmd_p[2],
                             "ex": P[i, 0], "ey": P[i, 1], "ez": P[i, 2], "eyaw": Y[i],
                             "x": gt_p[i, 0], "y": gt_p[i, 1], "z": gt_p[i, 2], "yaw": gt_rpy[i, 2],
                             "rc_roll": rc["roll"], "rc_pitch": rc["pitch"], "rc_yaw": rc["yaw"],
                             "rc_thr": rc["throttle"], "hover_thr": ctrl.hover_throttle(i),
                             "Ix": ctrl.I[i, 0], "Iy": ctrl.I[i, 1], "Iz": ctrl.I[i, 2],
                             "battery": world.battery_scale(), "slot": int(plan.slot_index[i]),
                             "phi": float(plan.phase[i])})
            if dash is not None and int(round(tm / dt_c)) % cfg["logging"]["live_every"] == 0:
                dash.update(tm, phase, gt_p, P, tgt_p, S, sup.state, last_frame)
            if clock:
                tau += dt_c
            done = all(s in (LANDED, IDLE) for s in sup.state)
            if done or tau > plan.T_total + 3.0 or tm > plan.T_total + 40:
                for pl in plutos:
                    pl.disarm()
                sup.log(tm, -1, "END", "all drones down" if done else "timeout")
                break

        # ---------------- physics ----------------
        for _ in range(n_sub):
            world.step()
            if cam.due(world.t):
                gp, gq, _, _ = world.ground_truth()
                cam.capture(world.t, gp, gq, None if t_arm is None else world.t - t_arm)

    print(f"[run] sim {world.t:.1f}s in {time.time() - wall0:.1f}s wall; camera "
          f"{cam.stats['captured']} frames, {cam.stats['dropped']} dropped")
    world.close()
    if dash is not None:
        dash.close()
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "telemetry.csv"), index=False)
    pd.DataFrame(sup.events).to_csv(os.path.join(out_dir, "events.csv"), index=False)
    with open(os.path.join(out_dir, "model_info.json"), "w") as f:
        json.dump(info, f, indent=1)
    if not isinstance(plan, MissionPlan):          # e.g. step_test.py makes its own plots
        return df
    from .metrics import compute_metrics, make_plots
    m = compute_metrics(df, cfg, plan)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(m, f, indent=1)
    make_plots(df, cfg, plan, out_dir, sup.events)
    print("[metrics]", json.dumps(m, indent=1))
    return m
