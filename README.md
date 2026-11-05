# SwarmSim-Lite — Drona Mock PS

N Crazyflie nano-quads (Pluto stand-ins) fly a **line** and a **circle** doing a Mexican wave, using **only an overhead ArUco camera** for position and **only RC commands (1000–2000)** for control.

- **Simulator:** PyBullet via `gym-pybullet-drones` (main branch, v2.x), model `cf2x`, `Physics.PYB_GND_DRAG_DW` (ground effect + drag + downwash), 250 Hz physics, 25 Hz control.
- **ROS 2:** bonus track not attempted.
- **Tested on:** Ubuntu 24.04, Python 3.12.3, gym-pybullet-drones 2.2.0, PyBullet 3.2.7, OpenCV 5.0.0, NumPy 2.5.4, SciPy 1.18.1.
- **Downwash guard:** the upstream downwash force diverges as Δz → 0 and threw grounded drones away; `swarm/sim_world.py` applies the same model only for Δz ≥ `sim.dw_min_dz` (5 cm) with the upper drone's motors on.
- **External code used:** `gym-pybullet-drones` (physics, and the attitude part of `DSLPIDControl` inside `PlutoInterface`, which the PS explicitly allows). No swarm or position-control libraries. Everything in `swarm/` is ours.

## Install

```bash
sudo apt install -y python3-venv python3-pip python3-tk git build-essential
python3 -m venv .venv && source .venv/bin/activate
git clone https://github.com/utiasDSL/gym-pybullet-drones.git
cd gym-pybullet-drones && pip install --no-deps -e . && cd ..
pip install pybullet gymnasium transforms3d pillow   # torch / RL extras are not needed
pip install -r requirements.txt
```

## Run (from this folder)

| Task | Command |
|---|---|
| 0. Plumbing check (signs, hover) | `python sanity_check.py` |
| B. Validator, N = 4 (accepted) | `python validate_plan.py --n 4` |
| B. Validator, N = 10, both formations (rejected + fixes) | `python validate_plan.py --n 10 --try-fix` (saved output: `logs/validator_N10.txt`) |
| A+C+D+E+F. Line, N = 4 | `python run_mission.py --formation line` |
| Circle, N = 4 | `python run_mission.py --formation circle` |
| With 3-D view, live dashboard, camera feed (for the video) | `python run_mission.py --formation circle --gui --live --cam` |
| Fault drill F1 (drone 2 marker hidden 0.4 s) | `python run_mission.py --formation line --fault F1` |
| Fault drill F3 (one frame: drone 1 jumped 1 m) | `python run_mission.py --formation line --fault F3` |
| Math 5: predicted vs simulated step response | `python step_test.py` |
| Vision calibration / accuracy | `python tools/calibrate_vision.py` |
| Fill the report and print the PDF | `python tools/build_report.py --name "Vishesh Singh" --roll 2502MT08` then `google-chrome --headless=new --no-pdf-header-footer --print-to-pdf=docs/REPORT.pdf "file://$PWD/docs/REPORT.html"` |

Every run writes to `logs/<run>/`:

- `telemetry.csv` holds true, estimated and target positions, RC, status, mode and integrators per drone per tick.
- `events.csv` is the safety log, with timestamps.
- `metrics.json` holds the auto-generated metrics.
- `summary.png` and `z_tracking.png` are the plots.
- `validator.json` and `model_info.json` record the validator result and the parameters read from the URDF.

All parameters are in `config.yaml`. Things like N, R and d can be overridden from the command line (`--n`, `--R`, `--d`).

## Layout

```
swarm/planner.py          A  slots, Hungarian assignment, min-jerk, wave, mission timeline
swarm/validator.py        B  pre-flight sampling at 0.02 s + analytic suggestions
swarm/camera.py              sensor model: pinhole + warpPerspective, noise, drops, latency
swarm/localization.py     C  ArUco -> pose, outlier gate, alpha-beta filter, status
swarm/controller.py       D  cascaded P -> PI, anti-windup, accel -> RC
swarm/safety.py           E  stale/lost/separation/geofence rules + event log
swarm/metrics.py          F  metrics + plots      swarm/dashboard.py  F  live view
swarm/pluto_interface.py     the only door to a drone (RC in, firmware attitude loop)
swarm/sim_world.py           PyBullet plant + drift push + battery sag
swarm/faults.py              F1 / F3 injection at the camera
swarm/runner.py              main loop
```

## Rules we respect

- **Controller inputs:** the controller only sees camera estimates. Ground truth is read only by the camera renderer and by `metrics.py`.
- **Simulator access:** the planner and controller never call PyBullet. Only `PlutoInterface` (firmware) and `SimWorld` touch it.
- **Unknown disturbances:** the controller is not told about drift or battery sag, or about the hover throttle; it starts from a guess of 1500 and learns everything by integration.
