"""Run the full mission (A + C + D + E + F) in PyBullet.

  python run_mission.py --formation line                 # N=4 line wave
  python run_mission.py --formation circle --gui --live  # with 3-D view + live dashboard
  python run_mission.py --formation line --fault F1      # drill F1 (marker hidden 0.4 s)
  python run_mission.py --formation line --fault F3      # drill F3 (1 m outlier frame)
Outputs -> logs/<run>/ : telemetry.csv, events.csv, metrics.json, *.png
"""
import argparse

from swarm.config import load_config
from swarm.runner import run

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="config.yaml")
ap.add_argument("--formation", choices=["line", "circle"], default=None)
ap.add_argument("--n", type=int, default=None)
ap.add_argument("--fault", default="", help="comma list, e.g. F1 or F3 or F1,F3")
ap.add_argument("--gui", action="store_true", help="PyBullet 3-D window")
ap.add_argument("--live", action="store_true", help="live matplotlib dashboard")
ap.add_argument("--cam", action="store_true", help="show the synthetic camera feed")
ap.add_argument("--R", type=float, default=None)
ap.add_argument("--d", type=float, default=None)
ap.add_argument("--seed", type=int, default=None)
a = ap.parse_args()

cfg = load_config(a.config, {"mission.circle_radius": a.R, "mission.line_spacing": a.d, "seed": a.seed})
formation = a.formation or cfg["mission"]["formation"]
n = a.n or cfg["mission"]["n_drones"]
faults = [f.strip().upper() for f in a.fault.split(",") if f.strip()]
run(cfg, n, formation, faults=faults, gui=a.gui or cfg["sim"]["gui"], live=a.live, show_cam=a.cam)
