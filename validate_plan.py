"""Task B demo: python validate_plan.py --n 10 --formation both
Builds the full mission plan (no sim needed) and runs the pre-flight validator."""
import argparse
import numpy as np

from swarm.config import load_config
from swarm.planner import MissionPlan, spawn_positions
from swarm.validator import validate

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="config.yaml")
ap.add_argument("--n", type=int, default=None)
ap.add_argument("--formation", default="both", choices=["line", "circle", "both"])
ap.add_argument("--d", type=float, default=None, help="override line spacing")
ap.add_argument("--R", type=float, default=None, help="override circle radius")
ap.add_argument("--try-fix", action="store_true", help="re-run circle with a suggested R")
a = ap.parse_args()

cfg = load_config(a.config, {"mission.line_spacing": a.d, "mission.circle_radius": a.R})
n = a.n or cfg["mission"]["n_drones"]
rng = np.random.default_rng(cfg["seed"])
start = spawn_positions(cfg, n, rng)
forms = ["line", "circle"] if a.formation == "both" else [a.formation]
for f in forms:
    plan = MissionPlan(cfg, start, f)
    rep = validate(plan, cfg)
    if f == "circle" and a.try_fix and not rep["ok"]:
        lim = rep["limits"]
        if lim["feasible"]:
            R = 0.5 * (lim["R_min"] + lim["R_max"])
            print(f"   -> retrying circle with suggested R = {R:.3f}")
            cfg2 = load_config(a.config, {"mission.circle_radius": R})
            validate(MissionPlan(cfg2, start, f), cfg2)
    print()
