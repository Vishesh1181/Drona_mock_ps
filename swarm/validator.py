"""B. Pre-flight validator.

Samples the whole plan every dt (0.02 s) BEFORE arming and checks:
  1. pairwise separation >= 0.35 m at every sample,
  2. geofence during the fenced phases (transit .. hold2),
  3. (extra) the plan's accelerations are flyable at 93% battery.
If anything fails it rejects the plan, says why and suggests a fix.
"""
import numpy as np

from .planner import MissionPlan, formation_slots


def analytic_limits(cfg, formation, n):
    """Closed-form feasibility (see report, Math 3).
    Wave z-offsets between two drones pass through zero every cycle, so the
    minimum 3-D separation equals the minimum HORIZONTAL slot separation."""
    m, w = cfg["mission"], cfg["world"]
    fx, fy, fz = w["geofence"]["x"], w["geofence"]["y"], w["geofence"]["z"]
    dmin = w["min_separation"]
    out = {"A_max": min(m["height"] - fz[0], fz[1] - m["height"])}
    if formation == "line":
        span = fx[1] - fx[0]
        out["d_min"] = dmin
        out["d_max"] = span / (n - 1) if n > 1 else np.inf
        out["N_max"] = int(np.floor(span / dmin + 1e-9)) + 1
        out["feasible"] = out["d_min"] <= out["d_max"]
    else:
        th = 2 * np.pi * np.arange(n) / n
        out["R_min"] = dmin / (2 * np.sin(np.pi / n)) if n > 1 else 0.0
        cmax, smax = np.max(np.abs(np.cos(th))), np.max(np.abs(np.sin(th)))
        out["R_max"] = min(fx[1] / cmax, fy[1] / smax if smax > 1e-9 else np.inf)
        nmax = 1
        for k in range(2, 60):
            t = 2 * np.pi * np.arange(k) / k
            rmin = dmin / (2 * np.sin(np.pi / k))
            rmax = min(fx[1] / np.max(np.abs(np.cos(t))), fy[1] / np.max(np.abs(np.sin(t))))
            if rmin <= rmax:
                nmax = k
        out["N_max"] = nmax
        out["feasible"] = out["R_min"] <= out["R_max"]
    return out


def validate(plan: MissionPlan, cfg, verbose=True):
    w, v = cfg["world"], cfg["validator"]
    fence = w["geofence"]
    lo = np.array([fence["x"][0], fence["y"][0], fence["z"][0]])
    hi = np.array([fence["x"][1], fence["y"][1], fence["z"][1]])
    dmin = w["min_separation"]
    n = plan.n
    iu = np.triu_indices(n, 1)

    reasons, suggestions = [], []
    worst = {"sep": np.inf, "t": None, "pair": None, "phase": None}
    worst_form = dict(worst)          # only hold/wave/hold2 (the formation itself)
    fence_hit = None
    max_axy, max_az_up, max_az_dn = 0.0, 0.0, 0.0

    for tau in np.arange(0.0, plan.T_total + 1e-9, v["dt"]):
        pos, vel, acc = plan.sample(tau)
        phase = plan.phase_at(tau)[0]
        if n > 1:
            d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=2)[iu]
            k = int(np.argmin(d))
            rec = {"sep": float(d[k]), "t": tau, "pair": (int(iu[0][k]), int(iu[1][k])), "phase": phase}
            if d[k] < worst["sep"]:
                worst = rec
            if phase in ("hold", "wave", "hold2") and d[k] < worst_form["sep"]:
                worst_form = rec
        if plan.fence_active(tau) and fence_hit is None:
            bad = np.where(np.any((pos < lo) | (pos > hi), axis=1))[0]
            if len(bad):
                fence_hit = (tau, int(bad[0]), phase, pos[bad[0]].round(3))
        max_axy = max(max_axy, float(np.max(np.linalg.norm(acc[:, :2], axis=1))))
        max_az_up = max(max_az_up, float(np.max(acc[:, 2])))
        max_az_dn = max(max_az_dn, float(-np.min(acc[:, 2])))

    lim = analytic_limits(cfg, plan.formation, n)
    m = cfg["mission"]

    if worst_form["sep"] < dmin:
        i, j = worst_form["pair"]
        reasons.append(f"formation separation {worst_form['sep']:.3f} m < {dmin} m (drones {i},{j}, "
                       f"{worst_form['phase']})")
        if plan.formation == "line":
            suggestions.append(f"min d = {lim['d_min']:.3f} m (now {m['line_spacing']})")
        else:
            suggestions.append(f"min R = {lim['R_min']:.3f} m (now {m['circle_radius']})")
    if worst["sep"] < dmin and worst["phase"] not in ("hold", "wave", "hold2"):
        i, j = worst["pair"]
        reasons.append(f"separation {worst['sep']:.3f} m < {dmin} m between drones {i} and {j} "
                       f"at t={worst['t']:.2f}s ({worst['phase']})")
        if True:
            suggestions.append(f"transit/takeoff conflict: spawn spacing >= sqrt(2)*{dmin} = "
                               f"{np.sqrt(2) * dmin:.3f} m guarantees safe straight transits, "
                               f"or stagger transit altitudes")
    if fence_hit:
        t, i, ph, p = fence_hit
        reasons.append(f"geofence: drone {i} at {p} at t={t:.2f}s ({ph})")
        if plan.formation == "line":
            suggestions.append(f"max d = {lim['d_max']:.3f} m for N={n}")
        else:
            suggestions.append(f"max R = {lim['R_max']:.3f} m for N={n}")
        if m["wave_amplitude"] > lim["A_max"]:
            suggestions.append(f"max wave amplitude A = {lim['A_max']:.2f} m")
    if not lim["feasible"]:
        what = "d" if plan.formation == "line" else "R"
        rng_txt = (f"needs d>={lim['d_min']:.3f} but fence allows d<={lim['d_max']:.3f}"
                   if what == "d" else
                   f"needs R>={lim['R_min']:.3f} but fence allows R<={lim['R_max']:.3f}")
        reasons.append(f"{plan.formation} with N={n} is infeasible: {rng_txt}")
        suggestions.append(f"max N = {lim['N_max']} for a {plan.formation}")
        if plan.formation == "line":
            suggestions.append("or use a circle (N=10 fits with R in [0.567, 0.631] m)")
    elif plan.formation == "circle" and not (lim["R_min"] <= m["circle_radius"] <= lim["R_max"]):
        suggestions.append(f"choose R in [{lim['R_min']:.3f}, {lim['R_max']:.3f}] m")

    # --- extra: physical flyability at 93% battery (methodology 4, simplified) ---
    g = 9.81
    b = cfg["validator"]["battery_for_check"]
    tw = cfg["drone_model"]["thrust_to_weight"]
    margin = cfg["validator"]["accel_margin"]
    a_xy_lim = margin * g * np.tan(np.radians(cfg["pluto"]["max_tilt_deg"]))
    a_up_lim = margin * (b * tw - 1.0) * g
    a_dn_lim = margin * g
    if max_axy > a_xy_lim:
        reasons.append(f"horizontal accel {max_axy:.2f} > {a_xy_lim:.2f} m/s^2 (tilt limit)")
        suggestions.append("increase transit time / lower transit_vmax")
    if max_az_up > a_up_lim or max_az_dn > a_dn_lim:
        reasons.append(f"vertical accel up {max_az_up:.2f}/{a_up_lim:.2f}, down {max_az_dn:.2f}/{a_dn_lim:.2f}")
        suggestions.append("lower wave amplitude or omega")

    ok = len(reasons) == 0
    report = {"ok": ok, "reasons": reasons, "suggestions": suggestions,
              "min_separation": worst["sep"], "worst": worst, "worst_formation": worst_form, "limits": lim,
              "max_acc": {"xy": max_axy, "z_up": max_az_up, "z_down": max_az_dn}}
    if verbose:
        print_report(plan, report)
    return report


def print_report(plan, r):
    tag = "ACCEPTED" if r["ok"] else "REJECTED"
    print(f"[validator] {plan.formation} N={plan.n}: {tag}  "
          f"(min planned separation {r['min_separation']:.3f} m)")
    for x in r["reasons"]:
        print(f"   reason    : {x}")
    for x in r["suggestions"]:
        print(f"   suggestion: {x}")
