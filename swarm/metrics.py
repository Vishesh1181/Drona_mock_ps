"""F. Metrics (auto-generated after each run). Ground truth is used HERE only.

  tracking RMS      ||p_true - p_target|| over hold/wave/hold2           (target <= 10 cm)
  formation shape   RMS after removing the formation centroid (shape only, not position)
  phase sync        fit z_i(t) - h = a sin(wt) + b cos(wt) + c on the full-amplitude
                    part of the wave -> phi_i = atan2(-b, a); error vs commanded phase;
                    'sync' = worst pairwise error of relative phases       (target <= 10 deg)
  min separation    min pairwise TRUE distance among airborne drones
  localisation RMS  ||p_est - p_true|| while vision is valid
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

FORM = ("hold", "wave", "hold2")


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def _rms(x):
    x = np.asarray(x)
    return float(np.sqrt(np.mean(x**2))) if len(x) else float("nan")


def compute_metrics(df, cfg, plan):
    m = {}
    trk = df[df.phase.isin(FORM) & (df["mode"] == "track") & df.clock]
    err = np.linalg.norm(trk[["x", "y", "z"]].values - trk[["tx", "ty", "tz"]].values, axis=1)
    m["tracking_rms_m"] = _rms(err)
    m["tracking_rms_per_drone_m"] = {int(i): _rms(err[trk.drone.values == i]) for i in sorted(trk.drone.unique())}
    m["tracking_max_m"] = float(err.max()) if len(err) else float("nan")
    ax = trk[["x", "y", "z"]].values - trk[["tx", "ty", "tz"]].values
    m["tracking_rms_axis_m"] = dict(zip("xyz", [_rms(ax[:, k]) for k in range(3)]))

    # formation shape error (centroid removed)
    shp = []
    for _, g in trk.groupby("t"):
        if len(g) < 2:
            continue
        a = g[["x", "y", "z"]].values
        b = g[["tx", "ty", "tz"]].values
        shp.append(np.linalg.norm((a - a.mean(0)) - (b - b.mean(0)), axis=1))
    m["formation_shape_rms_m"] = _rms(np.concatenate(shp)) if shp else float("nan")

    # phase sync
    w = plan.omega
    t0 = plan.t_start["wave"]
    Tw = plan.durations["wave"]
    wv = df[(df.phase == "wave") & (df["mode"] == "track") & df.clock]
    wv = wv[(wv.tau - t0 > plan.ramp) & (wv.tau - t0 < Tw - plan.ramp)]
    ph = {}
    for i, g in wv.groupby("drone"):
        if len(g) < 25:                                  # need > 1 s of data
            continue
        tw = g.tau.values - t0
        Aq = np.c_[np.sin(w * tw), np.cos(w * tw), np.ones_like(tw)]
        a, b, _ = np.linalg.lstsq(Aq, g.z.values, rcond=None)[0]
        ph[int(i)] = {"measured_deg": float(np.degrees(np.arctan2(-b, a))),
                      "commanded_deg": float(np.degrees(wrap(g.phi.values[0]))),
                      "amplitude_m": float(np.hypot(a, b))}
        ph[int(i)]["error_deg"] = float(np.degrees(wrap(np.radians(ph[int(i)]["measured_deg"]
                                                                   - ph[int(i)]["commanded_deg"]))))
    ids = sorted(ph)
    pair = [abs(ph[i]["error_deg"] - ph[j]["error_deg"]) for k, i in enumerate(ids) for j in ids[k + 1:]]
    m["phase"] = ph
    m["phase_sync_error_deg"] = float(max(pair)) if pair else float("nan")
    m["phase_lag_vs_command_deg"] = float(np.mean([abs(ph[i]["error_deg"]) for i in ids])) if ids else float("nan")

    # minimum separation among airborne drones (true positions)
    best = (np.inf, None, None)
    for t, g in df[df.z > 0.3].groupby("t"):
        if len(g) < 2:
            continue
        p = g[["x", "y", "z"]].values
        d = np.linalg.norm(p[:, None] - p[None], axis=2)
        d[np.diag_indices(len(p))] = np.inf
        k = np.unravel_index(np.argmin(d), d.shape)
        if d[k] < best[0]:
            best = (float(d[k]), float(t), (int(g.drone.values[k[0]]), int(g.drone.values[k[1]])))
    m["min_separation_m"], m["min_separation_t"], m["min_separation_pair"] = best

    # localisation
    lv = df[(df.status == "valid") & (df["mode"] != "off")]
    le = lv[["ex", "ey", "ez"]].values - lv[["x", "y", "z"]].values
    m["localization_rms_m"] = _rms(np.linalg.norm(le, axis=1))
    m["localization_rms_axis_m"] = dict(zip("xyz", [_rms(le[:, k]) for k in range(3)]))
    m["status_fraction"] = df[df["mode"] != "off"].status.value_counts(normalize=True).round(4).to_dict()
    m["learned_hover_throttle_end"] = {int(i): round(float(g.hover_thr.values[-1]), 1)
                                       for i, g in df[df["mode"] == "track"].groupby("drone")}
    m["pass"] = {"tracking<=0.10m": m["tracking_rms_m"] <= 0.10,
                 "phase_sync<=10deg": bool(m["phase_sync_error_deg"] <= 10.0),
                 "min_sep>=0.35m": m["min_separation_m"] >= cfg["world"]["min_separation"]}
    return m


def make_plots(df, cfg, plan, out, events):
    n = plan.n
    cols = plt.cm.tab10(np.arange(10))
    # ---- 1. z: true vs estimate vs target ----
    fig, axs = plt.subplots(n if n <= 6 else 1, 1, figsize=(10, 1.9 * min(n, 6) + 1), sharex=True, squeeze=False)
    for i in range(n):
        ax = axs[min(i, len(axs) - 1), 0]
        g = df[df.drone == i]
        ax.plot(g.t, g.tz, "k--", lw=1, label="target")
        ax.plot(g.t, g.ez, color=cols[i % 10], alpha=.5, lw=1, label="estimate")
        ax.plot(g.t, g.z, color=cols[i % 10], lw=1.6, label="true")
        ax.set_ylabel(f"z{i} [m]")
    axs[0, 0].legend(ncol=3, fontsize=8, loc="upper right")
    axs[-1, 0].set_xlabel("time since arming [s]")
    fig.suptitle("Altitude / wave: true vs estimate vs target")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "z_tracking.png"), dpi=120)
    plt.close(fig)

    # ---- 2. summary panel ----
    fig, ax = plt.subplots(3, 2, figsize=(13, 11))
    f = cfg["world"]["geofence"]
    a = ax[0, 0]
    a.add_patch(plt.Rectangle((f["x"][0], f["y"][0]), f["x"][1] - f["x"][0], f["y"][1] - f["y"][0],
                              fill=False, ec="r", ls="--", label="geofence"))
    for i in range(n):
        g = df[df.drone == i]
        a.plot(g.x, g.y, color=cols[i % 10], lw=1.2, label=f"drone {i}")
        a.plot(g.tx, g.ty, "k:", lw=.8)
        a.plot(plan.slot[i, 0], plan.slot[i, 1], "o", mfc="none", mec="k")
    a.set_aspect("equal"), a.set_title("Top view: true trails (colour), targets (dotted)"), a.legend(fontsize=7)

    a = ax[0, 1]
    seps = []
    for t, g in df[df.z > 0.3].groupby("t"):
        if len(g) > 1:
            p = g[["x", "y", "z"]].values
            d = np.linalg.norm(p[:, None] - p[None], axis=2)
            d[np.diag_indices(len(p))] = np.inf
            seps.append((t, d.min()))
    if seps:
        s = np.array(seps)
        a.plot(s[:, 0], s[:, 1])
    a.axhline(cfg["world"]["min_separation"], color="r", ls="--", label="0.35 m limit")
    a.set_title("Minimum pairwise separation (true)"), a.set_ylabel("m"), a.legend()

    a = ax[1, 0]
    for i in range(n):
        g = df[(df.drone == i) & (df["mode"] == "track")]
        e = np.linalg.norm(g[["x", "y", "z"]].values - g[["tx", "ty", "tz"]].values, axis=1)
        a.plot(g.t, e, color=cols[i % 10], lw=1)
    a.axhline(0.10, color="r", ls="--")
    a.set_title("Tracking error |true - target| (track mode)"), a.set_ylabel("m"), a.set_ylim(0, 0.3)

    a = ax[1, 1]
    lv = df[(df.status == "valid") & (df["mode"] != "off")]
    for k, c in zip("xyz", ["C0", "C1", "C2"]):
        a.plot(lv.t, lv["e" + k] - lv[k], ".", ms=1.5, color=c, label=f"{k} error")
    a.set_title("Localisation error (estimate - true)"), a.set_ylabel("m"), a.legend(markerscale=6)
    a.set_ylim(-0.08, 0.08)

    a = ax[2, 0]
    for i in range(n):
        g = df[(df.drone == i) & (df["mode"] != "off")]
        a.plot(g.t, g.hover_thr, color=cols[i % 10], lw=1)
    a2 = a.twinx()
    a2.plot(df.t, df.battery * 100, "k:", lw=1)
    a2.set_ylabel("battery max-thrust [%] (dotted)")
    a.set_title("Learned hover throttle (integrator) - it finds hover & follows sag")
    a.set_ylabel("throttle")

    a = ax[2, 1]
    code = {"valid": 0, "stale": 1, "lost": 2}
    for i in range(n):
        g = df[df.drone == i]
        a.plot(g.t, g.status.map(code) + i * 3, drawstyle="steps-post", color=cols[i % 10])
    for e in events:
        if e["event"] not in ("ARM", "END"):
            a.axvline(e["t"], color="gray", lw=.6, ls=":")
    a.set_yticks([i * 3 + k for i in range(n) for k in range(3)])
    a.set_yticklabels([f"{i}:{s}" for i in range(n) for s in ("valid", "stale", "lost")], fontsize=6)
    a.set_title("Vision status per drone (dotted lines = safety events)")
    for r in ax.flat:
        r.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "summary.png"), dpi=110)
    plt.close(fig)
