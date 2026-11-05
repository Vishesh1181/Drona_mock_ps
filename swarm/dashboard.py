"""F. Live view: true (filled) vs estimated (x) vs target (hollow) + trails,
top view and side view, per-drone status panel. Optional camera window
showing what the vision system actually sees."""
from collections import deque

import cv2
import matplotlib.pyplot as plt
import numpy as np

COLORS = {"valid": "green", "stale": "orange", "lost": "red"}


class LiveDashboard:
    def __init__(self, cfg, n, show_plot=True, show_cam=False):
        self.n, self.show_plot, self.show_cam = n, show_plot, show_cam
        self.trails = [deque(maxlen=150) for _ in range(n)]
        if not show_plot:
            return
        plt.ion()
        self.fig, (self.top, self.side, self.txt) = plt.subplots(
            1, 3, figsize=(15, 5), gridspec_kw={"width_ratios": [2, 2, 1]})
        f = cfg["world"]["geofence"]
        self.top.add_patch(plt.Rectangle((f["x"][0], f["y"][0]), 2.0, 1.2, fill=False, ec="r", ls="--"))
        self.side.add_patch(plt.Rectangle((f["x"][0], f["z"][0]), 2.0, 0.8, fill=False, ec="r", ls="--"))
        self.top.set(xlim=(-1.2, 1.2), ylim=(-0.8, 0.8), aspect="equal", title="top view (x-y)")
        self.side.set(xlim=(-1.2, 1.2), ylim=(0, 1.6), title="side view (x-z)")
        c = plt.cm.tab10(np.arange(10))
        self.h = []
        for i in range(n):
            col = c[i % 10]
            hs = {}
            for ax, key in ((self.top, "t"), (self.side, "s")):
                hs[key + "true"], = ax.plot([], [], "o", color=col, ms=9)
                hs[key + "est"], = ax.plot([], [], "x", color="k", ms=8)
                hs[key + "tgt"], = ax.plot([], [], "o", mfc="none", mec=col, ms=14)
                hs[key + "trail"], = ax.plot([], [], "-", color=col, lw=1, alpha=.6)
            self.h.append(hs)
        self.top.plot([], [], "o", color="gray", label="true")
        self.top.plot([], [], "x", color="k", label="estimate")
        self.top.plot([], [], "o", mfc="none", mec="gray", label="target")
        self.top.legend(loc="upper right", fontsize=8)
        self.txt.axis("off")
        self.text = self.txt.text(0, 1, "", va="top", family="monospace", fontsize=10)

    def update(self, t, phase, P_true, P_est, P_tgt, status, states, frame):
        if self.show_plot:
            for i in range(self.n):
                self.trails[i].append(P_true[i].copy())
                tr = np.array(self.trails[i])
                hs = self.h[i]
                for key, a, b in (("t", 0, 1), ("s", 0, 2)):
                    hs[key + "true"].set_data([P_true[i, a]], [P_true[i, b]])
                    hs[key + "est"].set_data([P_est[i, a]], [P_est[i, b]])
                    hs[key + "tgt"].set_data([P_tgt[i, a]], [P_tgt[i, b]])
                    hs[key + "trail"].set_data(tr[:, a], tr[:, b])
                    hs[key + "est"].set_color(COLORS[status[i]])
            lines = [f"t = {t:6.2f} s", f"phase: {phase}", ""]
            lines += [f"drone {i}: {status[i]:5s} {states[i]:7s}\n   err {np.linalg.norm(P_true[i]-P_tgt[i])*100:5.1f} cm"
                      for i in range(self.n)]
            self.text.set_text("\n".join(lines))
            self.fig.canvas.draw_idle()
            plt.pause(0.001)
        if self.show_cam and frame is not None:
            small = cv2.resize(frame, (640, 360))
            cv2.imshow("overhead camera (what the controller sees)", small)
            cv2.waitKey(1)

    def close(self):
        if self.show_cam:
            cv2.destroyAllWindows()
        if self.show_plot:
            plt.ioff()
            plt.close(self.fig)
