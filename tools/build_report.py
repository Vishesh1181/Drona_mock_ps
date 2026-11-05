"""Fill docs/REPORT.md with the real metrics from logs/ and render HTML.
    python tools/build_report.py --name "Your Name" --roll 12345
Then open docs/REPORT.html in a browser -> Print -> Save as PDF (check <= 4 pages)."""
import argparse, glob, json, os, re, shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--name", default="NAME")
ap.add_argument("--roll", default="ROLL NO.")
a = ap.parse_args()


def f(x, k=1, unit=""):
    return "–" if x is None or x != x else f"{x * k:.1f}{unit}"


rows = ["| Run | Track RMS (cm) ≤10 | Shape RMS (cm) | Phase sync (°) ≤10 | Min sep (m) ≥0.35 | Loc RMS (cm) | x/y/z loc (mm) |",
        "|---|---|---|---|---|---|---|"]
for path in sorted(glob.glob(os.path.join(ROOT, "logs", "*", "metrics.json"))):
    m = json.load(open(path))
    run = os.path.basename(os.path.dirname(path))
    la = m["localization_rms_axis_m"]
    rows.append(f"| {run} | {f(m['tracking_rms_m'], 100)} | {f(m['formation_shape_rms_m'], 100)} | "
                f"{f(m['phase_sync_error_deg'])} | {m['min_separation_m']:.3f} | {f(m['localization_rms_m'], 100)} | "
                f"{la['x']*1000:.0f}/{la['y']*1000:.0f}/{la['z']*1000:.0f} |")
table = "\n".join(rows) if len(rows) > 2 else "_Run the missions first (no logs/*/metrics.json found)._"
ev = []
for path in sorted(glob.glob(os.path.join(ROOT, "logs", "*F*", "events.csv"))):
    ev.append(f"Events ({os.path.basename(os.path.dirname(path))}): see `{os.path.relpath(path, ROOT)}`.")
table += "\n\n" + " ".join(ev)

step_txt = "_Run `python step_test.py` and paste its printed line here._"
src = os.path.join(ROOT, "logs", "step_test", "step_response.png")
if os.path.exists(src):
    shutil.copy(src, os.path.join(ROOT, "docs", "step_response.png"))
    step_txt = "Predicted vs simulated x-step (0.3 m): see figure; numbers printed by `step_test.py`."

md = open(os.path.join(ROOT, "docs", "REPORT.md")).read()
md = md.replace("{{METRICS_TABLE}}", table).replace("{{STEP_RESULT}}", step_txt)
md = md.replace("{{NAME}}", a.name).replace("{{ROLL}}", a.roll)
open(os.path.join(ROOT, "docs", "REPORT_filled.md"), "w").write(md)
try:
    import markdown
    body = markdown.markdown(md, extensions=["tables"])
    css = ("@page{size:A4;margin:9mm}"
           "body{font-family:Arial,sans-serif;font-size:9.2pt;line-height:1.25;margin:0}.cols{column-count:2;column-gap:6mm}"
           "table{border-collapse:collapse;font-size:7.9pt;width:100%;break-inside:avoid}"
           "td,th{border:1px solid #999;padding:1px 3px;vertical-align:top}"
           "img{max-width:100%;break-inside:avoid}p,ul{margin:2px 0}li{margin:0}ul{padding-left:14px}"
           "h1{font-size:12pt;margin:0 0 3px;column-span:all}h2{font-size:9.8pt;margin:6px 0 2px;break-after:avoid}"
           "code{font-size:8.3pt}")
    open(os.path.join(ROOT, "docs", "REPORT.html"), "w").write(
        f"<html><head><meta charset='utf-8'><style>{css}</style></head><body><div class=cols>{body}</div></body></html>")
    print("wrote docs/REPORT_filled.md and docs/REPORT.html -> open in browser, Print -> Save as PDF")
except ImportError:
    print("wrote docs/REPORT_filled.md (pip install markdown for the HTML version)")
