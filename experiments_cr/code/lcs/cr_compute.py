"""GPU-hours from the scheduler log (finished jobs only) -> tables/cr_compute.tex (a bare number for \input)."""
import datetime, os, re
H = os.path.expanduser("~/lcg-cr"); st = {}; tot = 0.0
for l in open(f"{H}/logs/sched.log"):
    m = re.match(r"(\d\d-\d\d \d\d:\d\d:\d\d) (START|DONE|FAILED) (\S+).*gpus=\[([^\]]*)\]", l)
    if not m: continue
    t = datetime.datetime.strptime("2026-" + m.group(1), "%Y-%m-%d %H:%M:%S"); n = len(m.group(4).split(","))
    if m.group(2) == "START": st[m.group(3)] = (t, n)
    elif m.group(3) in st: t0, n = st.pop(m.group(3)); tot += (t - t0).total_seconds() / 3600 * n
open(f"{H}/tables/cr_compute.tex", "w").write(f"{round(tot / 10) * 10:,.0f}%\n"); print(round(tot, 1), "running", len(st))
