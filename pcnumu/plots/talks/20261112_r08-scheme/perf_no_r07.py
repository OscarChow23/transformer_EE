"""Replot performance12.py's bias/resolution-vs-trueE overlay without r07, from its cached per-bin
stats (cache/performance_<horn>_s<k>.json; bootstrap intervals unchanged). Same styling as performance12.py.
usage: python -I perf_no_r07.py OUT_DIR"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "/data/nova/chow/talk-plots_20261106/update_20261111/scripts")
import talk12 as t  # noqa: E402  (sets the rootstyle rc and the r05-r09 MODELS styles)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

out = Path(sys.argv[1])
RUNS = ["r05", "r06", "r08", "r09"]
bins = np.array(t.rs.PERF_BINS)
cent = (bins[:-1] + bins[1:]) / 2
for h in ("fhc", "rhc"):
    for s in (1, 2, 3):
        stats = json.loads((t.ROOT / "cache" / f"performance_{h}_s{s}.json").read_text())
        fig, (a, b) = plt.subplots(2, 1, figsize=(7.5, 6.8), sharex=True, gridspec_kw={"hspace": 0})
        offsets = dict(zip(RUNS, np.linspace(-.03, .03, len(RUNS))))
        for ax, key, ylabel in [(a, "bias", "Median fractional\nresidual (%)"), (b, "res", "Resolution (%)")]:
            for r in RUNS:
                ok = np.array([z["drawn"] for z in stats])
                v = np.array([z.get(r, {}).get(key, 0) for z in stats]) * 100
                lo = np.array([z.get(r, {}).get(key + "_lo", 0) for z in stats]) * 100
                hi = np.array([z.get(r, {}).get(key + "_hi", 0) for z in stats]) * 100
                st = t.rs.MODELS[r]
                ax.errorbar(cent[ok] + offsets[r], v[ok],
                            yerr=np.vstack([np.maximum(0, v[ok] - lo[ok]), np.maximum(0, hi[ok] - v[ok])]),
                            fmt=st["marker"], color=st["color"], mfc=st["mfc"], mec=st["color"], ms=4,
                            elinewidth=1, label=r)
            t.rs.labels(ax, None, ylabel)
            ax.set_xlim(.5, 6)
        a.axhline(0, color="black", ls=":", lw=1)
        t.rs.tag(a, f"{h.upper()} · S{s} · test", loc="upper right")
        a.legend(loc="upper center", ncol=4, fontsize=10, bbox_to_anchor=(.5, 1.16), columnspacing=1.4)
        b.set_ylim(bottom=0)
        t.rs.labels(b, t.rs.E_TRUE, None)
        fig.align_ylabels([a, b])
        t.rs.save(fig, str(out), f"r05-r06-r08-r09-{h}-s{s}-bias-resolution",
                  "Same as update_20261111 r05-r09-s{s}-bias-resolution (performance12.py) with r07 removed; "
                  "replotted from its cached per-bin stats cache/performance_{h}_s{s}.json, unchanged. "
                  "Top: median of r = pred/true - 1; bottom: (q84 - q16)/2, percent, vs true neutrino energy; "
                  "0.25 GeV bins over 0.5-6 GeV, N<50 hidden; errors: central 68% of 2000 bootstrap resamples per bin "
                  "(statistical only, no training-seed variation). Unweighted test events, identical across models."
                  .replace("{s}", str(s)).replace("{h}", h))
        print("wrote", h, s)
