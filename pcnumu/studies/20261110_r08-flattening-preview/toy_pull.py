"""Toy (not the model): how a training spectrum's shape pulls an uncertain estimate.
Training spectrum = prior; event's inputs = Gaussian likelihood with relative width sigma;
the estimate = posterior median (MAPE behaves roughly like a median). Same two plateaus
(1.0 below 2 GeV, 0.05 above 5 GeV), joined by a cliff at 3 GeV or by a gradual ramp 2-5 GeV."""
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
E = np.linspace(0.2, 10, 4000)
lo, hi = np.log(1.0), np.log(0.05)
cliff = np.exp(np.where(E < 3, lo, hi) )
cliff = np.exp(lo + (hi - lo) / (1 + np.exp(-(E - 3) / 0.05)))   # drop over ~0.2 GeV
ramp = np.exp(np.interp(E, [2, 5], [lo, hi]))                     # same drop over 3 GeV
priors = {"cliff at 3 GeV": (cliff, "#eb6834"), "gradual ramp 2–5 GeV": (ramp, "#2a78d6")}
m = np.linspace(1.0, 6.0, 101)    # what the event's inputs say (GeV)
def shift(prior, sig):
    out = []
    for x in m:
        post = prior * np.exp(-0.5 * ((E - x) / (sig * x)) ** 2)
        c = np.cumsum(post); med = E[np.searchsorted(c, c[-1] / 2)]
        out.append(100 * (med - x) / x)
    return np.array(out)
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
for name, (p, col) in priors.items():
    ax[0].plot(E, p, color=col, lw=2, label=name)
    for sig, ls in ((0.07, ":"), (0.17, "-")):
        s = shift(p, sig)
        ax[1].plot(m, s, color=col, lw=2, ls=ls, label=f"{name}, σ {int(sig*100)}%")
        print(f"{name:22s} σ{int(sig*100):>3}%  worst shift {s.min():6.1f}% at {m[s.argmin()]:.1f} GeV;  at 2.5/4 GeV: {s[np.searchsorted(m,2.5)]:.1f}% / {s[np.searchsorted(m,4)]:.1f}%")
ax[0].set_yscale("log"); ax[0].set_xlim(0, 8); ax[0].set_xlabel("true energy (GeV)"); ax[0].set_ylabel("training events (relative)")
ax[0].set_title("two training spectra, same plateaus"); ax[0].legend(frameon=False)
ax[1].axhline(0, color="#8a8a85", lw=1); ax[1].set_xlabel("energy the event's inputs point to (GeV)")
ax[1].set_ylabel("estimate shift (%)"); ax[1].set_title("pull on the estimate (toy)"); ax[1].legend(frameon=False, fontsize=8)
for a in ax: a.grid(alpha=0.25, lw=0.5)
fig.tight_layout(); fig.savefig("toy_pull.png", dpi=120)
