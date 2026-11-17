"""r08 proposal (10 Nov, not yet agreed): per-sample flattening, before vs after, per sample and combined, both horns.
Proposal: S1 alpha 0.05, S2 and S3 alpha 0.02; flat from 1.0 GeV (S1 RHC 0.9) to 5 GeV, easing off to 8 GeV;
each sample's weights average 1 (natural sample mix). r05 (pooled alpha 0.05) shown dashed for reference.
Bottom row: toy lean (median estimate/true - 1) vs true energy, same toy as alpha_tradeoff.py."""
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
ns = {}; exec(open("alpha_tradeoff.py").read().split("for horn in")[0], ns)
SIG = {"fhc": {1: .0675, 2: .1013, 3: .1518}, "rhc": {1: .0612, 2: .0920, 3: .1724}}
LO = {"fhc": {1: 1.0, 2: 1.0, 3: 1.0}, "rhc": {1: 0.9, 2: 1.0, 3: 1.0}}
ALPHA = {1: 0.05, 2: 0.02, 3: 0.02}
C_BEFORE, C_AFTER, C_R05 = "#8a8a85", "#eb6834", "#2a78d6"
b = np.arange(0, 10.0001, 0.1)

def lean_curve(y, w, sig, edges):
    h, _ = np.histogram(y, ns["bins"], weights=w); xc = ns["xc"]
    grid = np.arange(0.3, 12, 0.01); est = np.empty_like(grid)
    for i, m in enumerate(grid):
        post = h * np.exp(-0.5 * ((xc - m) / (sig * m)) ** 2); c = np.cumsum(post); est[i] = xc[np.searchsorted(c, c[-1] / 2)]
    t = ns["rng"].choice(y, 300000); m = t * (1 + sig * ns["rng"].standard_normal(t.size)); r = np.interp(m, grid, est) / t - 1
    return np.array([100 * np.median(r[(t >= lo) & (t < lo + 0.25)]) for lo in edges])

for horn in ("fhc", "rhc"):
    d = np.load(f"/data/nova/chow/pcnumu-runs/r06_flat-alpha001/flattening_scan/edge_and_tail/train_{horn}.npz"); y_all, s = d["y"], d["s"]
    w05 = ns["FlatSpectraWeights"](y_all, bins=500, alpha=0.05).getweight(y_all)
    w_new = np.ones_like(y_all)
    for smp in (1, 2, 3):
        k = s == smp; w_new[k] = ns["persample"](y_all[k], LO[horn][smp], 5.0, 3.0, ALPHA[smp])
    fig, ax = plt.subplots(3, 4, figsize=(18, 11), sharex=True)
    edges = np.arange(1.0, 5.0, 0.25); lines = []
    for j, smp in enumerate((1, 2, 3, "all")):
        k = np.ones_like(s, bool) if smp == "all" else s == smp; y = y_all[k]
        for w, col, lab, ls, lw in ((np.ones_like(y), C_BEFORE, "before (no reweighting)", "-", 2),
                                    (w05[k] * k.sum() / w05[k].sum(), C_R05, "r05 (pooled α 0.05), for reference", "--", 1.2),
                                    (w_new[k], C_AFTER, "after: r08 proposal", "-", 2)):
            h, _ = np.histogram(y, b, weights=w); ax[0, j].stairs(h, b, color=col, ls=ls, lw=lw, label=lab)
            if smp != "all":
                pk = w[(y > 1.8) & (y < 2.2)].mean(); idx = np.clip(np.digitize(y, b) - 1, 0, len(b) - 2)
                n = np.bincount(idx, minlength=len(b) - 1); sw = np.bincount(idx, w, len(b) - 1); ok = n > 20
                ax[1, j].plot(0.5 * (b[1:] + b[:-1])[ok], sw[ok] / n[ok] / pk, color=col, ls=ls, lw=lw)
                ax[2, j].plot(edges + 0.125, lean_curve(y, w, SIG[horn][smp], edges), color=col, ls=ls, lw=lw, marker="o", ms=3)
        if smp == "all":
            for smp2, ls2 in ((1, ":"), (2, "-."), (3, (0, (5, 1)))):
                h, _ = np.histogram(y_all[s == smp2], b, weights=w_new[s == smp2]); ax[0, j].stairs(h, b, color=C_AFTER, ls=ls2, lw=1, label=f"after, S{smp2} part")
            for r in (1, 2):
                ax[r, j].axis("off")
            ax[1, j].text(0.02, 0.9, "Proposal\n\nS1: α 0.05\nS2: α 0.02\nS3: α 0.02\n\nflat from 1.0 GeV"
                          + (" (S1 0.9)" if horn == "rhc" else "") + "\nto 5 GeV, easing off\nby 8 GeV\n\neach sample keeps its\nshare of the total",
                          transform=ax[1, j].transAxes, va="top", fontsize=11)
            ax[0, j].set_title(f"{horn.upper()} all samples combined")
        else:
            wv = w_new[k]; ess = wv.sum() ** 2 / (wv ** 2).sum() / wv.size
            ax[0, j].set_title(f"{horn.upper()} sample {smp}   ({k.sum():,} events)\nafter: max weight {wv.max()/wv[(y>1.8)&(y<2.2)].mean():.0f}× peak event, "
                               f"{100*ess:.0f}% effectively used", fontsize=10)
            ax[1, j].set_yscale("log"); ax[1, j].axhline(1, color="k", lw=0.5)
            ax[2, j].axhline(0, color="k", lw=0.5); ax[2, j].axhspan(-3, 3, color="#1baf7a", alpha=0.1); ax[2, j].set_ylim(-15, 20)
            ax[2, j].set_xlabel("true neutrino energy (GeV)")
        ax[0, j].set_yscale("log"); ax[0, j].set_ylim(10, None)
        for r in range(3): ax[r, j].grid(alpha=0.2); ax[r, j].set_xlim(0, 10)
    ax[0, 0].set_ylabel("training events per 0.1 GeV\n(after: weighted)"); ax[1, 0].set_ylabel("how much one event counts\n(÷ an event at 2 GeV)")
    ax[2, 0].set_ylabel("lean of the guess (%)\n(simple model; + = guessed too high)")
    ax[0, 0].legend(frameon=False, fontsize=8, loc="lower left"); ax[0, 3].legend(frameon=False, fontsize=8, loc="lower left")
    fig.suptitle(f"{horn.upper()} r08 proposal: before vs after reweighting (training rows; preview, no training)", fontsize=13)
    fig.tight_layout(); fig.savefig(f"proposal_{horn}.png", dpi=105)
print("done")
