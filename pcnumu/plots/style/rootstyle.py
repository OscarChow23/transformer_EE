"""ROOT/thesis-like matplotlib style shared by every plot in talk-plots_20261106.

    import sys; sys.path.insert(0, "<talk-plots>/style"); import rootstyle as rs
    rs.use()                       # once, before making figures
    fig, ax = rs.figure()          # single panel, 1 column of a slide / thesis page
    ...
    rs.labels(ax, "True neutrino energy (GeV)", "Events / 0.2 GeV")
    rs.tag(ax, "FHC  ·  S3 (peripheral, muon uncontained)")
    rs.save(fig, out_dir, "p5_s3_bias_vs_trueE", caption_text)

Look: Helvetica-like font (TeX Gyre Heros, bundled), ticks inward on all four sides with
minor ticks, no grid, thin black frame, axis titles flush to the right / top end of the axes,
frameless legends, points with error bars, step histograms. No titles on the figure: what the
plot is goes in the .txt caption written by save().
"""
import os

import matplotlib
from matplotlib import font_manager

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import AutoMinorLocator  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FONT = "TeX Gyre Heros"

# ---- fixed vocabulary: use these everywhere, never ad-hoc colours or names ----------------------
MODELS = {  # model -> style; same in every figure and both horns
    "r05": dict(color="#000000", marker="o", mfc="#000000", ls="-", label=r"r05 ($\alpha$ = 0.05)"),
    "r06": dict(color="#0b5fbf", marker="s", mfc="none", ls="--", label=r"r06 ($\alpha$ = 0.01)"),
    "r07": dict(color="#c81e1e", marker="^", mfc="#c81e1e", ls="-.", label=r"r07 ($\alpha$ = 0.01, ½ S1)"),
}
SAMPLES = {  # sample -> (short, long, fill colour, line colour)
    1: dict(short="S1", long="S1: contained", fill="#9db7de", line="#2f5d9e"),
    2: dict(short="S2", long="S2: peripheral, $\\mu$ contained", fill="#f2c27a", line="#b8741a"),
    3: dict(short="S3", long="S3: peripheral, $\\mu$ uncontained", fill="#a7d3a0", line="#2e7d32"),
}
HORN = {"fhc": "FHC", "rhc": "RHC"}
WINDOW = (0.5, 6.0)  # GeV, analysis window on true neutrino energy
PERF_BINS = [0.5 + 0.25 * i for i in range(23)]  # 0.5-6.0 GeV, 0.25 GeV: bias/resolution vs trueE
MIN_PER_BIN = 50  # bins with fewer test events are not drawn (same rule as eval_by_sample.py)
N_BOOT, BOOT_SEED = 2000, 1
SPEC_BINS = [0.2 * i for i in range(51)]  # 0-10 GeV, 0.2 GeV: training spectra
E_TRUE = r"True neutrino energy $E_\nu^{\rm true}$ (GeV)"
E_RECO = r"Reconstructed energy $E_\nu^{\rm reco}$ (GeV)"
FRAC = r"$E_\nu^{\rm reco}/E_\nu^{\rm true} - 1$"


def use():
    for f in os.listdir(os.path.join(HERE, "fonts")):
        if f.endswith(".otf"):
            font_manager.fontManager.addfont(os.path.join(HERE, "fonts", f))
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": [FONT, "DejaVu Sans"], "font.size": 13,
        "mathtext.fontset": "custom", "mathtext.rm": FONT, "mathtext.it": f"{FONT}:italic",
        "mathtext.bf": f"{FONT}:bold", "mathtext.sf": FONT, "mathtext.default": "it",
        "axes.linewidth": 1.0, "axes.edgecolor": "black", "axes.labelsize": 15, "axes.grid": False,
        "axes.prop_cycle": matplotlib.cycler(color=["#000000", "#0b5fbf", "#c81e1e", "#2e7d32"]),
        "axes.unicode_minus": True, "axes.xmargin": 0, "axes.formatter.use_mathtext": True,
        "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
        "xtick.minor.visible": True, "ytick.minor.visible": True, "xtick.labelsize": 13,
        "ytick.labelsize": 13, "xtick.major.size": 8, "ytick.major.size": 8, "xtick.minor.size": 4,
        "ytick.minor.size": 4, "xtick.major.width": 1.0, "ytick.major.width": 1.0,
        "xtick.major.pad": 6, "ytick.major.pad": 5,
        "legend.frameon": False, "legend.fontsize": 12, "legend.handlelength": 1.6,
        "lines.linewidth": 1.6, "lines.markersize": 6, "errorbar.capsize": 0,
        "image.cmap": "viridis",
        "figure.dpi": 100, "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def figure(ncols=1, nrows=1, size=(6.4, 4.8), **kw):
    """Figure with ROOT-like proportions. size is per panel."""
    fig, ax = plt.subplots(nrows, ncols, figsize=(size[0] * ncols, size[1] * nrows), **kw)
    return fig, ax


def labels(ax, xlabel=None, ylabel=None):
    """Axis titles flush with the end of each axis, ROOT/HEP convention."""
    if xlabel is not None:
        ax.set_xlabel(xlabel, loc="right")
    if ylabel is not None:
        ax.set_ylabel(ylabel, loc="top")
    for a in (ax.xaxis, ax.yaxis):
        if a.get_scale() == "linear":
            a.set_minor_locator(AutoMinorLocator())


def tag(ax, text, loc="upper left", **kw):
    """Small in-frame label (horn, sample, test set). Keep it to one or two short lines."""
    x, ha = (0.04, "left") if "left" in loc else (0.96, "right")
    y, va = (0.95, "top") if "upper" in loc else (0.05, "bottom")
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va, fontsize=12, **kw)


def model_points(ax, x, y, yerr, model, xerr=None, offset=0.0, **kw):
    """Points with error bars in the fixed model style (small x offset avoids overlap)."""
    s = MODELS[model]
    return ax.errorbar([v + offset for v in x], y, yerr=yerr, xerr=xerr, fmt=s["marker"], color=s["color"],
                       mfc=s["mfc"], mec=s["color"], elinewidth=1.1, label=s["label"], **kw)


def save(fig, out_dir, name, caption):
    """Write name.pdf (vector, for the thesis), name.png (300 dpi, for slides) and name.txt."""
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"{name}.{ext}"))
    with open(os.path.join(out_dir, f"{name}.txt"), "w") as f:
        f.write(caption.strip() + "\n")
    plt.close(fig)
