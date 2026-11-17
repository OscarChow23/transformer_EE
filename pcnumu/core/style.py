"""House plot style: every pc-numu figure (diagnostics, studies, talks, thesis) uses this module.

    from pcnumu.core import style
    style.use()                                    # once, before making figures
    fig, ax = style.figure()
    ax.plot(x, y, **style.SAMPLE_STYLE["3"])       # fixed colours/markers: never ad-hoc ones
    style.labels(ax, style.E_TRUE, "Resolution [%]")
    style.tag(ax, "FHC · S3 · test set")
    style.save(fig, out_dir, "name", caption="what is plotted, which events, the one claim")

Look (ROOT/thesis-like): Helvetica-like font (TeX Gyre Heros, bundled), ticks inward on all four
sides with minor ticks, no grid, thin black frame, axis titles flush to the end of each axis,
frameless legends. No figure titles: context goes in a short in-frame tag and the caption file.
"""
import os

import matplotlib
from matplotlib import font_manager

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import AutoMinorLocator  # noqa: E402

FONT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")
FONT = "TeX Gyre Heros"

# ---- fixed vocabulary: use these everywhere, never ad-hoc colours or names ----
INK, MUTED = "#000000", "#6b6b6b"
SERIES = [  # runs or models compared in one figure, in order
    dict(color="#000000", marker="o", ls="-"), dict(color="#0b5fbf", marker="s", ls="--"),
    dict(color="#c81e1e", marker="^", ls="-."), dict(color="#2e7d32", marker="v", ls=":"),
    dict(color="#7b3294", marker="D", ls=(0, (5, 1)))]
BARS = ["#0b5fbf", "#c81e1e", "#2e7d32"]  # grouped/stacked bars: first, second, third series
SAMPLE_STYLE = {  # sample -> line style; the long label is the legend text
    "1": dict(color="#2f5d9e", marker="o", ls="-", label="S1: contained"),
    "2": dict(color="#b8741a", marker="s", ls="-", label=r"S2: peripheral, $\mu$ contained"),
    "3": dict(color="#2e7d32", marker="^", ls="-", label=r"S3: peripheral, $\mu$ uncontained"),
    "all": dict(color="#000000", marker="D", ls="--", label="All samples"),
}
SAMPLE_FILL = {"1": "#9db7de", "2": "#f2c27a", "3": "#a7d3a0"}
HORN = {"fhc": "FHC", "rhc": "RHC"}
E_TRUE = r"True neutrino energy $E_\nu^{\rm true}$ (GeV)"
E_RECO = r"Reconstructed energy $E_\nu^{\rm reco}$ (GeV)"
FRAC = r"$E_\nu^{\rm reco}/E_\nu^{\rm true} - 1$"


def line(k):
    """Line/marker keyword arguments for sample k, without the label."""
    return {key: v for key, v in SAMPLE_STYLE[k].items() if key != "label"}


def use():
    if os.path.isdir(FONT_DIR):
        for f in os.listdir(FONT_DIR):
            if f.endswith(".otf"):
                font_manager.fontManager.addfont(os.path.join(FONT_DIR, f))
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": [FONT, "DejaVu Sans"], "font.size": 13,
        "mathtext.fontset": "custom", "mathtext.rm": FONT, "mathtext.it": f"{FONT}:italic",
        "mathtext.bf": f"{FONT}:bold", "mathtext.sf": FONT, "mathtext.cal": FONT, "mathtext.default": "it",
        "axes.linewidth": 1.0, "axes.edgecolor": "black", "axes.labelsize": 15, "axes.grid": False,
        "axes.prop_cycle": matplotlib.cycler(color=[s["color"] for s in SERIES]),
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
        "savefig.facecolor": "white", "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def figure(ncols=1, nrows=1, size=(6.4, 4.8), **kw):
    """Figure with ROOT-like proportions; size is per panel."""
    return plt.subplots(nrows, ncols, figsize=(size[0] * ncols, size[1] * nrows), **kw)


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


def header(ax, left, right=None):
    """Small text just above the frame: what the plot shows (left) and horn/sample (right)."""
    ax.text(0, 1.015, left, transform=ax.transAxes, ha="left", va="bottom", fontsize=12)
    if right:
        ax.text(1, 1.015, right, transform=ax.transAxes, ha="right", va="bottom", fontsize=11, color=MUTED)


def save(fig, out_dir, name, caption=None, formats=("png", "pdf")):
    """Write name.<fmt> for each format, and name.txt when a caption is given; closes the figure."""
    os.makedirs(out_dir, exist_ok=True)
    for ext in formats:
        fig.savefig(os.path.join(out_dir, f"{name}.{ext}"))
    if caption:
        with open(os.path.join(out_dir, f"{name}.txt"), "w") as f:
            f.write(caption.strip() + "\n")
    plt.close(fig)
