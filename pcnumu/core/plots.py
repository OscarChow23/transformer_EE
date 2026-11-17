"""Plots shared by several commands, drawn in the house style (pcnumu.core.style).

  relerr_hist             relative-error distributions, one step curve per series
  bias_resolution_panels  bias and resolution vs true energy, two stacked panels, one curve per series
A series is (label, style kwargs, values...). Single-purpose plots live with their command.
"""
import numpy as np

from pcnumu.core import style
from pcnumu.core.metrics import MIN_PER_BIN, binned_bias_resolution

RELERR_EDGES = np.linspace(-1, 1, 101)


def relerr_hist(series, xlabel, tag=None):
    """series: [(label, style kwargs, r)]. Each curve normalised to unit area."""
    fig, ax = style.figure()
    for label, st, r in series:
        ax.hist(r, bins=RELERR_EDGES, density=True, histtype="step", lw=1.8, color=st["color"],
                ls=st["ls"], label=label)
    ax.axvline(0, color=style.MUTED, lw=1, ls=":")
    style.labels(ax, xlabel, "Unit-normalised density")
    if tag:
        style.tag(ax, tag)
    ax.legend(loc="upper right", fontsize=10)
    return fig


def bias_resolution_panels(series, edges, xlabel, centers=None, logx=False, tag=None):
    """series: [(label, style kwargs, x, r)]; bins with < MIN_PER_BIN events are not drawn."""
    centers = 0.5 * (edges[1:] + edges[:-1]) if centers is None else centers
    fig, (ax1, ax2) = style.figure(nrows=2, size=(6.4, 3.4), sharex=True, gridspec_kw=dict(hspace=0))
    for label, st, x, r in series:
        bias, res = binned_bias_resolution(x, r, edges)
        kw = dict(color=st["color"], marker=st["marker"], ls=st["ls"], lw=1.6, ms=5, label=label)
        ax1.plot(centers, bias * 100, **kw)
        ax2.plot(centers, res * 100, **kw)
    ax1.axhline(0, color=style.MUTED, lw=1, ls=":")
    if logx:
        for ax in (ax1, ax2):
            ax.set_xscale("log")
    style.labels(ax1, None, "Bias [%]")
    style.labels(ax2, xlabel, "Resolution [%]")
    if tag:
        style.tag(ax1, tag + f"\nbins with ≥ {MIN_PER_BIN} events")
    ax1.legend(loc="upper right", fontsize=10)
    return fig, (ax1, ax2)
