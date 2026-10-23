#!/usr/bin/env python3
"""
Relative L2 error against time for the checkpoints after 10,000 epochs.

    figures/error_vs_time.{pdf,png}   (a) two-layer and (b) six-layer medium, t_max = 1: Vanilla, Fourier-Feature
                                      and the PirateNet with the lowest error; (c) six-layer medium, t_max = 2:
                                      the PirateNet sizes. One figure since 2026-10-12 (author: combine the figure
                                      of t_max = 2 with this one); the separate figure error_vs_time_longtime is
                                      no longer drawn (the old files are in archive/figures_before_fig12_merge_2026-10-12/).

Quantity, for each reference time step: the error metric of the manuscript,

    || u_ref(t) - u_PINN(t) ||_2 / || u_ref(t) ||_2   in percent,

with u the displacement vector on the 512 x 512 evaluation grid. Its mean over the time steps is
the trajectory-averaged error of the tables.

Panels of the first figure (author, 2026-10-12): two-layer and six-layer only, and of the PirateNet
sizes only the one with the lowest error in that medium (2 x 64 in the two-layer medium, the only size
run there; 3 x 128 in the six-layer medium).

Data: figures/data/error_vs_time.csv, written by scripts/diagnostics/score_soap_runs.py with the
DEVITO reference of the manuscript. Every curve of the figure must have rows there; the script
stops if one is missing.

Axes are linear and start at zero; each panel has its own y axis. Style: scripts/figure_style.py;
network colours as in make_convergence_figures.py.

Usage:  python scripts/make_error_time_figures.py
"""
import csv
from typing import Dict, List, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt

from figure_style import TEXTWIDTH_IN, apply_style
from make_convergence_figures import PANEL_LEGEND, ROOT, STYLE
from make_energy_figures import LONG_RUNS, save

DATA = ROOT / "figures" / "data" / "error_vs_time.csv"
YLABEL = r"Relative $L_2$ error (%)"
LONG_BENCHMARK = "Six-layer [0, 2]"
# Panel -> curves in drawing order: (network name in the data file, legend label, style key).
SHORT_PANELS = {
    "Two-layer": [("Vanilla", PANEL_LEGEND["Vanilla"], "Vanilla"), ("Fourier-feature", PANEL_LEGEND["Fourier-feature"], "Fourier-feature"),
                  ("PirateNet", PANEL_LEGEND["PirateNet"], "PirateNet")],
    "Six-layer": [("Vanilla", PANEL_LEGEND["Vanilla"], "Vanilla"), ("Fourier-feature", PANEL_LEGEND["Fourier-feature"], "Fourier-feature"),
                  ("PirateNet 3x128", PANEL_LEGEND["PirateNet"], "PirateNet")],
}
Curves = Dict[Tuple[str, str], List[Tuple[float, float]]]


def load() -> Curves:
    """(benchmark, network) -> (time, error in percent) pairs, sorted by time.

    Raises:
        ValueError: if the rows of one (benchmark, network) pair come from more than one run, or if a
            curve of the figure has no rows.
    """
    curves = {}  # type: Curves
    runs = {}  # type: Dict[Tuple[str, str], set]
    with DATA.open() as handle:
        for row in csv.DictReader(handle):
            key = (row["benchmark"], row["network"])
            curves.setdefault(key, []).append((float(row["time"]), float(row["error_percent"])))
            runs.setdefault(key, set()).add(row["run"])
    doubled = sorted(key for key, names in runs.items() if len(names) > 1)
    if doubled:
        raise ValueError(f"more than one run in {DATA.name} for {doubled}")
    expected = {(b, network) for b, panel in SHORT_PANELS.items() for network, _, _ in panel} | {(LONG_BENCHMARK, n) for n in LONG_RUNS}
    missing = sorted(expected - set(curves))
    if missing:
        raise ValueError(f"no rows in {DATA.name} for {missing}")
    return {key: sorted(points) for key, points in curves.items()}


def finish(ax: plt.Axes, tmax: float, title: str) -> None:
    """Linear axes from zero, at most five y ticks."""
    ax.set_xlim(0, tmax)
    ax.set_ylim(bottom=0)
    ax.set_xlabel(r"Time $t$")
    ax.set_ylabel(YLABEL)
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=5))
    ax.set_title(title)


def plot_all(curves: Curves) -> None:
    """Three panels with their own y axes: two-layer and six-layer medium for t_max = 1, then the six-layer medium
    for t_max = 2. Each panel has its legend below the axes, where it covers no curve."""
    fig, axes = plt.subplots(1, 3, figsize=(TEXTWIDTH_IN, 2.3))
    for ax, (benchmark, panel), letter in zip(axes, SHORT_PANELS.items(), "ab"):
        for network, label, key in panel:
            ax.plot(*zip(*curves[(benchmark, network)]), lw=1.0, label=label, color=STYLE[key]["color"], linestyle=STYLE[key]["linestyle"])
        finish(ax, 1.0, rf"({letter}) {benchmark}, $t_{{\max}}=1$")
        ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.21), handlelength=2.0)
    ax = axes[2]
    for network, (label, style) in LONG_RUNS.items():
        ax.plot(*zip(*curves[(LONG_BENCHMARK, network)]), lw=1.0, label=label.replace("PirateNet ", ""), **style)
    # Dashed gray lines at 5 % and 10 %, the levels that the text gives crossing times for (as the lines at 1 % and 5 % of Fig. 5).
    for level in (5.0, 10.0):
        ax.axhline(level, color="0.55", lw=0.5, ls="--", zorder=0)
    finish(ax, 2.0, r"(c) Six-layer, $t_{\max}=2$")
    # The legend title names the network, so the entries are the sizes only.
    ax.legend(title="PirateNet", frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.21), handlelength=2.0, ncol=2,
              columnspacing=1.0)
    fig.subplots_adjust(wspace=0.38)
    save(fig, "error_vs_time")


def main() -> None:
    """Draw the figure from the data file."""
    apply_style()
    plot_all(load())


if __name__ == "__main__":
    main()
