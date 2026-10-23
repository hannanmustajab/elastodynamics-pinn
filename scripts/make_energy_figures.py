#!/usr/bin/env python3
"""
Energy difference against time for the checkpoints after 10,000 epochs.

    figures/energy_difference_vs_time.{pdf,png}    one panel per benchmark on the window [0, 1],
                                                   three networks per panel
    figures/energy_difference_longtime.{pdf,png}   six-layer medium on the window [0, 2]

Quantity, for each reference time step (compute_energy_difference_over_time in the run scripts):

    sum over the grid of (|u_PINN| - |u_ref|)^2  /  sum over the grid of |u_ref|^2

with |u| the magnitude of the displacement and a nine-point border of the difference set to zero.
It is the square of the relative L2 error of the magnitude at that time step. It contains no
velocity or strain term; "energy" is the name used in the run scripts.

Data: figures/data/error_vs_time.csv, written by scripts/diagnostics/score_soap_runs.py with the
DEVITO reference of the manuscript. (The values the run scripts saved in energy_difference_<T>s.pkl
use the finite-difference reference the runs were monitored with; they are kept in
figures/data/energy_difference_fd_reference.csv and differ from these by less than 1%.)
A run that has not finished has no rows and no curve.

Axes are linear. The three media differ by two orders of magnitude, so each panel has its own
y axis; tick labels are plain decimals (no power-of-ten factor).

Style: scienceplots, through scripts/figure_style.py; network colours as in make_convergence_figures.py.

Usage:  python scripts/make_energy_figures.py
"""
import csv
from typing import Dict, List, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt

from figure_style import TEXTWIDTH_IN, apply_style
from make_convergence_figures import BENCHMARKS, LEGEND, NETWORKS, ROOT, STYLE

DATA = ROOT / "figures" / "data" / "error_vs_time.csv"
# Six-layer medium, window [0, 2]: network name in the data file -> legend label and line style.
# Colours differ from those of the three networks in make_convergence_figures.STYLE except for PirateNet 2 x 64.
LONG_RUNS = {
    "PirateNet 2x64": (r"PirateNet $2\times64$", dict(color="#D55E00", linestyle="-")),
    "PirateNet 3x64": (r"PirateNet $3\times64$", dict(color="#009E73", linestyle="--")),
    "PirateNet 2x128": (r"PirateNet $2\times128$", dict(color="#000000", linestyle=":")),
    "PirateNet 3x128": (r"PirateNet $3\times128$", dict(color="#56B4E9", linestyle="-.")),
}
Curves = Dict[Tuple[str, str], List[Tuple[float, float]]]


def load() -> Curves:
    curves = {}  # type: Curves
    with DATA.open() as handle:
        for row in csv.DictReader(handle):
            curves.setdefault((row["benchmark"], row["network"]), []).append((float(row["time"]), float(row["energy_difference"])))
    return curves


def finish(ax: plt.Axes, tmax: float, title: str) -> None:
    """Linear axes from zero; plain decimal tick labels, at most five per axis."""
    ax.set_xlim(0, tmax)
    ax.set_ylim(bottom=0)
    ax.set_xlabel(r"Time $t$")
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=5))
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda value, _: f"{value:.10f}".rstrip("0").rstrip(".") if value else "0"))
    ax.set_title(title)


def plot_short(curves: Curves) -> None:
    """One row of three panels, window [0, 1], each with its own y axis."""
    fig, axes = plt.subplots(1, 3, figsize=(TEXTWIDTH_IN, 1.9))
    for ax, benchmark, letter in zip(axes, BENCHMARKS, "abc"):
        for network in NETWORKS:
            points = curves.get((benchmark, network), [])
            if points:
                ax.plot(*zip(*points), lw=1.0, color=STYLE[network]["color"], linestyle=STYLE[network]["linestyle"])
        finish(ax, 1.0, f"({letter}) {benchmark}")
    axes[0].set_ylabel("Energy Difference")
    handles = [mpl.lines.Line2D([], [], lw=1.0, color=STYLE[n]["color"], linestyle=STYLE[n]["linestyle"]) for n in NETWORKS]
    fig.legend(handles, [LEGEND[n] for n in NETWORKS], loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.08), handlelength=2.6, columnspacing=1.6)
    fig.subplots_adjust(wspace=0.62)
    save(fig, "energy_difference_vs_time")


def plot_long(curves: Curves) -> None:
    """Six-layer medium on the window [0, 2]."""
    fig, ax = plt.subplots(figsize=(0.5 * TEXTWIDTH_IN, 1.9))
    for network, (label, style) in LONG_RUNS.items():
        points = curves.get(("Six-layer [0, 2]", network), [])
        if points:
            ax.plot(*zip(*points), lw=1.0, label=label, **style)
    finish(ax, 2.0, r"Six-layer, $t_{\max}=2$")
    ax.set_ylabel("Energy Difference")
    ax.legend(frameon=False, loc="upper left")
    save(fig, "energy_difference_longtime")


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(ROOT / "figures" / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(ROOT / "figures" / f"{name}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote figures/{name}.pdf and .png (600 dpi)")


def main() -> None:
    apply_style()
    curves = load()
    plot_short(curves)
    plot_long(curves)


if __name__ == "__main__":
    main()
