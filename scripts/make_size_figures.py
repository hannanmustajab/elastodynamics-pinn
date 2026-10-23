#!/usr/bin/env python3
"""
Error of the PirateNet size variants in the six-layer medium during training.

    figures/error_vs_epoch_sizes.{pdf,png}   (a) t_max = 1: 2 x 64, 2 x 128, 3 x 128
                                             (b) t_max = 2: 2 x 64, 3 x 64, 2 x 128, 3 x 128

Quantity: trajectory-averaged relative L2 error of the displacement field in percent, for the
checkpoint of every 500th epoch (figures/data/error_vs_epoch.csv, written by
scripts/diagnostics/score_soap_runs.py with the DEVITO reference). The untrained network (epoch 0)
is left out. A run that was interrupted is drawn as far as it has checkpoints.

The horizontal axis is the epoch, as in every figure of the manuscript (author, 2026-10-12). The runs
differ in the number of batches per epoch; that number is in the settings table of the manuscript.

Style: scienceplots through scripts/figure_style.py; line styles as in make_energy_figures.LONG_RUNS.

Usage:  python scripts/make_size_figures.py
"""
import csv
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt

from figure_style import TEXTWIDTH_IN, apply_style
from make_convergence_figures import ROOT
from make_energy_figures import save

DATA = ROOT / "figures" / "data" / "error_vs_epoch.csv"
# Panel title -> run folder (without the suffix _FINISHED) -> legend label and line style.
PANELS = {
    r"(a) $t_{\max}=1$": {
        "SixLayer_Pirate_2x64_T1.0_soap_rad": (r"PirateNet $2\times64$", dict(color="#D55E00", linestyle="-", marker="^")),
        "SixLayer_Pirate_2x128_T1.0_soap_rad": (r"PirateNet $2\times128$", dict(color="#000000", linestyle=":", marker="o")),
        "SixLayer_Pirate_3x128_T1.0_soap_rad": (r"PirateNet $3\times128$", dict(color="#56B4E9", linestyle="-.", marker="s")),
    },
    r"(b) $t_{\max}=2$": {
        "SixLayer_Pirate_2x64_T2.0_scale1_soap_rad": (r"PirateNet $2\times64$", dict(color="#D55E00", linestyle="-", marker="^")),
        "SixLayer_Pirate_3x64_T2.0_soap_rad": (r"PirateNet $3\times64$", dict(color="#009E73", linestyle="--", marker="v")),
        "SixLayer_Pirate_2x128_T2.0_soap_rad": (r"PirateNet $2\times128$", dict(color="#000000", linestyle=":", marker="o")),
        "SixLayer_Pirate_3x128_T2.0_soap_rad": (r"PirateNet $3\times128$", dict(color="#56B4E9", linestyle="-.", marker="s")),
    },
}
Curve = List[Tuple[int, float]]


def load() -> Dict[str, Curve]:
    """Run -> (epoch, error in percent) pairs sorted by epoch, without epoch 0.

    Raises:
        ValueError: if a run of the figure has no rows in the data file.
    """
    curves = {}  # type: Dict[str, Curve]
    with DATA.open() as handle:
        for row in csv.DictReader(handle):
            if int(row["epoch"]) > 0:
                curves.setdefault(row["run"], []).append((int(row["epoch"]), float(row["error_percent"])))
    missing = sorted(run for runs in PANELS.values() for run in runs if run not in curves)
    if missing:
        raise ValueError(f"no rows in {DATA.name} for {missing}")
    return {run: sorted(points) for run, points in curves.items()}


def main() -> None:
    """Draw the figure from the data file."""
    apply_style()
    curves = load()
    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH_IN, 2.3), sharey=True)
    for ax, (title, runs) in zip(axes, PANELS.items()):
        for run, (label, style) in runs.items():
            ax.plot(*zip(*curves[run]), lw=1.0, ms=2.4, markevery=2, label=label, **style)
        ax.set_yscale("log")
        ax.set_xlim(0, 10000)
        ax.set_xticks([0, 5000, 10000])
        ax.set_xlabel("Epoch")
        ax.set_title(title)
        ax.legend(frameon=False, loc="upper right", handlelength=2.2)
    axes[0].set_ylabel("Time-averaged relative\n" r"$L_2$ error (%)")
    fig.subplots_adjust(wspace=0.22)   # room for the tick labels 10000 and 0 of neighbouring panels
    save(fig, "error_vs_epoch_sizes")


if __name__ == "__main__":
    main()
