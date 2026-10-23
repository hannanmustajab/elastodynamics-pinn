#!/usr/bin/env python3
"""
Convergence figure for the three benchmarks: two rows of three panels, three networks per panel.

    figures/convergence_error_loss.{pdf,png}

    (a)-(c)  relative L2 error against epoch      columns: homogeneous, two-layer, six-layer (t_max = 1)
    (d)-(f)  training loss against epoch

Dashed grey lines in (a)-(c) mark errors of 1% and 5%.

The two quantities were separate figures until 2026-10-12; the author asked for one figure to save space.

Networks: Vanilla 5 x 128, Fourier-feature 5 x 128 and PirateNet. PirateNet is 2 x 64 in the first two
columns and, in the six-layer column, the size with the lowest error in that medium, 3 x 128 (author,
2026-10-12: "add result for the best piratenet model", for the error and for the loss). The legend
therefore names the networks without their sizes; the caption of the manuscript gives them.

Error: the metric of Staub's thesis (Eq. 3.11), the relative L2 error of the displacement field
averaged over all reference time steps of the training window, computed for the checkpoint of
every 500th epoch by scripts/diagnostics/score_soap_runs.py (figures/data/error_vs_epoch.csv).
Its value at epoch 10,000 is the number in the tables of the manuscript.

Loss: the training loss the scripts print every 50 epochs, the mean over the batches of an epoch
of [mean squared PDE residual + 100 x mean squared absorbing-boundary residual]
(figures/data/loss_vs_epoch.csv, rebuilt from the training logs in <benchmark>/results/ with --fetch).
The collocation points are redrawn every 1,000 epochs with more points where the residual is
large, so the loss steps up at these epochs; dotted lines mark them.

Runs that have not started have no rows and no curve; a run in progress has a curve that ends
at its last value.

Style: scienceplots, through scripts/figure_style.py.

Usage:  python scripts/make_convergence_figures.py [--fetch]
"""
import csv
import string
import sys
from pathlib import Path
from typing import Dict, List, NamedTuple, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt

from figure_style import TEXTWIDTH_IN, apply_style

ROOT = Path(__file__).resolve().parent.parent
ERROR_DATA = ROOT / "figures" / "data" / "error_vs_epoch.csv"
LOSS_DATA = ROOT / "figures" / "data" / "loss_vs_epoch.csv"
EPOCHS, RESAMPLING_INTERVAL = 10000, 1000

# Benchmark folders of this repository (runs.csv lists every run).
HOMOGENEOUS, TWO_LAYER, SIX_LAYER = "homogeneous", "two_layer", "six_layer"
BENCHMARKS = ["Homogeneous", "Two-layer", "Six-layer"]
NETWORKS = ["Vanilla", "Fourier-feature", "PirateNet"]
# (benchmark, network) -> (benchmark folder, run). SOAP with residual-based resampling, 10,000 epochs.
RUNS = {
    ("Homogeneous", "Vanilla"): (HOMOGENEOUS, "Homogeneous_Vanilla_5x128_soap_rad"),
    ("Homogeneous", "Fourier-feature"): (HOMOGENEOUS, "Homogeneous_FourierXB_5x128_T1.0_soap_rad"),
    ("Homogeneous", "PirateNet"): (HOMOGENEOUS, "Homogeneous_Pirate_2x64_soap_rad"),
    ("Two-layer", "Vanilla"): (TWO_LAYER, "TwoLayer_Vanilla_5x128_T1.0_soap_rad"),
    ("Two-layer", "Fourier-feature"): (TWO_LAYER, "TwoLayer_FourierXB_5x128_T1.0_soap_rad"),
    ("Two-layer", "PirateNet"): (TWO_LAYER, "TwoLayer_Pirate_2x64_T1.0_soap_rad"),
    # Two batches per epoch, as the other two six-layer runs (an earlier run with one batch is not in the paper).
    ("Six-layer", "Vanilla"): (SIX_LAYER, "SixLayer_Vanilla_5x128_T1.0_2batch_soap_rad"),
    ("Six-layer", "Fourier-feature"): (SIX_LAYER, "SixLayer_FourierXB_5x128_T1.0_soap_rad"),
    # The PirateNet with the lowest error in this medium (0.34%); 2 x 64 has 3.34% and is drawn in make_size_figures.py.
    ("Six-layer", "PirateNet"): (SIX_LAYER, "SixLayer_Pirate_3x128_T1.0_soap_rad"),
}
# Legend labels with sizes: used by the figures that import them (error against time, energy difference).
LEGEND = {"Vanilla": r"Vanilla $5\times128$", "Fourier-feature": r"Fourier-Feature $5\times128$", "PirateNet": r"PirateNet $2\times64$"}
# Legend labels of the two figures of this script: no sizes, because the PirateNet of panel (c) differs.
PANEL_LEGEND = {"Vanilla": "Vanilla", "Fourier-feature": "Fourier-Feature", "PirateNet": "PirateNet"}
# Okabe-Ito colours; line style and marker differ as well, so the curves survive a greyscale print.
STYLE = {
    "Vanilla": dict(color="#0072B2", linestyle="--", marker="o"),
    "Fourier-feature": dict(color="#E69F00", linestyle="-.", marker="s"),
    "PirateNet": dict(color="#D55E00", linestyle="-", marker="^"),
}
Curves = Dict[Tuple[str, str], List[Tuple[int, float]]]


def read_losses(benchmark_folder: str, run: str) -> List[Tuple[int, float]]:
    """(epoch, training loss) pairs from the training log in <benchmark>/results/<run>/; empty if there is none.
    Every results folder holds one log (the first job of a continued run is in its subfolder first_job/)."""
    pairs = []
    for log in (ROOT / benchmark_folder / "results" / run).glob("full_time_ext_*.txt"):
        pairs += [(int(line.split()[1]), float(line.split()[-1])) for line in log.read_text().splitlines()
                  if line.startswith("Epoch ") and " - Total loss" in line]
    return pairs


def fetch_losses() -> None:
    """Rebuild the loss data file from the training logs; it is written only after every log has been read."""
    rows = []
    for (benchmark, network), (benchmark_folder, run) in RUNS.items():
        pairs = read_losses(benchmark_folder, run)
        if len(pairs) != EPOCHS // 50 or pairs[-1][0] != EPOCHS - 50:   # a loss line every 50 epochs, 0 to 9,950
            raise SystemExit(f"{run}: {len(pairs)} loss lines in its log, expected {EPOCHS // 50}; {LOSS_DATA.name} not changed")
        rows += [[benchmark, network, run, epoch, f"{loss:.6g}"] for epoch, loss in pairs]
        print(f"{benchmark:12s} {network:16s} {len(pairs):3d} loss values" + (f", last at epoch {pairs[-1][0]}" if pairs else ""))
    LOSS_DATA.parent.mkdir(parents=True, exist_ok=True)
    with LOSS_DATA.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["benchmark", "network", "run", "epoch", "loss"])
        writer.writerows(rows)


def load(path: Path, column: str) -> Curves:
    """Curves of the runs in RUNS, matched by run folder (the data files hold more runs than the figures show)."""
    curves = {key: [] for key in RUNS}  # type: Curves
    key_of_run = {run: key for key, (_, run) in RUNS.items()}
    if path.exists():
        with path.open() as handle:
            for row in csv.DictReader(handle):
                if row["run"] in key_of_run:
                    curves[key_of_run[row["run"]]].append((int(row["epoch"]), float(row[column])))
    for key, points in curves.items():
        if not points:  # e.g. a loss file that was not rebuilt with --fetch after RUNS changed
            print(f"warning: no rows in {path.name} for {RUNS[key][1]}; its curve is not drawn")
    return curves


class Row(NamedTuple):
    """One row of panels: a quantity for every run and how its curves are drawn."""
    curves: Curves
    ylabel: str
    markevery: int
    first_epoch: int        # earlier epochs are left out (the untrained start)
    mark_resampling: bool   # dotted lines at the epochs at which the collocation points are redrawn
    plain_ticks: bool       # 1, 10, 100 instead of powers of ten
    reference_levels: Tuple[float, ...] = ()   # dashed horizontal lines at these values, labelled in percent


def plot(rows: List[Row], name: str) -> None:
    """One row of three panels per quantity, one column per benchmark; panels are lettered row by row."""
    apply_style()
    fig, axes = plt.subplots(len(rows), 3, figsize=(TEXTWIDTH_IN, 0.3 + 1.5 * len(rows)), sharex=True, sharey="row",
                             squeeze=False)
    letters = iter(string.ascii_lowercase)
    for row_axes, row in zip(axes, rows):
        for ax, benchmark in zip(row_axes, BENCHMARKS):
            drawn = 0
            if row.mark_resampling:
                for epoch in range(RESAMPLING_INTERVAL, EPOCHS, RESAMPLING_INTERVAL):
                    ax.axvline(epoch, color="0.8", lw=0.5, ls=":", zorder=0)
            for level in row.reference_levels:
                ax.axhline(level, color="0.55", lw=0.5, ls="--", zorder=0)
                if ax is row_axes[0]:
                    # Labelled once per row (the axis is shared), below the line at the left edge of the first
                    # panel, where every curve is still far above these levels.
                    ax.annotate(f"{level:g}%", xy=(0.03, level), xycoords=ax.get_yaxis_transform(), xytext=(0, -1.5),
                                textcoords="offset points", ha="left", va="top", fontsize=6, color="0.35")
            for network in NETWORKS:
                points = [(e, v) for e, v in sorted(row.curves[(benchmark, network)]) if e >= row.first_epoch]
                if not points:
                    continue
                epochs, values = zip(*points)
                ax.plot(epochs, values, lw=1.0, ms=2.4, markevery=row.markevery, label=PANEL_LEGEND[network], **STYLE[network])
                drawn += 1
            if drawn == 0:
                ax.text(0.5, 0.5, "runs pending", transform=ax.transAxes, ha="center", va="center", color="0.5", fontsize=7)
            ax.set_yscale("log")
            ax.set_xlim(0, EPOCHS)
            ax.set_xticks([0, 5000, 10000])
            ax.set_xticklabels(["0", "5000", "10000"])
            if row.plain_ticks:
                ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda value, _: f"{value:g}"))
            # The curves fall from the upper left, so the upper right corner of every panel is free.
            ax.text(0.95, 0.93, f"({next(letters)})", transform=ax.transAxes, ha="right", va="top")
        row_axes[0].set_ylabel(row.ylabel)
    for ax, benchmark in zip(axes[0], BENCHMARKS):
        ax.set_title(benchmark)
    for ax in axes[-1]:
        ax.set_xlabel("Epoch")
    # One legend for all panels, below them, with every network even if its curve is not drawn yet.
    handles = [mpl.lines.Line2D([], [], lw=1.0, ms=2.4, **STYLE[network]) for network in NETWORKS]
    fig.legend(handles, [PANEL_LEGEND[network] for network in NETWORKS], loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.0), handlelength=2.6, columnspacing=1.6)
    fig.subplots_adjust(wspace=0.25, hspace=0.12)
    fig.savefig(ROOT / "figures" / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(ROOT / "figures" / f"{name}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote figures/{name}.pdf and .png (600 dpi)")


def main() -> None:
    if "--fetch" in sys.argv[1:]:
        fetch_losses()
    # The untrained networks (epoch 0) have an error of several hundred percent and are left out.
    plot([Row(load(ERROR_DATA, "error_percent"), "Time-averaged relative\n" r"$L_2$ error (%)", markevery=2, first_epoch=500, mark_resampling=False, plain_ticks=True,
              reference_levels=(1.0, 5.0)),   # author, 2026-10-12: "add a line at 1 percent and 5 percent"
          Row(load(LOSS_DATA, "loss"), r"Training loss $\mathcal{L}/2$", markevery=20, first_epoch=0, mark_resampling=True, plain_ticks=False)],
         "convergence_error_loss")


if __name__ == "__main__":
    main()
