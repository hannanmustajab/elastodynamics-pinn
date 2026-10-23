#!/usr/bin/env python3
"""
Figures of the Marmousi2 case: the medium with the error of the three networks, and the centreline traces.

    figures/marmousi2_model_errors.{pdf,png}

    (a)  P-wave speed of the crop in km/s on the domain of the runs, with the source (star)
    (b)  relative L2 error against epoch, every 500 epochs from epoch 500 on; dashed lines at 1% and 5%
    (c)  relative L2 error against time after 10,000 epochs

    figures/marmousi2_traces_all_networks_t09.{pdf,png}
        left: u_x along x at y = 0     right: u_y along y at x = 0     at t = 0.9, the three networks and the
        reference on the same axes, in the layout of make_trace_figures.py (author, 2026-10-13: "with all three
        models on the same plot (just like figure 9)"); the lines are those of grid index 256 (coordinate 0.003)

Networks: PirateNet 3 x 128, Vanilla 9 x 128 and Fourier-Feature 9 x 128; colours and line styles of
make_convergence_figures.py. The first figure names them without sizes, the trace figure with sizes, as the
corresponding figures of the layered media do.

Data (figures/data/):
    marmousi2_crop_s010.npz        cosine-series coefficients of lambda and mu, the file the runs and the
                                   reference read (scripts/diagnostics/marmousi2_model.py, filter width 0.01)
    marmousi2_error_vs_epoch.csv   rows of the three runs, columns as error_vs_epoch.csv
    marmousi2_error_vs_time.csv    rows of the three runs, columns as error_vs_time.csv
    marmousi2_traces_t09.csv       the two centreline traces of the reference and of the three networks
The two error files are the rows of these runs in the output of the scoring script of the paper for all
checkpoints (every 500 epochs); this repository holds the final checkpoints only, so they are given as data.
--traces evaluates the networks after 10,000 epochs from the checkpoints in marmousi2/results/ and reads the
reference frame in marmousi2/ground_truth/, with the ansatz of
scripts/diagnostics/score_soap_runs.py, and writes the trace file; it prints the relative L2 error of each
network along the two lines, which is not the error of the tables.

Style: scripts/figure_style.py.

Usage:  python scripts/make_marmousi2_figures.py [--traces]
"""
import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from figure_style import TEXTWIDTH_IN, apply_style
from make_convergence_figures import PANEL_LEGEND, ROOT, STYLE

sys.path.insert(0, str(ROOT / "scripts" / "diagnostics"))
import marmousi2_model as m  # noqa: E402  (needs the path above)

DATA = ROOT / "figures" / "data"
MEDIUM, EPOCH_DATA, TIME_DATA = DATA / "marmousi2_crop_s010.npz", DATA / "marmousi2_error_vs_epoch.csv", DATA / "marmousi2_error_vs_time.csv"
TRACE_DATA, TRACE_TIME, LINE = DATA / "marmousi2_traces_t09.csv", 0.9, 256   # LINE: grid index of the two centrelines
TRACE_LEGEND = {"Vanilla": r"Vanilla $9\times128$", "Fourier-feature": r"Fourier-Feature $9\times128$", "PirateNet": r"PirateNet $3\times128$"}
REFERENCE_STYLE = dict(color="0.55", linestyle="-", lw=2.2)   # as in make_trace_figures.py
# Network of the figure -> run folder (without the suffix _FINISHED).
RUNS = {
    "Vanilla": "Marmousi2_s010_Vanilla_9x128_T1.0_soap_rad",
    "Fourier-feature": "Marmousi2_s010_FourierXB_9x128_T1.0_soap_rad",
    "PirateNet": "Marmousi2_s010_Pirate_3x128_T1.0_soap_rad",
}
YLABEL = r"Relative $L_2$ error (%)"
N, FIRST_EPOCH, EPOCHS = 512, 500, 10000
HEIGHT_IN, PANEL_IN, BOTTOM_IN = 2.25, 1.38, 0.62   # figure height, height of every panel, space below for labels and legend
Curve = List[Tuple[float, float]]


def load(path: Path, column: str) -> Dict[str, Curve]:
    """Network -> (epoch or time, error in percent), sorted; every network must have rows."""
    network_of = {run: network for network, run in RUNS.items()}
    curves = {network: [] for network in RUNS}  # type: Dict[str, Curve]
    with path.open() as handle:
        for row in csv.DictReader(handle):
            if row["run"] in network_of:
                curves[network_of[row["run"]]].append((float(row[column]), float(row["error_percent"])))
    missing = [network for network, points in curves.items() if not points]
    if missing:
        raise ValueError(f"no rows in {path.name} for {missing}")
    return {network: sorted(points) for network, points in curves.items()}


def wave_speed() -> np.ndarray:
    """P-wave speed in km/s on the evaluation grid, indexed [x, y]."""
    data = m.load(str(MEDIUM))
    axis = np.linspace(m.X_MIN, m.X_MIN + m.LENGTH, N)
    lam, mu = m.evaluate(data["lam"], axis, axis), m.evaluate(data["mu"], axis, axis)
    return float(data["v0_kms"]) * np.sqrt((lam + 2.0 * mu) / m.RHO)


def draw() -> None:
    by_epoch, by_time = load(EPOCH_DATA, "epoch"), load(TIME_DATA, "time")
    # Axes placed in inches: the square model panel with its colour bar, then the two error panels of the same height.
    fig = plt.figure(figsize=(TEXTWIDTH_IN, HEIGHT_IN))
    place = lambda left, width: fig.add_axes([left / TEXTWIDTH_IN, BOTTOM_IN / HEIGHT_IN, width / TEXTWIDTH_IN, PANEL_IN / HEIGHT_IN])
    ax_model, ax_bar, ax_epoch, ax_time = place(0.45, PANEL_IN), place(1.88, 0.06), place(2.62, 1.0), place(4.10, 1.0)
    low, high = m.X_MIN, m.X_MIN + m.LENGTH
    image = ax_model.imshow(wave_speed().T, extent=[low, high, low, high], origin="lower", cmap="viridis")
    ax_model.plot(0, 0, marker="*", markersize=5, markerfacecolor="gold", markeredgecolor="black", markeredgewidth=0.4)
    ax_model.set_xlabel("$x$")
    ax_model.set_ylabel("$y$")
    ax_model.set_xticks([-1.5, 0, 1.5])
    ax_model.set_yticks([-1.5, 0, 1.5])
    ax_model.set_title(r"(a) $V_{\mathrm{p}}^{*}$ (km/s)")
    fig.colorbar(image, cax=ax_bar)

    for level in (1.0, 5.0):  # as in convergence_error_loss
        ax_epoch.axhline(level, color="0.55", lw=0.5, ls="--", zorder=0)
    for network in RUNS:
        epochs, values = zip(*[(e, v) for e, v in by_epoch[network] if e >= FIRST_EPOCH])
        ax_epoch.plot(epochs, values, lw=1.0, ms=2.4, markevery=2, label=PANEL_LEGEND[network], **STYLE[network])
        ax_time.plot(*zip(*by_time[network]), lw=1.0, color=STYLE[network]["color"], linestyle=STYLE[network]["linestyle"])
    ax_epoch.set_yscale("log")
    ax_epoch.set_xlim(0, EPOCHS)
    ax_epoch.set_xticks([0, 5000, 10000])
    ax_epoch.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda value, _: f"{value:g}"))
    ax_epoch.set_xlabel("Epoch")
    ax_epoch.set_ylabel("Time-averaged relative\n" r"$L_2$ error (%)")
    ax_epoch.set_title("(b) During training")
    ax_time.set_xlim(0, 1)
    ax_time.set_ylim(bottom=0)
    ax_time.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=5))
    ax_time.set_xlabel(r"Time $t$")
    ax_time.set_ylabel(YLABEL)
    ax_time.set_title(r"(c) After $10{,}000$ epochs")
    handles, labels = ax_epoch.get_legend_handles_labels()
    # One legend for both error panels, below them and centred between them.
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(3.86 / TEXTWIDTH_IN, 0.0), handlelength=2.6,
               columnspacing=1.6)
    fig.savefig(ROOT / "figures" / "marmousi2_model_errors.pdf", bbox_inches="tight")
    fig.savefig(ROOT / "figures" / "marmousi2_model_errors.png", dpi=600, bbox_inches="tight")
    plt.close(fig)
    print("wrote figures/marmousi2_model_errors.pdf and .png (600 dpi)")


def collect_traces() -> None:
    """Evaluate the three networks and the reference on the two centrelines at TRACE_TIME and write the trace file."""
    import importlib.util
    import torch  # only this step needs torch and the checkpoints

    spec = importlib.util.spec_from_file_location("score_soap_runs", ROOT / "scripts" / "diagnostics" / "score_soap_runs.py")
    soap = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(soap)
    grid = torch.linspace(m.X_MIN, m.X_MIN + m.LENGTH, soap.N)
    gx, gy = torch.meshgrid(grid, grid, indexing="ij")
    points = torch.stack([torch.zeros(soap.N**2), gx.reshape(-1), gy.reshape(-1)], 1)
    frame = int(round(TRACE_TIME / soap.DT_FRAME))
    ref_x, ref_y = soap.load_reference(soap.RUNS[RUNS["PirateNet"]][6], frame + 1, torch.device("cpu"))
    columns = {"coordinate": grid.numpy(), "Numerical_ux": ref_x[frame][:, LINE].numpy(), "Numerical_uy": ref_y[frame][LINE, :].numpy()}
    for network, run in RUNS.items():
        if soap.RUNS[run][6] != soap.RUNS[RUNS["PirateNet"]][6]:
            raise SystemExit(f"{run} has another reference than the PirateNet run")
        net = soap.build(run)
        net.load_state_dict(torch.load(soap.checkpoints(run)[soap.FINAL_EPOCH], map_location="cpu")["model_u_state_dict"])
        net.eval()
        ux, uy = soap.predict(net, points, TRACE_TIME)
        columns[f"{network}_ux"], columns[f"{network}_uy"] = ux[:, LINE].numpy(), uy[LINE, :].numpy()
        errors = [100 * np.linalg.norm(columns[f"{network}_u{c}"] - columns[f"Numerical_u{c}"]) / np.linalg.norm(columns[f"Numerical_u{c}"])
                  for c in "xy"]
        print(f"{run}: relative L2 error along the lines, u_x {errors[0]:.2f}%, u_y {errors[1]:.2f}%")
    with TRACE_DATA.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(zip(*[[f"{value:.6g}" for value in column] for column in columns.values()]))
    print(f"wrote {TRACE_DATA.relative_to(ROOT)} ({soap.N} rows, t = {TRACE_TIME})")


def draw_traces() -> None:
    """u_x along x at y = 0 and u_y along y at x = 0: reference underneath, the three networks on top."""
    with TRACE_DATA.open() as handle:
        rows = list(csv.DictReader(handle))
    column = lambda name: np.array([float(row[name]) for row in rows])
    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH_IN, 2.15))
    for ax, component in zip(axes, "xy"):
        ax.plot(column("coordinate"), column(f"Numerical_u{component}"), **REFERENCE_STYLE)
        for network in RUNS:
            ax.plot(column("coordinate"), column(f"{network}_u{component}"), lw=0.9, color=STYLE[network]["color"],
                    linestyle=STYLE[network]["linestyle"])
        ax.set_xlim(m.X_MIN, m.X_MIN + m.LENGTH)
        ax.set_xlabel(f"${component}$")
        ax.set_title(f"$u_{component}$")
    handles = [mpl.lines.Line2D([], [], **REFERENCE_STYLE)] + [
        mpl.lines.Line2D([], [], lw=0.9, color=STYLE[network]["color"], linestyle=STYLE[network]["linestyle"]) for network in RUNS]
    fig.legend(handles, ["Numerical"] + [TRACE_LEGEND[network] for network in RUNS], loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.04), handlelength=2.6, columnspacing=2.0)
    fig.subplots_adjust(wspace=0.22)
    fig.savefig(ROOT / "figures" / "marmousi2_traces_all_networks_t09.pdf", bbox_inches="tight")
    fig.savefig(ROOT / "figures" / "marmousi2_traces_all_networks_t09.png", dpi=600, bbox_inches="tight")
    plt.close(fig)
    print("wrote figures/marmousi2_traces_all_networks_t09.pdf and .png (600 dpi)")


def main() -> None:
    if "--traces" in sys.argv[1:]:
        collect_traces()
    apply_style()
    draw()
    draw_traces()


if __name__ == "__main__":
    main()
