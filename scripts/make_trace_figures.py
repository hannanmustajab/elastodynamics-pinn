#!/usr/bin/env python3
"""
Centreline traces of the six-layer medium at one time, all networks and the reference on the same axes.

    figures/sixlayer_traces_all_networks_t09.{pdf,png}
        left: u_x along x at y = 0 (panel title u_x)     right: u_y along y at x = 0 (panel title u_y)     at t = 0.9

Curves: the DEVITO reference ("Numerical", thick grey line underneath) and the networks after 10,000
epochs: Vanilla 5 x 128, Fourier-feature 5 x 128 and the PirateNet with the lowest error on this
window, 3 x 128 (author, 2026-10-12: one plot with PirateNet, Fourier, Vanilla and Numerical, to show
how far each is from the reference).

The traces lie on the grid line of index 256 of the 512 x 512 evaluation grid (coordinate 0.003), as
in scripts/snapshots/plot_paper_snapshots.py. Reference arrays are indexed [x, y]; snapshots_y holds u_x
and snapshots_x holds u_y (scripts/diagnostics/score_soap_runs.py, load_reference).

The networks are evaluated here from the final checkpoints in six_layer/results/, with the hard
initial-condition ansatz of scripts/diagnostics/score_soap_runs.py; the reference frame is read from
six_layer/ground_truth/. Before drawing,
the relative L2 error of each network along the two lines is printed; it is not the error of the
tables, which is taken over the whole grid.

Style: scienceplots through scripts/figure_style.py; network colours as in make_convergence_figures.py.

Usage:  python scripts/make_trace_figures.py
"""
import importlib.util
import math
from typing import Dict, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch

from figure_style import TEXTWIDTH_IN, apply_style
from make_convergence_figures import ROOT, STYLE
from make_energy_figures import save

TIME = 0.9
LINE = 256                      # grid index of the two centrelines
REFERENCE = "six_layer"
# Legend label -> (run in six_layer/, line style); drawn in this order.
NETWORKS = {
    r"Vanilla $5\times128$": ("SixLayer_Vanilla_5x128_T1.0_2batch_soap_rad", dict(color=STYLE["Vanilla"]["color"], linestyle="--")),
    r"Fourier-Feature $5\times128$": ("SixLayer_FourierXB_5x128_T1.0_soap_rad", dict(color=STYLE["Fourier-feature"]["color"], linestyle="-.")),
    r"PirateNet $3\times128$": ("SixLayer_Pirate_3x128_T1.0_soap_rad", dict(color=STYLE["PirateNet"]["color"], linestyle="-")),
}
REFERENCE_STYLE = dict(color="0.55", linestyle="-", lw=2.2)


def load_scorer():
    """The scoring script as a module: network construction, checkpoint lookup and constants."""
    spec = importlib.util.spec_from_file_location("score_soap_runs", ROOT / "scripts" / "diagnostics" / "score_soap_runs.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def network_traces(score, run: str, axis: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    """u_x along x at y = axis[LINE] and u_y along y at x = axis[LINE], at TIME, for the final checkpoint of a run.

    Raises:
        ValueError: if the run folder has no checkpoint after 10,000 epochs.
    """
    files = score.checkpoints(run)
    if score.FINAL_EPOCH not in files:
        raise ValueError(f"{run}: no checkpoint after {score.FINAL_EPOCH} epochs in {score.RUNS[run][2]}")
    net = score.build(run)
    net.load_state_dict(torch.load(files[score.FINAL_EPOCH], map_location="cpu")["model_u_state_dict"])
    net.eval()
    fixed = torch.full_like(axis, float(axis[LINE]))
    time = torch.full_like(axis, TIME)
    rise, decay = math.tanh(2.5 * TIME / score.T1) ** 2, math.exp(-0.5 * (1.5 * TIME / score.T1) ** 2)
    traces = []
    with torch.no_grad():
        for component, (x, y) in enumerate(((axis, fixed), (fixed, axis))):   # u_x along x, then u_y along y
            raw = net(torch.stack([time, x, y], 1))[:, component]
            source = (x, y)[component] * torch.exp(-(x**2 + y**2) / (2 * score.SIGMA**2)) / score.PEAK
            traces.append((raw * rise + source * decay).numpy())
    return traces[0], traces[1]


def main() -> None:
    """Evaluate the networks on the two centrelines and draw them with the reference."""
    apply_style()
    score = load_scorer()
    axis = torch.linspace(-1.5, 1.5, score.N)
    frame = int(round(TIME / score.DT_FRAME))
    ref_x, ref_y = score.load_reference(REFERENCE, frame + 1, torch.device("cpu"))
    reference = (ref_x[frame][:, LINE].numpy(), ref_y[frame][LINE, :].numpy())

    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH_IN, 2.15))
    for ax, values in zip(axes, reference):
        ax.plot(axis.numpy(), values, **REFERENCE_STYLE)
    for label, (run, style) in NETWORKS.items():
        traces = network_traces(score, run, axis)
        errors = [100 * np.linalg.norm(t - r) / np.linalg.norm(r) for t, r in zip(traces, reference)]
        print(f"{run}: relative L2 error along the lines, u_x {errors[0]:.2f}%, u_y {errors[1]:.2f}%")
        for ax, values in zip(axes, traces):
            ax.plot(axis.numpy(), values, lw=0.9, **style)
    for ax, component in zip(axes, "xy"):
        ax.set_xlim(-1.5, 1.5)
        ax.set_xlabel(f"${component}$")
        ax.set_title(f"$u_{component}$")
    handles = [mpl.lines.Line2D([], [], **REFERENCE_STYLE)] + [mpl.lines.Line2D([], [], lw=0.9, **style) for _, style in NETWORKS.values()]
    fig.legend(handles, ["Numerical"] + list(NETWORKS), loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.04), handlelength=2.6, columnspacing=2.0)
    fig.subplots_adjust(wspace=0.22)
    save(fig, "sixlayer_traces_all_networks_t09")


if __name__ == "__main__":
    main()
