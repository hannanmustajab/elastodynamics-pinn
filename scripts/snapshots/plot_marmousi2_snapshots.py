#!/usr/bin/env python3
"""Wavefield snapshots of a Marmousi2 run for the manuscript, drawn on the machine that holds the run.

The panels, colours, fonts, checks against the log and the output folder are those of
plot_paper_snapshots.py, which is loaded and run unchanged. The Marmousi2 medium has no layer
interfaces, so the grey interface lines of that script are replaced by contours of the P-wave speed
of the medium (from marmousi2_crop.npz in the run folder), drawn behind the wavefield in the same
way: visible where the panel is at rest, faded where the displacement is large. A second output is
the P-wave speed of the crop with the source, in the layout of the velocity-model figure.

With --row the wavefield figure is drawn by plot_row_snapshots.py for print at a third of the text
width, three of them in one row, with the same contours behind the wavefield. These files get the
suffix _row; the traces and the velocity model are not drawn, and the standard files stay.

Usage (marmousi2_model.py is read from scripts/diagnostics/):
    python plot_marmousi2_snapshots.py <run folder> <time> [<time> ...] [--reference DIR] [--check-reference DIR] [--row]
Output: as plot_paper_snapshots.py, plus velocity_model.png in the same folder; with --row only
wavefield_t<time>_row.png there.
"""
import sys
from pathlib import Path
from typing import List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "diagnostics"))   # marmousi2_model.py
import marmousi2_model as m  # noqa: E402
import plot_paper_snapshots as pp
import plot_row_snapshots

plt = pp.plt
CONTOUR_LEVELS = 4      # contours of V_p between its smallest and largest value in the crop
REST, MOVING = 0.45, 0.04   # as pp.draw_interfaces: opacity at rest, displacement at which a line has faded


def wave_speed(run: Path) -> np.ndarray:
    """P-wave speed of the medium of the run on the 512 x 512 evaluation grid, indexed [x, y]."""
    data = m.load(str(pp.code_folder(run) / "marmousi2_crop.npz"))
    axis = np.linspace(pp.LOW, pp.HIGH, pp.N)
    lam, mu = m.evaluate(data["lam"], axis, axis), m.evaluate(data["mu"], axis, axis)
    return np.sqrt((lam + 2.0 * mu) / m.RHO)


def structure_lines(ax, field: np.ndarray, speed: np.ndarray) -> None:
    """Contours of the wave speed behind the wavefield, faded where |u| is large."""
    axis = np.linspace(pp.LOW, pp.HIGH, pp.N)
    levels = np.linspace(speed.min(), speed.max(), CONTOUR_LEVELS + 2)[1:-1]
    contours = ax.contour(axis, axis, speed.T, levels=levels, colors="none", linewidths=0.4, zorder=3)
    to_index = lambda value: np.clip(np.rint((value - pp.LOW) / (pp.HIGH - pp.LOW) * (pp.N - 1)).astype(int), 0, pp.N - 1)
    for segment in (s for level in contours.allsegs for s in level if len(s) > 1):
        middle = 0.5 * (segment[:-1] + segment[1:])
        strength = np.abs(field[to_index(middle[:, 0]), to_index(middle[:, 1])])
        opacity = REST * (1.0 - np.clip(strength / MOVING, 0.0, 1.0))
        pieces = np.stack([segment[:-1], segment[1:]], axis=1)
        ax.add_collection(pp.LineCollection(pieces, colors=[(0, 0, 0, value) for value in opacity], linewidths=0.4, zorder=3))


def draw_velocity_model(speed: np.ndarray, velocity_scale: float, path: Path) -> None:
    """P-wave speed of the crop in km/s with the source, y upward."""
    fig, ax = plt.subplots(figsize=(3.3, 2.9))
    image = ax.imshow((velocity_scale * speed).T, extent=pp.EXTENT, origin="lower", cmap="viridis")
    ax.plot(0, 0, marker="*", markersize=6, markerfacecolor="gold", markeredgecolor="black", markeredgewidth=0.4)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    fig.colorbar(image, ax=ax, label="$V_p$ (km/s)")
    fig.savefig(path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    row = "--row" in sys.argv[1:]
    sys.argv = [a for a in sys.argv if a != "--row"]   # pp.main does not know this flag
    # The run folder is the first argument that is neither a flag nor the folder a flag names, as pp.main reads it.
    args: List[str] = sys.argv[1:]
    for flag in ("--reference", "--check-reference"):
        if flag in args:
            position = args.index(flag)
            args = args[:position] + args[position + 2:]
    run = Path(args[0]).resolve()
    speed = wave_speed(run)
    pp.interfaces = lambda tree: []   # no layer interfaces in this medium
    pp.draw_interfaces = lambda ax, field, lines: structure_lines(ax, field, speed)
    if row:
        plot_row_snapshots.use_row_layout()   # draws the lines with pp.draw_interfaces as replaced above
        pp.main()   # its lines "wrote ...wavefield_t<time>.png" name the standard file; the file written is the _row one
        return
    pp.main()
    # The folder pp.main wrote to: plots/paper for a finished run, else the interim folder of the newest checkpoint.
    window = next(run.glob("experiments/full_time_ext_tmax_*"))
    finished = (window / "STAGE_DONE.json").exists()
    interim = sorted(window.glob("plots/paper_interim_epoch_*"), key=lambda p: int(p.name.rsplit("_", 1)[1]))
    target = window / "plots" / "paper" if finished else interim[-1]
    data = m.load(str(pp.code_folder(run) / "marmousi2_crop.npz"))
    draw_velocity_model(speed, float(data["v0_kms"]), target / "velocity_model.png")
    print(f"wrote {target / 'velocity_model.png'}")


if __name__ == "__main__":
    main()
