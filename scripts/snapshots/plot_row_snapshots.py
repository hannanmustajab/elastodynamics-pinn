#!/usr/bin/env python3
"""Wavefield snapshots of a run for print at a third of the text width, three of them in one row.

The run, the checks against its log, the reference and the output folder are those of
plot_paper_snapshots.py, which is loaded and run unchanged. Only the drawing differs: the same six
panels (prediction, reference, difference; u_x above u_y), colour range, interface lines and source
star, with the text larger by ROW_SCALE and the margins set for it, so that the printed text is as
large as in the snapshot figures that are two to a row (author, 2026-10-13: "only keep 3 in a row.
regenerate them in a bigger way so they are clear"). The files get the suffix _row, so the standard
files of plot_paper_snapshots.py stay; the traces are not drawn.

Usage (in a folder that also holds plot_paper_snapshots.py):
    python plot_row_snapshots.py <run folder> <time> [<time> ...] [--reference DIR] [--check-reference DIR]
Output: wavefield_t<time>_row.png in the folder plot_paper_snapshots.py writes to.
"""
from pathlib import Path
from typing import List

import plot_paper_snapshots as pp

plt = pp.plt
ROW_SCALE = 0.45 / 0.325    # subfigure widths in the manuscript: two to a row and three to a row


def draw_row(pinn: pp.Field, ref: pp.Field, lines: List[float], label: str, path: Path) -> None:
    """The six panels of pp.draw with text larger by ROW_SCALE. The lines behind the wavefield are drawn by
    pp.draw_interfaces as it is at the time of the call, so a wrapper that replaces it is respected."""
    size = 8 * ROW_SCALE
    # Taller than the standard figure (3.3 x 2.5 in), so that the square panels fill the width.
    fig, axes = plt.subplots(2, 3, figsize=(3.3, 2.9), gridspec_kw={"wspace": 0.08, "hspace": 0.08, "left": 0.01, "right": 0.99,
                                                                    "top": 0.915, "bottom": 0.185})
    for ax, title in zip(axes[0], [label, "Numerical", "Difference"]):
        ax.set_title(title, fontsize=size)
    for row in range(2):
        for ax, field in zip(axes[row], (pinn[row], ref[row], pinn[row] - ref[row])):
            image = ax.imshow(field.T, vmin=-0.2, vmax=0.2, cmap="seismic", extent=pp.EXTENT, origin="lower")
            pp.draw_interfaces(ax, field, lines)
    for ax in axes.flatten():
        ax.plot(0, 0, marker="*", markersize=3.5, markerfacecolor="gold", markeredgecolor="black", markeredgewidth=0.3, zorder=4)
        ax.set_xlim(pp.LOW, pp.HIGH)
        ax.set_ylim(pp.LOW, pp.HIGH)
        ax.set_xticks([])
        ax.set_yticks([])
    bar = fig.colorbar(image, cax=fig.add_axes([0.12, 0.105, 0.76, 0.03]), orientation="horizontal", ticks=[-0.2, -0.1, 0.0, 0.1, 0.2])
    bar.ax.tick_params(labelsize=size)
    fig.savefig(path, dpi=600)
    plt.close(fig)
    print(f"wrote {path}")


def use_row_layout() -> None:
    """Make pp.main draw the row figure in place of the standard one and no traces."""
    pp.draw = lambda pinn, ref, lines, label, path: draw_row(pinn, ref, lines, label, path.with_name(f"{path.stem}_row.png"))
    pp.draw_trace = lambda *args, **kwargs: None


def main() -> None:
    use_row_layout()
    pp.main()   # its lines "wrote ...wavefield_t<time>.png" name the standard file; the file written is the _row one


if __name__ == "__main__":
    main()
