#!/usr/bin/env python3
"""
Generate the combined velocity-model figure for the PINN elastodynamics manuscript.

Renders all three benchmark media as one three-panel figure:
    (a) homogeneous          no interfaces
    (b) two-layer            interface at y = 0
    (c) six-layer            five interfaces
All three on the domain [-1.5, 1.5]^2 of the runs.

Material parameters are taken verbatim from the manuscript
(sections/homogeneous_benchmark.tex, two_layer_benchmark.tex,
six_layer_benchmark.tex; tab:layer_props and tab:multilayer_speeds).
Wave speeds are DERIVED here from those parameters rather than copied, and
asserted against the published tables, so the figure cannot drift from the text.

Colour encodes P-wave speed V_p = sqrt((lambda + 2 mu)/rho) on a single shared
scale across all three panels, so the panels are directly comparable as a
complexity ladder. Each layer is annotated with its defining Lame parameters.

Output: figures/velocity_models.pdf (vector) and .png (600 dpi).

Usage:  python scripts/make_velocity_models.py
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from figure_style import apply_style

# ----------------------------------------------------------------------------
# Model definitions -- verbatim from the manuscript
# ----------------------------------------------------------------------------
RHO = 100.0

# Body text width of the target class (sn-jnl, sn-nature): 372 pt.
# The figure is authored at exactly this width so that \includegraphics
# [width=\textwidth] performs NO scaling and label point sizes survive to print.
TEXTWIDTH_IN = 372.0 / 72.27

# Homogeneous benchmark: lambda=20, mu=30, rho=100
HOMO = {"lam": 20.0, "mu": 30.0, "half": 1.5}

# Two-layer benchmark: interface at y=0; upper (y>0) then lower (y<0)
TWO = {"upper": (20.0, 30.0), "lower": (25.0, 35.0), "half": 1.5, "interface": 0.0}

# Six-layer benchmark: layer 1 at the bottom (y = -1.5), layer 6 at the top; equal thickness.
# This is the order of the run scripts and of the reference solution: lambda and mu increase with y.
SIX = {
    "lam": [20.0, 30.0, 40.0, 50.0, 60.0, 70.0],
    "mu": [10.0, 15.0, 25.0, 35.0, 45.0, 55.0],
    "half": 1.5,
}

# Published wave speeds, used only as an assertion target
TABLE_VP = [0.6325, 0.7746, 0.9487, 1.0954, 1.2247, 1.3416]
TABLE_VS = [0.3162, 0.3873, 0.5000, 0.5916, 0.6708, 0.7416]


def panel_letter(ax, letter, dx=-0.16, dy=1.03):
    """Bold panel letter outside the top-left of the axes box."""
    ax.text(dx, dy, letter, transform=ax.transAxes, fontweight="bold",
            fontsize=mpl.rcParams["font.size"] + 1, va="bottom", ha="left")


def vp(lam, mu, rho=RHO):
    """P-wave speed."""
    return np.sqrt((lam + 2.0 * mu) / rho)


def vs(lam, mu, rho=RHO):
    """S-wave speed."""
    return np.sqrt(mu / rho)


def verify_against_manuscript():
    """Fail loudly if derived speeds disagree with the published tables."""
    got_vp = [vp(l, m) for l, m in zip(SIX["lam"], SIX["mu"])]
    got_vs = [vs(l, m) for l, m in zip(SIX["lam"], SIX["mu"])]
    for got, want, name in ((got_vp, TABLE_VP, "V_p"), (got_vs, TABLE_VS, "V_s")):
        for i, (g, w) in enumerate(zip(got, want), start=1):
            assert abs(g - w) < 5e-5, f"{name} layer {i}: derived {g:.6f} != table {w}"
    return got_vp, got_vs


# ----------------------------------------------------------------------------
# Figure
# ----------------------------------------------------------------------------
def build_figure():
    apply_style(base=9, small=7)   # scienceplots, shared with the other figure scripts
    # Filled panels: ticks on the left and bottom only, no minor ticks, titles at the left.
    mpl.rcParams.update({"axes.titlelocation": "left", "xtick.top": False, "ytick.right": False,
                         "xtick.minor.visible": False, "ytick.minor.visible": False})
    six_vp, _ = verify_against_manuscript()

    # Shared colour scale spans every V_p appearing in any panel.
    all_vp = six_vp + [vp(*TWO["upper"]), vp(*TWO["lower"]), vp(HOMO["lam"], HOMO["mu"])]
    vmin, vmax = min(all_vp), max(all_vp)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = mpl.cm.viridis

    # The manuscript body is single-column with \textwidth = 372 pt = 5.147 in.
    # The figure is drawn AT that final width so \includegraphics applies no
    # scaling and in-figure type renders at its nominal point size.
    fig, axes = plt.subplots(1, 3, figsize=(TEXTWIDTH_IN, 2.02))

    def draw_layer(ax, y0, y1, half, lam, mu, label_fs, ytext=None):
        """Fill one layer and annotate it with its Lame parameters.

        V_p is carried by colour (and read precisely off the colourbar, whose
        ticks sit at the six-layer speeds), so the in-layer text stays to a
        single short line and remains legible at final print size.

        ytext overrides the label's vertical position, used where the layer
        centre would collide with the source marker at the origin.
        """
        speed = vp(lam, mu)
        ax.add_patch(
            Rectangle(
                (-half, y0), 2 * half, y1 - y0,
                facecolor=cmap(norm(speed)), edgecolor="none", zorder=1,
            )
        )
        # White text is legible on viridis except at the top of the ramp.
        txt_col = "white" if norm(speed) < 0.62 else "black"
        ax.text(
            0.0, 0.5 * (y0 + y1) if ytext is None else ytext,
            f"$\\lambda={lam:g}$, $\\mu={mu:g}$",
            ha="center", va="center", fontsize=label_fs, color=txt_col,
            zorder=3,
        )

    # ---- (a) homogeneous ----
    # Label is offset above centre: the layer centre coincides with the source
    # marker at the origin.
    ax = axes[0]
    h = HOMO["half"]
    draw_layer(ax, -h, h, h, HOMO["lam"], HOMO["mu"], 6.2, ytext=0.42 * h)
    ax.set_title("No interfaces")

    # ---- (b) two-layer ----
    ax = axes[1]
    h, yi = TWO["half"], TWO["interface"]
    draw_layer(ax, yi, h, h, *TWO["upper"], 6.2, ytext=0.55 * h)   # y > 0
    draw_layer(ax, -h, yi, h, *TWO["lower"], 6.2, ytext=-0.55 * h)  # y < 0
    ax.plot([-h, h], [yi, yi], color="black", lw=1.2, zorder=4)
    ax.set_title("One interface")

    # ---- (c) six-layer ----
    ax = axes[2]
    h = SIX["half"]
    n = len(SIX["lam"])
    edges = np.linspace(-h, h, n + 1)   # bottom -> top, equal thickness
    for i in range(n):
        draw_layer(ax, edges[i], edges[i + 1], h, SIX["lam"][i], SIX["mu"][i], 5.6)
    for y in edges[1:-1]:
        ax.plot([-h, h], [y, y], color="black", lw=0.8, zorder=4)
    ax.set_title("Five interfaces")

    # ---- shared axis treatment; source marker ----
    for ax, half in zip(axes, (HOMO["half"], TWO["half"], SIX["half"])):
        ax.set_xlim(-half, half)
        ax.set_ylim(-half, half)
        ax.set_aspect("equal")
        ax.set_xlabel("$x$", labelpad=1.0)
        # Tick spacing is chosen per panel so labels never abut: the wider
        # six-layer domain gets a coarser step than the unit-square panels.
        step = 0.5 if half <= 1.0 else 0.75
        ticks = np.arange(-half, half + 1e-9, step)
        # Drop the shared bottom-left corner tick on x: its label would collide
        # with the identical y-axis label at the same corner.
        ax.set_xticks([t for t in ticks if abs(t + half) > 1e-9])
        ax.set_yticks(ticks)
        ax.tick_params(labelsize=5.2, length=2.0, pad=1.4)
        # Source location: Gaussian initial condition centred at the origin.
        ax.plot(0, 0, marker="*", ms=6.5, mfc="white", mec="black", mew=0.7, zorder=6)

    axes[0].set_ylabel("$y$", labelpad=1.0)
    for ax in axes[1:]:
        ax.set_ylabel("")

    # Legend entry for the source marker, below the panel row.
    src = mpl.lines.Line2D([], [], marker="*", ms=6.5, mfc="white", mec="black",
                           mew=0.7, ls="none", label="source")
    axes[1].legend(handles=[src], loc="upper center", bbox_to_anchor=(0.5, -0.24),
                   frameon=False, fontsize=6.0, handletextpad=0.2, borderpad=0.0,
                   handlelength=1.0)

    # ---- shared colourbar; ticks sit AT the six-layer wave speeds so the
    # colour of any layer can be read back to a number exactly.
    fig.subplots_adjust(left=0.075, right=0.855, bottom=0.155, top=0.855, wspace=0.38)
    cax = fig.add_axes([0.875, 0.155, 0.018, 0.70])
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cb.set_label(r"$V_\mathrm{p}$ (dimensionless)", fontsize=6.2, labelpad=2.0)
    cb.ax.tick_params(labelsize=5.6, length=1.8, pad=1.2)
    cb.set_ticks([round(v, 2) for v in six_vp])
    cb.outline.set_linewidth(0.5)

    for ax, letter in zip(axes, "abc"):
        panel_letter(ax, letter, dx=-0.22, dy=1.04)

    return fig


if __name__ == "__main__":
    fig = build_figure()
    fig.savefig("figures/velocity_models.pdf", bbox_inches="tight")
    fig.savefig("figures/velocity_models.png", dpi=600, bbox_inches="tight")
    print("wrote figures/velocity_models.pdf and .png (600 dpi)")
