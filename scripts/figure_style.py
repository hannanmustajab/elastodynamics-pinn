"""Style shared by all script-generated figures of the manuscript.

scienceplots styles science + nature + no-latex, as in the run scripts, with the fonts fixed to
Computer Modern so that every machine and scienceplots version draws the same figure (version
2.1.0 falls back to DejaVu Sans for this combination, 2.1.1 uses Computer Modern). The snapshot
script scripts/snapshots/plot_paper_snapshots.py sets the same values itself because it was run on
the training machines without this file.
"""
import logging

import matplotlib.pyplot as plt
import scienceplots  # noqa: F401  registers the styles below

# Body text width of the target class (sn-jnl, sn-nature): 372 pt.
TEXTWIDTH_IN = 372.0 / 72.27

logging.getLogger("fontTools").setLevel(logging.ERROR)  # timestamp notes while embedding the font in a PDF


def apply_style(base: float = 8, small: float = 7) -> None:
    """Activate the manuscript style: `base` points for labels, titles and legends, `small` for tick labels."""
    plt.style.use(["science", "nature", "no-latex"])
    plt.rcParams.update({
        "font.size": base, "axes.labelsize": base, "axes.titlesize": base,
        "xtick.labelsize": small, "ytick.labelsize": small, "legend.fontsize": base,
        "font.family": "serif", "font.serif": ["cmr10", "Computer Modern Serif", "DejaVu Serif"],
        "mathtext.fontset": "cm", "axes.formatter.use_mathtext": True,
        "pdf.fonttype": 42, "ps.fonttype": 42,  # embed TrueType, not Type-3 bitmaps
    })
