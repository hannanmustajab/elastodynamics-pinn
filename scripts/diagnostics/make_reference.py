"""Write a reference solution for the divergence form rho u_tt = div(sigma(u)), with lambda(y)
and mu(y) inside the divergence, for one of the benchmark media.

Output layout matches the existing reference folders so main.py reads it unchanged:

    <out_dir>/snapshots_x/snapshot_%04d.npy   layer-normal component (u_y of the PINN)
    <out_dir>/snapshots_y/snapshot_%04d.npy   layer-parallel component (u_x of the PINN)
    <out_dir>/snapshots_u/snapshot_%04d.npy   displacement magnitude

Arrays are float32, 512 x 512, indexed [x, y] on [-1.5, 1.5]^2; frame i is at t = 0.01 i.
The swapped file names follow the convention of the earlier reference, which the plotting
code in main.py expects. The magnitude metric does not depend on it.

Usage: python make_reference.py <out_dir> [refine=2] [t_end=2.2] [medium=six_layer]
Media: six_layer, homogeneous, two_layer (values listed from y = -1.5 upward, as in main.py).
"""
import sys
from pathlib import Path
from typing import Dict

import numpy as np

import fd_two_forms as fd

# Lame parameters per layer and the y positions of the interfaces. "six_layer" keeps the values
# defined in fd_two_forms. Density is 100 and the interface smoothing tanh(90 (y - y_c)) in all media.
MEDIA = {
    "six_layer": None,
    "homogeneous": dict(LAMBDA=[20.0], MU=[30.0], CENTERS=np.array([])),
    "two_layer": dict(LAMBDA=[25.0, 20.0], MU=[35.0, 30.0], CENTERS=np.array([0.0])),
}  # type: Dict[str, dict]


def main() -> None:
    out = Path(sys.argv[1])
    refine = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    t_end = float(sys.argv[3]) if len(sys.argv) > 3 else 2.2
    medium = sys.argv[4] if len(sys.argv) > 4 else "six_layer"
    if medium not in MEDIA:
        raise SystemExit(f"unknown medium {medium!r}; choose from {', '.join(MEDIA)}")
    if MEDIA[medium] is not None:
        fd.LAMBDA, fd.MU, fd.CENTERS = MEDIA[medium]["LAMBDA"], MEDIA[medium]["MU"], MEDIA[medium]["CENTERS"]
    print(f"medium {medium}: lambda {fd.LAMBDA}, mu {fd.MU}, interfaces at y = {list(fd.CENTERS)}", flush=True)
    n_frames = int(round(t_end / fd.DT_FRAME)) + 1
    fd.FRAMES = list(range(n_frames))
    # Padding so that no wave reflected at the outer boundary re-enters [-1.5, 1.5]^2 before t_end:
    # the fastest P speed of the three media is 1.342, so the half-width must exceed (1.342 t_end + 1.5) / 2.
    fd.PAD_CELLS = 200
    half_width = 1.5 + fd.PAD_CELLS * 3.0 / (fd.N_EVAL - 1)
    assert half_width > (1.342 * t_end + 1.5) / 2 + 0.2, "padding too small for this t_end"

    frames = fd.solve("div", refine, t_end)
    for name in ("snapshots_x", "snapshots_y", "snapshots_u"):
        (out / name).mkdir(parents=True, exist_ok=True)
    for i in range(n_frames):
        ux, uy = frames[i]
        np.save(out / "snapshots_x" / f"snapshot_{i:04d}.npy", uy.astype(np.float32))
        np.save(out / "snapshots_y" / f"snapshot_{i:04d}.npy", ux.astype(np.float32))
        np.save(out / "snapshots_u" / f"snapshot_{i:04d}.npy", np.hypot(ux, uy).astype(np.float32))
    print(f"wrote {n_frames} frames to {out} (medium {medium}, refine {refine}, t_end {t_end})")


if __name__ == "__main__":
    main()
