"""DEVITO reference for a medium that varies in x and y (crop of Marmousi2, see marmousi2_model.py).

Equation, grid, time step, initial displacement and output layout are those of devito_check.py, the
reference of the three layered media: rho u_tt = div(sigma(u)) in displacement form, fourth-order
differences in space, second-order leapfrog in time, frames at t = 0.01 i on the 512 x 512
evaluation grid over [-1.5, 1.5]^2. New here: lambda and mu are functions of x and y, so the
product rule adds the terms with d(lambda)/dx and d(mu)/dx:

  (div sigma)_x = (lam + 2 mu) ux_xx + mu ux_yy + (lam + mu) uy_xy
                  + lam_x (ux_x + uy_y) + 2 mu_x ux_x + mu_y (ux_y + uy_x)
  (div sigma)_y = (lam + 2 mu) uy_yy + mu uy_xx + (lam + mu) ux_xy
                  + lam_y (ux_x + uy_y) + 2 mu_y uy_y + mu_x (ux_y + uy_x)

The derivatives of lambda and mu are the exact derivatives of the cosine series.

Usage:
  python devito_marmousi2.py selftest [refine=1]
      (1) the six-layer medium through this solver against devito_check.solve;
      (2) the six-layer medium turned by 90 degrees against (1) with x and y exchanged, which
          exercises the terms with lam_x and mu_x
  python devito_marmousi2.py reference <coefficients.npz> <out_dir> [refine=2] [t_end=1.0] [exterior=mirror]
      all frames, in the layout of the reference folders: snapshots_x (u_y of the PINN),
      snapshots_y (u_x of the PINN), snapshots_u (magnitude). exterior: how the medium continues
      outside [-1.5, 1.5]^2, "mirror" (the series as it is) or "clamp" (constant along the normal to
      the nearest edge). Inside the domain the medium is the same. The absorbing condition of the
      PINN, with the local wave speeds at each boundary point, assumes the second, which is also how
      the layered references continue their media. Until the fronts leave the domain the two agree
      (0.07% at t = 1 for the crop of 2026-10-12); at t = 2 they differ by 14% there, so a reference
      for t > 1 must use "clamp".
  python devito_marmousi2.py layered <tanh factor k> <out_dir> [refine=2] [t_end=1.0]
      the six-layer medium of the paper with interfaces tanh(k (y - y_c)) instead of k = 90 (10%-90% width
      2.197 / k), in the same layout; the outer layers continue beyond the domain
  python devito_marmousi2.py remaining <folder>
      norm of the displacement left in [-1.5, 1.5]^2, relative to t = 0
  python devito_marmousi2.py compare <folder A> <folder B>
      frame-by-frame difference of two reference folders, in percent of B
"""
import sys
import time
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import numpy as np

import devito_check as dc
import fd_two_forms as fd
import marmousi2_model as m

PAD_CELLS = 200   # as devito_check.solve: no reflection from the outer edge returns before t = 2.4
EXTERIORS = ("mirror", "clamp")   # continuation of the medium outside [-1.5, 1.5]^2, see series_material
Material = Dict[str, np.ndarray]   # lam, mu, lam_x, lam_y, mu_x, mu_y on the padded grid, indexed [x, y]


def padded_grid(refine: int) -> Tuple[np.ndarray, float, int]:
    """Coordinates of the padded grid along one axis, its spacing and the padding in grid points."""
    h = 3.0 / (fd.N_EVAL - 1) / refine
    pad = PAD_CELLS * refine
    n = (fd.N_EVAL - 1) * refine + 1 + 2 * pad
    return -1.5 + h * (np.arange(n) - pad), h, pad


def series_material(coefficient_file: str, coords: np.ndarray, exterior: str = "mirror") -> Material:
    """Lame parameters of a coefficient file and their derivatives on the grid coords x coords.

    Outside [-1.5, 1.5]^2 the series continues the medium by mirroring ("mirror"). With "clamp" the
    medium outside is constant along the normal to the nearest edge. Both are continuous with
    continuous first derivatives at the edges, where the normal derivative of the series is zero.
    """
    if exterior not in EXTERIORS:
        raise ValueError(f"exterior must be one of {EXTERIORS}")
    data = m.load(coefficient_file)
    if float(data["rho"]) != fd.RHO:
        raise ValueError("density of the coefficient file differs from fd_two_forms.RHO")
    clamp = exterior == "clamp"
    points = np.clip(coords, m.X_MIN, m.X_MIN + m.LENGTH) if clamp else coords
    varies = ((coords >= m.X_MIN) & (coords <= m.X_MIN + m.LENGTH)).astype(float) if clamp else np.ones(len(coords))
    material = {}
    for name in ("lam", "mu"):
        material[name] = m.evaluate(data[name], points, points)
        material[name + "_x"] = m.evaluate(data[name], points, points, dx=True) * varies[:, None]
        material[name + "_y"] = m.evaluate(data[name], points, points, dy=True) * varies[None, :]
    return material


def layered_material(coords: np.ndarray, along_x: bool) -> Material:
    """The six-layer medium of fd_two_forms; with along_x the profile varies in x instead of y."""
    zero = np.zeros((len(coords), len(coords)))
    material = {}
    for name, values in (("lam", fd.LAMBDA), ("mu", fd.MU)):
        profile, slope = fd.layered(values, coords), dc.profile_derivative(values, coords)
        if along_x:
            material[name], material[name + "_x"], material[name + "_y"] = zero + profile[:, None], zero + slope[:, None], zero
        else:
            material[name], material[name + "_x"], material[name + "_y"] = zero + profile[None, :], zero, zero + slope[None, :]
    return material


def solve(material: Material, refine: int, levels: List[int]) -> Iterator[Tuple[int, np.ndarray, np.ndarray]]:
    """(level, u_x, u_y) on the evaluation grid for the requested time levels (level n is t = n dt), in order."""
    from devito import Eq, Function, Grid, Operator, TimeFunction, configuration
    from devito import solve as devito_solve
    configuration["log-level"] = "WARNING"
    dc.check_buffer_indexing()

    coords, h, pad = padded_grid(refine)
    n, dt = len(coords), 1e-3 / refine
    grid = Grid(shape=(n, n), extent=((n - 1) * h, (n - 1) * h), dtype=np.float64)
    speed = np.sqrt((material["lam"] + 2 * material["mu"]).max() / fd.RHO)
    print(f"n = {n}, h = {h:.5f}, dt = {dt}, largest V_p {speed:.3f}, Courant number {speed * dt / h:.2f}", flush=True)

    c = {}
    for name, values in material.items():
        c[name] = Function(name=name, grid=grid, space_order=4)
        c[name].data[:] = values

    def divergence(ux, uy):
        lam, mu = c["lam"], c["mu"]
        div_x = ((lam + 2 * mu) * ux.dx2 + mu * ux.dy2 + (lam + mu) * uy.dx.dy
                 + c["lam_x"] * (ux.dx + uy.dy) + 2 * c["mu_x"] * ux.dx + c["mu_y"] * (ux.dy + uy.dx))
        div_y = ((lam + 2 * mu) * uy.dy2 + mu * uy.dx2 + (lam + mu) * ux.dx.dy
                 + c["lam_y"] * (ux.dx + uy.dy) + 2 * c["mu_y"] * uy.dy + c["mu_x"] * (ux.dy + uy.dx))
        return div_x, div_y

    # Level 0 is the initial displacement; level 1 the second-order start with zero initial velocity.
    u0x, u0y = fd.initial_displacement(coords, coords)
    start_x, start_y = Function(name="sx", grid=grid, space_order=4), Function(name="sy", grid=grid, space_order=4)
    acc_x, acc_y = Function(name="ax", grid=grid, space_order=4), Function(name="ay", grid=grid, space_order=4)
    start_x.data[:], start_y.data[:] = u0x, u0y
    d0x, d0y = divergence(start_x, start_y)
    Operator([Eq(acc_x, d0x / fd.RHO), Eq(acc_y, d0y / fd.RHO)])()

    ux = TimeFunction(name="ux", grid=grid, space_order=4, time_order=2)
    uy = TimeFunction(name="uy", grid=grid, space_order=4, time_order=2)
    ux.data[0], uy.data[0] = u0x, u0y
    ux.data[1], uy.data[1] = u0x + 0.5 * dt**2 * np.asarray(acc_x.data), u0y + 0.5 * dt**2 * np.asarray(acc_y.data)
    dx, dy = divergence(ux, uy)
    op = Operator([Eq(ux.forward, devito_solve(fd.RHO * ux.dt2 - dx, ux.forward)),
                   Eq(uy.forward, devito_solve(fd.RHO * uy.dt2 - dy, uy.forward))])

    sub = slice(pad, n - pad, refine)
    start = 1
    for level in levels:
        if level >= 2:
            op(time_m=start, time_M=level - 1, dt=dt)
            start = level
        yield level, np.asarray(ux.data[level % 3])[sub, sub].copy(), np.asarray(uy.data[level % 3])[sub, sub].copy()


def selftest(refine: int) -> None:
    """The new solver against devito_check.solve for the six-layer medium, and the medium turned by 90 degrees."""
    coords, _, _ = padded_grid(refine)
    levels = [int(round(t * refine / 1e-3)) for t in (0.5, 1.0)]
    old = {level: (ux, uy) for level, ux, uy in dc.solve("div", refine, levels)}
    new = {level: (ux, uy) for level, ux, uy in solve(layered_material(coords, along_x=False), refine, levels)}
    turned = {level: (ux, uy) for level, ux, uy in solve(layered_material(coords, along_x=True), refine, levels)}
    print("relative L2 difference of the displacement field, percent")
    for level in levels:
        exchanged = (new[level][1].T, new[level][0].T)   # u_x(x, y) of the turned medium is u_y(y, x) of the original
        print(f"  t = {level * 1e-3 / refine:.1f}: six-layer medium, this solver against devito_check: {dc.difference(new[level], old[level]):.2e}; "
              f"turned medium against the original with x and y exchanged: {dc.difference(turned[level], exchanged):.2e}; "
              f"turned medium against the original as it is (must be large): {dc.difference(turned[level], new[level]):.1f}")


def reference(coefficient_file: str, out_dir: str, refine: int, t_end: float, exterior: str = "mirror") -> None:
    """All frames of the medium of a coefficient file, in the layout the run scripts read."""
    coords, _, _ = padded_grid(refine)
    material = series_material(coefficient_file, coords, exterior)
    # A wave that leaves [-1.5, 1.5]^2, is reflected at the outer edge of the grid and comes back travels
    # at least 2 * coords[-1] - 1.5; it must not arrive before t_end.
    speed = np.sqrt((material["lam"] + 2 * material["mu"]).max() / fd.RHO)
    if (2 * coords[-1] - 1.5) / speed <= t_end:
        raise ValueError(f"padding too small for t_end = {t_end}: the outer edge would be felt after {(2 * coords[-1] - 1.5) / speed:.2f}")
    out = Path(out_dir)
    for name in ("snapshots_x", "snapshots_y", "snapshots_u"):
        (out / name).mkdir(parents=True, exist_ok=True)
    steps_per_frame = int(round(fd.DT_FRAME / (1e-3 / refine)))
    n_frames = int(round(t_end / fd.DT_FRAME)) + 1
    began = time.time()
    for level, ux, uy in solve(material, refine, [i * steps_per_frame for i in range(n_frames)]):
        i = level // steps_per_frame
        if not (np.isfinite(ux).all() and np.isfinite(uy).all()):
            raise RuntimeError(f"frame {i} is not finite")
        np.save(out / "snapshots_x" / f"snapshot_{i:04d}.npy", uy.astype(np.float32))
        np.save(out / "snapshots_y" / f"snapshot_{i:04d}.npy", ux.astype(np.float32))
        np.save(out / "snapshots_u" / f"snapshot_{i:04d}.npy", np.hypot(ux, uy).astype(np.float32))
    # Written last, so that only a complete folder names its medium (setup_marmousi2_run.py checks it).
    # The medium inside the domain is the same for both exteriors; the note names the exterior as well.
    (out / m.MEDIUM_NOTE).write_text(f"{m.sha256(coefficient_file)}  {Path(coefficient_file).name}  exterior={exterior}\n")
    edge = max(np.abs(field[index]).max() for field in (ux, uy) for index in (np.s_[0, :], np.s_[-1, :], np.s_[:, 0], np.s_[:, -1]))
    print(f"wrote {n_frames} frames to {out} ({Path(coefficient_file).name}, refine {refine}, t_end {t_end}, exterior {exterior}, {time.time() - began:.0f} s); "
          f"largest displacement on the edge of [-1.5, 1.5]^2 at t = {t_end}: {edge:.2e}")


def compare(folder_a: str, folder_b: str) -> None:
    """Difference of every frame of folder A from folder B."""
    n = min(len(list((Path(folder) / "snapshots_x").glob("snapshot_*.npy"))) for folder in (folder_a, folder_b))
    values = [dc.difference(dc.stored(folder_a, i), dc.stored(folder_b, i)) for i in range(n)]
    picks = ", ".join(f"t = {i / 100:.2f}: {values[i]:.3f}" for i in range(25, n, 25))
    print(f"{Path(folder_a).name} against {Path(folder_b).name}: {n} frames, mean {np.mean(values):.3f}%, max {np.max(values):.3f}% ({picks})")


LAYERED_NOTE = "medium.txt"   # in a six-layer reference folder: the tanh factor of its interfaces


def layered(sharpness: float, out_dir: str, refine: int, t_end: float) -> None:
    """All frames of the six-layer medium with interfaces tanh(sharpness (y - y_c))."""
    fd.SHARPNESS = sharpness   # read by fd.layered and dc.profile_derivative when layered_material is built
    coords, _, _ = padded_grid(refine)
    material = layered_material(coords, along_x=False)
    out = Path(out_dir)
    for name in ("snapshots_x", "snapshots_y", "snapshots_u"):
        (out / name).mkdir(parents=True, exist_ok=True)
    steps_per_frame = int(round(fd.DT_FRAME / (1e-3 / refine)))
    n_frames = int(round(t_end / fd.DT_FRAME)) + 1
    began = time.time()
    for level, ux, uy in solve(material, refine, [i * steps_per_frame for i in range(n_frames)]):
        i = level // steps_per_frame
        if not (np.isfinite(ux).all() and np.isfinite(uy).all()):
            raise RuntimeError(f"frame {i} is not finite")
        np.save(out / "snapshots_x" / f"snapshot_{i:04d}.npy", uy.astype(np.float32))
        np.save(out / "snapshots_y" / f"snapshot_{i:04d}.npy", ux.astype(np.float32))
        np.save(out / "snapshots_u" / f"snapshot_{i:04d}.npy", np.hypot(ux, uy).astype(np.float32))
    (out / LAYERED_NOTE).write_text(f"six-layer medium, tanh factor {sharpness:g}\n")
    print(f"wrote {n_frames} frames to {out} (six-layer medium, tanh factor {sharpness:g}, refine {refine}, t_end {t_end}, {time.time() - began:.0f} s)")


def remaining(folder: str) -> None:
    """Norm of the displacement left in [-1.5, 1.5]^2, relative to its value at t = 0."""
    n = len(list((Path(folder) / "snapshots_x").glob("snapshot_*.npy")))
    norms = [np.sqrt(sum(np.sum(c**2) for c in dc.stored(folder, i))) for i in range(n)]
    picks = ", ".join(f"t = {i / 100:.2f}: {100 * norms[i] / norms[0]:.1f}%" for i in range(25, n, 25))
    print(f"{Path(folder).name}: norm of the displacement in the domain, relative to t = 0: {picks}")


if __name__ == "__main__":
    mode, args = sys.argv[1], sys.argv[2:]
    if mode == "selftest":
        selftest(int(args[0]) if args else 1)
    elif mode == "reference":
        reference(args[0], args[1], int(args[2]) if len(args) > 2 else 2, float(args[3]) if len(args) > 3 else 1.0,
                  args[4] if len(args) > 4 else "mirror")
    elif mode == "layered":
        layered(float(args[0]), args[1], int(args[2]) if len(args) > 2 else 2, float(args[3]) if len(args) > 3 else 1.0)
    elif mode == "remaining":
        remaining(args[0])
    elif mode == "compare":
        compare(args[0], args[1])
    else:
        raise SystemExit(f"unknown mode {mode!r}; see the module docstring")
