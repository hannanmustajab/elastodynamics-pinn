"""DEVITO solutions of the benchmark media, to check and to replace the finite-difference reference.

DEVITO solves the elastic wave equation on the grid, time step, material model and initial
displacement of fd_two_forms.solve (the reference solver), with fourth-order differences in space
(the reference solver is second order), in one of two forms:

  "div"       rho u_tt = div(sigma(u)) with lambda(y), mu(y) inside the divergence: the equation of
              the manuscript, of the PINN residual and of the reference. Written with the product
              rule, so the terms with d(lambda)/dy and d(mu)/dy appear explicitly (analytic
              derivatives of the profile).
  "pointwise" the same without those terms: coefficients outside the derivatives, as in the
              author's DEVITO notebooks (Multi-Layered-Model*.ipynb, cell "div_stress_ux = ...").

Frames are at exact times t = 0.01 i on the 512 x 512 evaluation grid over [-1.5, 1.5]^2, indexed [x, y].

Usage:
  python devito_check.py run <div|pointwise> <out.npz> [refine=2] [t_end=2.0]
      four frames of the six-layer medium (t = 0.5, 1.0, 1.5, 2.0)
  python devito_check.py compare <div.npz> <pointwise.npz> <reference folder> <stored DEVITO folder>
  python devito_check.py reference <out_dir> [refine=2] [t_end=2.0] [medium=six_layer]
      all frames in the divergence form, in the layout of make_reference.py:
      snapshots_x (u_y of the PINN), snapshots_y (u_x of the PINN), snapshots_u (magnitude)
  python devito_check.py against <DEVITO folder> <reference folder>
      frame-by-frame difference of two reference folders
"""
import sys
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import numpy as np

import fd_two_forms as fd

TIMES = (0.5, 1.0, 1.5, 2.0)
# As in make_reference.py: values listed from y = -1.5 upward; "six_layer" keeps those of fd_two_forms.
MEDIA = {
    "six_layer": None,
    "homogeneous": dict(LAMBDA=[20.0], MU=[30.0], CENTERS=np.array([])),
    "two_layer": dict(LAMBDA=[25.0, 20.0], MU=[35.0, 30.0], CENTERS=np.array([0.0])),
}  # type: Dict[str, dict]
Frame = Tuple[np.ndarray, np.ndarray]


def profile_derivative(values: List[float], y: np.ndarray) -> np.ndarray:
    """d/dy of fd.layered(values, y): the same recursion, differentiated."""
    out, d_out = np.full_like(y, values[0]), np.zeros_like(y)
    for i, center in enumerate(fd.CENTERS):
        step = 0.5 * (np.tanh(fd.SHARPNESS * (y - center)) + 1.0)
        d_step = 0.5 * fd.SHARPNESS / np.cosh(fd.SHARPNESS * (y - center)) ** 2
        d_out = d_out * (1.0 - step) + (values[i + 1] - out) * d_step
        out = out * (1.0 - step) + values[i + 1] * step
    return d_out


def check_buffer_indexing() -> None:
    """Level L of a second-order TimeFunction is in data[L % 3] after op(time_M=L - 1), also across calls."""
    from devito import Eq, Grid, Operator, TimeFunction
    grid = Grid(shape=(8, 8), extent=(1.0, 1.0))
    u = TimeFunction(name="u", grid=grid, space_order=2, time_order=2)
    u.data[:] = 0.0
    op = Operator([Eq(u.forward, u + 1.0)])   # level n holds n - 1 for n >= 1
    start = 1
    for level in (7, 20, 21, 50):
        op(time_m=start, time_M=level - 1, dt=1.0)
        if not np.allclose(u.data[level % 3], level - 1):
            raise RuntimeError(f"buffer indexing: level {level} is not in slot {level % 3}")
        start = level


def solve(form: str, refine: int, levels: List[int]) -> Iterator[Tuple[int, np.ndarray, np.ndarray]]:
    """(level, u_x, u_y) on the evaluation grid for the requested time levels (level n is t = n dt), in order."""
    from devito import Eq, Function, Grid, Operator, TimeFunction, configuration
    from devito import solve as devito_solve
    configuration["log-level"] = "WARNING"
    check_buffer_indexing()
    fd.PAD_CELLS = 200   # as in make_reference.py: no reflection from the outer edge returns before t = 2.4

    h = 3.0 / (fd.N_EVAL - 1) / refine
    pad = fd.PAD_CELLS * refine
    n = (fd.N_EVAL - 1) * refine + 1 + 2 * pad
    coords = -1.5 + h * (np.arange(n) - pad)
    dt = 1e-3 / refine
    grid = Grid(shape=(n, n), extent=((n - 1) * h, (n - 1) * h), dtype=np.float64)
    print(f"{form}: n = {n}, h = {h:.5f}, dt = {dt}, lambda {fd.LAMBDA}, mu {fd.MU}, interfaces {list(fd.CENTERS)}", flush=True)

    def field(name: str, row: np.ndarray) -> "Function":
        f = Function(name=name, grid=grid, space_order=4)
        f.data[:] = row[None, :]   # constant along x, varying along y
        return f

    lam, mu = field("lam", fd.layered(fd.LAMBDA, coords)), field("mu", fd.layered(fd.MU, coords))
    lam_y, mu_y = field("lam_y", profile_derivative(fd.LAMBDA, coords)), field("mu_y", profile_derivative(fd.MU, coords))

    def divergence(ux, uy):
        div_x = (lam + 2 * mu) * ux.dx2 + mu * ux.dy2 + (lam + mu) * uy.dx.dy
        div_y = (lam + 2 * mu) * uy.dy2 + mu * uy.dx2 + (lam + mu) * ux.dx.dy
        if form == "div":
            div_x = div_x + mu_y * (ux.dy + uy.dx)
            div_y = div_y + lam_y * ux.dx + (lam_y + 2 * mu_y) * uy.dy
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


def run(form: str, out: str, refine: int, t_end: float) -> None:
    dt = 1e-3 / refine
    saved = {}
    for level, ux, uy in solve(form, refine, [int(round(t / dt)) for t in TIMES if t <= t_end + 1e-9]):
        key = int(round(100 * level * dt))
        saved[f"x_{key:04d}"], saved[f"y_{key:04d}"] = ux.astype(np.float32), uy.astype(np.float32)
        print(f"{form}: t = {level * dt:.2f} (level {level}), max |u_x| = {np.abs(ux).max():.4f}", flush=True)
    np.savez_compressed(out, **saved)
    print(f"saved {out}")


def reference(out_dir: str, refine: int, t_end: float, medium: str) -> None:
    """All frames of a medium in the divergence form, in the layout the run scripts read."""
    if MEDIA[medium] is not None:
        fd.LAMBDA, fd.MU, fd.CENTERS = MEDIA[medium]["LAMBDA"], MEDIA[medium]["MU"], MEDIA[medium]["CENTERS"]
    out = Path(out_dir)
    for name in ("snapshots_x", "snapshots_y", "snapshots_u"):
        (out / name).mkdir(parents=True, exist_ok=True)
    steps_per_frame = int(round(fd.DT_FRAME / (1e-3 / refine)))
    n_frames = int(round(t_end / fd.DT_FRAME)) + 1
    for level, ux, uy in solve("div", refine, [i * steps_per_frame for i in range(n_frames)]):
        i = level // steps_per_frame
        np.save(out / "snapshots_x" / f"snapshot_{i:04d}.npy", uy.astype(np.float32))
        np.save(out / "snapshots_y" / f"snapshot_{i:04d}.npy", ux.astype(np.float32))
        np.save(out / "snapshots_u" / f"snapshot_{i:04d}.npy", np.hypot(ux, uy).astype(np.float32))
    print(f"wrote {n_frames} frames to {out} (DEVITO, divergence form, medium {medium}, refine {refine}, t_end {t_end})")


def stored(folder: str, frame: int) -> Frame:
    """(u_x, u_y) of a stored frame: snapshots_y holds u_x and snapshots_x holds u_y, indexed [x, y]."""
    load = lambda part: np.load(Path(folder) / f"snapshots_{part}" / f"snapshot_{frame:04d}.npy").astype(np.float64)
    return load("y"), load("x")


def difference(a: Frame, b: Frame) -> float:
    """Relative L2 difference of two displacement fields, in percent of the second."""
    return 100 * np.sqrt(sum(np.sum((p - q) ** 2) for p, q in zip(a, b)) / sum(np.sum(q**2) for q in b))


def compare(div_file: str, pointwise_file: str, reference_folder: str, devito_folder: str) -> None:
    new: Dict[str, dict] = {"div": dict(np.load(div_file)), "pointwise": dict(np.load(pointwise_file))}
    n_stored = len(list((Path(devito_folder) / "snapshots_x").glob("snapshot_*.npy")))
    print("relative L2 difference of the displacement field, percent")
    print("  t   | DEVITO div vs reference | DEVITO pointwise vs reference | stored DEVITO vs reference | stored DEVITO vs DEVITO pointwise (same index / best index)")
    for t in TIMES:
        key = int(round(100 * t))
        if f"x_{key:04d}" not in new["div"]:
            continue
        frame = {form: (data[f"x_{key:04d}"].astype(np.float64), data[f"y_{key:04d}"].astype(np.float64)) for form, data in new.items()}
        ref = stored(reference_folder, key)
        line = f"  {t:.1f} | {difference(frame['div'], ref):23.2f} | {difference(frame['pointwise'], ref):29.2f} |"
        if key < n_stored:
            old = stored(devito_folder, key)
            near = {i: difference(stored(devito_folder, i), frame["pointwise"]) for i in range(max(key - 2, 0), min(key + 8, n_stored))}
            best = min(near, key=near.get)
            line += f" {difference(old, ref):26.2f} | {near[key]:.2f} / {near[best]:.2f} at index {best}"
        print(line)


def against(devito_folder: str, reference_folder: str) -> None:
    """Difference of every frame of a DEVITO reference folder from the finite-difference reference folder."""
    n = min(len(list((Path(folder) / "snapshots_x").glob("snapshot_*.npy"))) for folder in (devito_folder, reference_folder))
    values = [difference(stored(devito_folder, i), stored(reference_folder, i)) for i in range(n)]
    picks = ", ".join(f"t = {i / 100:.1f}: {values[i]:.3f}" for i in range(0, n, 50))
    print(f"{Path(devito_folder).name} against {Path(reference_folder).name}: {n} frames, mean {np.mean(values):.3f}%, max {np.max(values):.3f}% ({picks})")


if __name__ == "__main__":
    mode, args = sys.argv[1], sys.argv[2:]
    if mode == "run":
        run(args[0], args[1], int(args[2]) if len(args) > 2 else 2, float(args[3]) if len(args) > 3 else 2.0)
    elif mode == "reference":
        reference(args[0], int(args[1]) if len(args) > 1 else 2, float(args[2]) if len(args) > 2 else 2.0, args[3] if len(args) > 3 else "six_layer")
    elif mode == "against":
        against(args[0], args[1])
    else:
        compare(*args[:4])
