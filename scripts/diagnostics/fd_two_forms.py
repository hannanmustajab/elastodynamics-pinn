"""Finite-difference solution of the six-layer benchmark for two forms of the elastic wave equation.

  "div"   : rho u_tt = div( sigma(u) ), with lambda(y), mu(y) inside the divergence
            (the equation written in the manuscript).
  "local" : rho u_tt = C(y) : grad grad u, i.e. lambda and mu treated as constants when the
            stress is differentiated. This is what compute_pde_residual() in main.py evaluates,
            because lambda_values / mu_values are precomputed tensors outside the autograd graph.

Both use the same grid, time step, initial condition and material model as the PINN runs.
Frames are saved on the 512 x 512 evaluation grid over [-1.5, 1.5]^2.

Usage: python fd_two_forms.py <out.npz> [refine=1] [t_end=1.5]
"""
import sys
import time

import numpy as np

LAMBDA = [20.0, 30.0, 40.0, 50.0, 60.0, 70.0]
MU = [10.0, 15.0, 25.0, 35.0, 45.0, 55.0]
RHO = 100.0
SHARPNESS = 90.0
CENTERS = np.linspace(-1.5, 1.5, 7)[1:-1]
SIGMA = 0.1
N_EVAL = 512
PAD_CELLS = 150  # coarse cells of padding on each side, so no wave reaches the outer boundary
FRAMES = [0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]
DT_FRAME = 0.01


def layered(values: list, y: np.ndarray) -> np.ndarray:
    """Material profile as built by SeismicSimulation.get_material_properties."""
    out = np.full_like(y, values[0])
    for i, c in enumerate(CENTERS):
        step = 0.5 * (np.tanh(SHARPNESS * (y - c)) + 1.0)
        out = out * (1.0 - step) + values[i + 1] * step
    return out


def initial_displacement(x: np.ndarray, y: np.ndarray) -> tuple:
    """Gradient of a Gaussian, each component scaled to unit maximum (initial_condition_gaussian)."""
    g = np.exp(-(x[:, None] ** 2 + y[None, :] ** 2) / (2 * SIGMA**2))
    peak = SIGMA * np.exp(-0.5)
    return x[:, None] * g / peak, y[None, :] * g / peak


def make_rhs(form: str, lam: np.ndarray, mu: np.ndarray, h: float):
    a = lam + 2.0 * mu
    lam_c, mu_c, a_c = lam[None, 1:-1], mu[None, 1:-1], a[None, 1:-1]
    mu_up, mu_dn = 0.5 * (mu[1:-1] + mu[2:])[None, :], 0.5 * (mu[1:-1] + mu[:-2])[None, :]
    a_up, a_dn = 0.5 * (a[1:-1] + a[2:])[None, :], 0.5 * (a[1:-1] + a[:-2])[None, :]
    lam_n, lam_s = lam[None, 2:], lam[None, :-2]
    mu_n, mu_s = mu[None, 2:], mu[None, :-2]
    inv_h2 = 1.0 / h**2

    def rhs(ux: np.ndarray, uy: np.ndarray) -> tuple:
        c = (slice(1, -1), slice(1, -1))
        dxx_ux = ux[2:, 1:-1] - 2 * ux[c] + ux[:-2, 1:-1]
        dxx_uy = uy[2:, 1:-1] - 2 * uy[c] + uy[:-2, 1:-1]
        # centered x-differences on the rows above (north) and below (south), times 2h
        dx_ux_n, dx_ux_s = ux[2:, 2:] - ux[:-2, 2:], ux[2:, :-2] - ux[:-2, :-2]
        dx_uy_n, dx_uy_s = uy[2:, 2:] - uy[:-2, 2:], uy[2:, :-2] - uy[:-2, :-2]
        if form == "div":
            fx = (a_c * dxx_ux + 0.25 * lam_c * (dx_uy_n - dx_uy_s)
                  + mu_up * (ux[1:-1, 2:] - ux[c]) - mu_dn * (ux[c] - ux[1:-1, :-2])
                  + 0.25 * (mu_n * dx_uy_n - mu_s * dx_uy_s))
            fy = (0.25 * mu_c * (dx_ux_n - dx_ux_s) + mu_c * dxx_uy
                  + 0.25 * (lam_n * dx_ux_n - lam_s * dx_ux_s)
                  + a_up * (uy[1:-1, 2:] - uy[c]) - a_dn * (uy[c] - uy[1:-1, :-2]))
        else:
            dyy_ux = ux[1:-1, 2:] - 2 * ux[c] + ux[1:-1, :-2]
            dyy_uy = uy[1:-1, 2:] - 2 * uy[c] + uy[1:-1, :-2]
            fx = a_c * dxx_ux + 0.25 * (lam_c + mu_c) * (dx_uy_n - dx_uy_s) + mu_c * dyy_ux
            fy = 0.25 * (lam_c + mu_c) * (dx_ux_n - dx_ux_s) + mu_c * dxx_uy + a_c * dyy_uy
        return fx * inv_h2 / RHO, fy * inv_h2 / RHO

    return rhs


def solve(form: str, refine: int, t_end: float) -> dict:
    h = 3.0 / (N_EVAL - 1) / refine
    pad = PAD_CELLS * refine
    n = (N_EVAL - 1) * refine + 1 + 2 * pad
    x = -1.5 + h * (np.arange(n) - pad)
    lam, mu = layered(LAMBDA, x), layered(MU, x)
    dt = 1e-3 / refine
    steps_per_frame = int(round(DT_FRAME / dt))
    rhs = make_rhs(form, lam, mu, h)
    sub = slice(pad, n - pad, refine)

    ux, uy = initial_displacement(x, x)
    fx, fy = rhs(ux, uy)
    ux_old, uy_old = ux.copy(), uy.copy()
    # first step with zero initial velocity
    ux_new, uy_new = ux.copy(), uy.copy()
    ux_new[1:-1, 1:-1] += 0.5 * dt**2 * fx
    uy_new[1:-1, 1:-1] += 0.5 * dt**2 * fy

    frames = {0: (ux[sub, sub].copy(), uy[sub, sub].copy())}
    n_steps = int(round(t_end / dt))
    ux, uy = ux_new, uy_new
    start = time.time()
    for step in range(1, n_steps + 1):
        if step % steps_per_frame == 0 and step // steps_per_frame in FRAMES:
            frames[step // steps_per_frame] = (ux[sub, sub].copy(), uy[sub, sub].copy())
        if step == n_steps:
            break
        fx, fy = rhs(ux, uy)
        ux_old[1:-1, 1:-1] = 2 * ux[1:-1, 1:-1] - ux_old[1:-1, 1:-1] + dt**2 * fx
        uy_old[1:-1, 1:-1] = 2 * uy[1:-1, 1:-1] - uy_old[1:-1, 1:-1] + dt**2 * fy
        ux, ux_old = ux_old, ux
        uy, uy_old = uy_old, uy
    print(f"{form}: n={n}, h={h:.5f}, dt={dt}, {n_steps} steps, {time.time() - start:.0f} s", flush=True)
    return frames


def main() -> None:
    out = sys.argv[1]
    refine = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    t_end = float(sys.argv[3]) if len(sys.argv) > 3 else 1.5
    saved = {}
    for form in ("div", "local"):
        for idx, (fx, fy) in solve(form, refine, t_end).items():
            saved[f"{form}_x_{idx:04d}"] = fx.astype(np.float32)
            saved[f"{form}_y_{idx:04d}"] = fy.astype(np.float32)
    np.savez_compressed(out, **saved)
    print("saved", out)


if __name__ == "__main__":
    main()
