"""Crop of the elastic Marmousi2 model as a cosine series of the Lame parameters.

The PINN and the DEVITO reference need lambda(x, y) and mu(x, y) as one differentiable formula.
This module reads the Marmousi2 files (Martin, Wiley and Marfurt 2006; CC BY 4.0), cuts a square
window, converts it to the dimensionless Lame parameters of the manuscript and writes the
coefficients of a two-dimensional cosine series,

    f(x, y) = sum_ij C[i, j] cos(i pi (x - X_MIN) / LENGTH) cos(j pi (y - X_MIN) / LENGTH),

low-pass filtered by a Gaussian of standard deviation sigma_f (factor exp(-k^2 sigma_f^2 / 2) per
direction, k = i pi / LENGTH) and truncated where that factor falls below TRUNCATION. The truncated
series is the medium: the run script and the reference both evaluate it, so they cannot differ.
Outside the window the series continues the model by mirroring, which defines it on the padded
domain of the reference.

Scaling: the window of size_km maps to the domain [-1.5, 1.5]^2, so L0 = size_km / 3; V0 = 3 km/s
as in the example of Section 2.1 of the manuscript; density constant, rho = 100. sigma_f is in the
dimensionless units of the domain. Arrays are indexed [x, y]; y increases upward, so y = +1.5 is
the top of the window.

Usage:
  python marmousi2_model.py inspect <vp.segy.gz> <vs.segy.gz> <density.segy.gz> <figure.png> [x_left_km top_km size_km]
      value ranges, a table of candidate windows and a figure of the model; the given window is drawn on it
  python marmousi2_model.py build <vp.segy.gz> <vs.segy.gz> <x_left_km> <top_km> <size_km> <out_dir> <sigma_f> [<sigma_f> ...]
      one coefficient file per filter width
  python marmousi2_model.py check <vp.segy.gz> <vs.segy.gz> <x_left_km> <top_km> <size_km> <sigma_f>
      the series against a direct Gaussian filter of the window, and its derivatives against differences
"""
import gzip
import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.fft import dctn
from scipy.ndimage import gaussian_filter

SPACING_KM = 0.00125         # grid spacing of the Marmousi2 files
MODEL_SHAPE = (13601, 2801)  # traces (x) and samples (depth)
V0_KMS = 3.0                 # velocity scale of the manuscript's example
RHO = 100.0                  # dimensionless density of all benchmarks (2500 kg/m3 for rho_0 = 25 kg/m3)
X_MIN, LENGTH = -1.5, 3.0    # the computational domain [-1.5, 1.5]^2
TRUNCATION = 1e-4            # series terms whose filter factor is below this are dropped
VS_LIMIT_KMS = 0.9           # the lowest V_s of the six-layer medium is 0.95 km/s; windows are reported against this
CANDIDATES = ((3.0, 0.5), (2.5, 1.0), (2.0, 1.5))   # (size_km, top_km) of the windows listed by inspect()
MEDIUM_NOTE = "medium.sha256"   # in a reference folder: checksum and name of the coefficient file it was computed from


@dataclass(frozen=True)
class Window:
    """Square window of the model: left edge, depth of its top and side length, in km."""
    x_left_km: float
    top_km: float
    size_km: float

    def __post_init__(self) -> None:
        """Only windows on the grid of the model: a rounded or wrapped window would not be the one named."""
        steps = [value / SPACING_KM for value in (self.x_left_km, self.top_km, self.size_km)]
        if min(steps) < 0 or self.size_km <= 0 or any(abs(s - round(s)) > 1e-6 for s in steps):
            raise ValueError(f"{self}: left edge and top must be non-negative, the size positive, all multiples of {SPACING_KM} km")

    @property
    def samples(self) -> int:
        return int(round(self.size_km / SPACING_KM)) + 1

    @property
    def tag(self) -> str:
        return f"x{int(round(1000 * self.x_left_km)):05d}_z{int(round(1000 * self.top_km)):04d}_w{int(round(1000 * self.size_km)):04d}"


def ibm_to_float(raw: np.ndarray) -> np.ndarray:
    """IBM System/360 single-precision numbers (given as uint32) as float64."""
    sign = np.where(raw >> 31, -1.0, 1.0)
    exponent = ((raw >> 24) & 0x7F).astype(np.int64) - 64
    return sign * (raw & 0x00FFFFFF) / 2.0**24 * 16.0**exponent


def read_segy(path: str) -> np.ndarray:
    """Samples of a Marmousi2 SEG-Y file as float32, indexed [trace, sample], that is [x, depth]."""
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rb") as handle:
        raw = handle.read()
    n_samples = int.from_bytes(raw[3220:3222], "big")
    code = int.from_bytes(raw[3224:3226], "big")
    words = 60 + n_samples   # a trace is a 240-byte header and n_samples 4-byte numbers
    n_traces, rest = divmod(len(raw) - 3600, 4 * words)
    if rest or (n_traces, n_samples) != MODEL_SHAPE:
        raise ValueError(f"{path}: {n_traces} traces of {n_samples} samples, expected {MODEL_SHAPE}")
    body = np.frombuffer(raw, dtype=">u4", offset=3600).reshape(n_traces, words)[:, 60:]
    if code == 1:
        return ibm_to_float(body).astype(np.float32)
    if code == 5:
        return body.view(">f4").astype(np.float32)
    raise ValueError(f"{path}: sample format code {code} is not handled (1 = IBM, 5 = IEEE)")


def sha256(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def kms(values: np.ndarray) -> np.ndarray:
    """Wave speeds in km/s; the files hold km/s, a file in m/s is converted."""
    return values / 1000.0 if values.max() > 100.0 else values


def crop(field: np.ndarray, window: Window) -> np.ndarray:
    """The window of a model field, indexed [x, y] with y upward."""
    ix, iz, n = int(round(window.x_left_km / SPACING_KM)), int(round(window.top_km / SPACING_KM)), window.samples
    part = field[ix:ix + n, iz:iz + n]
    if part.shape != (n, n):
        raise ValueError(f"{window} does not fit into the model")
    return part[:, ::-1].astype(np.float64)


def lame(vp_kms: np.ndarray, vs_kms: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Dimensionless (lambda, mu) for wave speeds in km/s and the constant density RHO."""
    mu = RHO * (vs_kms / V0_KMS) ** 2
    return RHO * (vp_kms / V0_KMS) ** 2 - 2.0 * mu, mu


def window_fields(vp_file: str, vs_file: str, window: Window) -> Tuple[np.ndarray, np.ndarray]:
    """Unfiltered (lambda, mu) of the window on the grid of the model files."""
    return lame(crop(kms(read_segy(vp_file)), window), crop(kms(read_segy(vs_file)), window))


def cosine_coefficients(grid: np.ndarray) -> np.ndarray:
    """C with grid[a, b] = sum_ij C[i, j] cos(pi i a / n) cos(pi j b / n), n = len(grid) - 1 (DCT-I)."""
    n = grid.shape[0] - 1
    weight = np.full(n + 1, 2.0)
    weight[[0, n]] = 1.0
    return dctn(grid, type=1) / (2.0 * n) ** 2 * weight[:, None] * weight[None, :]


def wavenumbers(order: int) -> np.ndarray:
    return np.pi * np.arange(order) / LENGTH


def filtered(coeff: np.ndarray, sigma_f: float) -> np.ndarray:
    """Coefficients of the Gaussian-filtered series, truncated where the filter factor is below TRUNCATION."""
    factor = np.exp(-0.5 * (wavenumbers(coeff.shape[0]) * sigma_f) ** 2)
    order = int(np.argmax(factor < TRUNCATION)) if factor[-1] < TRUNCATION else len(factor)
    return coeff[:order, :order] * factor[:order, None] * factor[None, :order]


def evaluate(coeff: np.ndarray, x: np.ndarray, y: np.ndarray, dx: bool = False, dy: bool = False) -> np.ndarray:
    """The series, or its first derivative in x or in y, on the grid x[:, None], y[None, :]."""
    k = wavenumbers(coeff.shape[0])

    def basis(points: np.ndarray, derivative: bool) -> np.ndarray:
        phase = np.outer(points - X_MIN, k)
        return -k * np.sin(phase) if derivative else np.cos(phase)

    return basis(x, dx) @ coeff @ basis(y, dy).T


def load(path: str) -> Dict[str, np.ndarray]:
    """Coefficient file written by build(); refuses a file made for another domain or density."""
    data = dict(np.load(path))
    if (float(data["x_min"]), float(data["length"]), float(data["rho"])) != (X_MIN, LENGTH, RHO):
        raise ValueError(f"{path} was written for another domain or density")
    return data


def build(vp_file: str, vs_file: str, window: Window, out_dir: str, widths: List[float]) -> None:
    lam, mu = window_fields(vp_file, vs_file, window)
    lam_c, mu_c = cosine_coefficients(lam), cosine_coefficients(mu)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    grid = np.linspace(X_MIN, X_MIN + LENGTH, 601)
    for sigma_f in widths:
        out = Path(out_dir) / f"marmousi2_{window.tag}_s{int(round(1000 * sigma_f)):03d}.npz"
        lam_f, mu_f = filtered(lam_c, sigma_f), filtered(mu_c, sigma_f)
        np.savez(out, lam=lam_f, mu=mu_f, x_min=X_MIN, length=LENGTH, rho=RHO, sigma_f=sigma_f,
                 x_left_km=window.x_left_km, top_km=window.top_km, size_km=window.size_km, v0_kms=V0_KMS,
                 source_sha256=np.array([sha256(vp_file), sha256(vs_file)]))
        lam_g, mu_g = evaluate(lam_f, grid, grid), evaluate(mu_f, grid, grid)
        vp, vs = np.sqrt((lam_g + 2 * mu_g) / RHO), np.sqrt(mu_g / RHO)
        print(f"{out.name}: {lam_f.shape[0]} terms per direction; lambda {lam_g.min():.2f} to {lam_g.max():.2f}, "
              f"mu {mu_g.min():.2f} to {mu_g.max():.2f}; V_p {vp.min():.3f} to {vp.max():.3f} "
              f"({V0_KMS * vp.min():.2f} to {V0_KMS * vp.max():.2f} km/s), V_s {vs.min():.3f} to {vs.max():.3f} "
              f"({V0_KMS * vs.min():.2f} to {V0_KMS * vs.max():.2f} km/s)")


def check(vp_file: str, vs_file: str, window: Window, sigma_f: float) -> None:
    """The series against a direct Gaussian filter of the window, and its derivatives against differences."""
    lam, mu = window_fields(vp_file, vs_file, window)
    nodes = np.linspace(X_MIN, X_MIN + LENGTH, window.samples)
    step = 1e-5
    fine = np.concatenate([p + np.array([-step, 0.0, step]) for p in np.linspace(X_MIN + 0.2, X_MIN + LENGTH - 0.2, 60)])
    for name, grid in (("lambda", lam), ("mu", mu)):
        coeff = filtered(cosine_coefficients(grid), sigma_f)
        direct = gaussian_filter(grid, sigma_f / (LENGTH / (window.samples - 1)), mode="mirror", truncate=6.0)
        series = evaluate(coeff, nodes[::8], nodes[::8])
        values = evaluate(coeff, fine, fine)
        d_x = (values[2::3, 1::3] - values[0::3, 1::3]) / (2 * step)
        d_y = (values[1::3, 2::3] - values[1::3, 0::3]) / (2 * step)
        exact_x = evaluate(coeff, fine[1::3], fine[1::3], dx=True)
        exact_y = evaluate(coeff, fine[1::3], fine[1::3], dy=True)
        print(f"{name}: {coeff.shape[0]} terms per direction; series against direct filter, largest difference "
              f"{np.abs(series - direct[::8, ::8]).max():.2e} (values up to {direct.max():.1f}); derivative against "
              f"differences, largest difference in x {np.abs(d_x - exact_x).max():.2e}, in y {np.abs(d_y - exact_y).max():.2e} "
              f"(derivatives up to {np.abs(exact_y).max():.0f}); minimum of the series {series.min():.3f}")


def inspect(vp_file: str, vs_file: str, rho_file: str, figure: str, marked: List[Window]) -> None:
    """Value ranges, a table of candidate windows, and a figure of the model with the marked windows."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    vp, vs, rho = kms(read_segy(vp_file)), kms(read_segy(vs_file)), read_segy(rho_file)
    for name, field, unit in (("V_p", vp, "km/s"), ("V_s", vs, "km/s"), ("density", rho, "g/cm3")):
        print(f"{name}: {field.min():.3f} to {field.max():.3f} {unit}")
    print("\nR is the vertical P reflection coefficient between samples 25 m apart; the lateral variation is the standard\n"
          "deviation of V_p along x, averaged over depth, relative to the mean V_p")
    for size, top in CANDIDATES:
        print(f"\nwindows of {size} km from {top} to {top + size} km depth\n"
              f" x_left | V_p km/s     | V_s km/s     | V_p/V_s      | V_s < {VS_LIMIT_KMS} km/s | max |R| | lateral variation")
        for x_left in np.arange(0.0, SPACING_KM * (MODEL_SHAPE[0] - 1) - size + 1e-9, 1.0):
            p, s = crop(vp, Window(x_left, top, size)), crop(vs, Window(x_left, top, size))
            refl = np.abs((p[:, 20:] - p[:, :-20]) / (p[:, 20:] + p[:, :-20]))
            print(f" {x_left:6.1f} | {p.min():.2f} to {p.max():.2f} | {s.min():.2f} to {s.max():.2f} | {(p / s).min():.2f} to {(p / s).max():.2f} "
                  f"| {100 * np.mean(s < VS_LIMIT_KMS):13.1f}% | {refl.max():7.2f} | {100 * np.std(p, axis=0).mean() / p.mean():.1f}%")

    width_km, depth_km = SPACING_KM * (MODEL_SHAPE[0] - 1), SPACING_KM * (MODEL_SHAPE[1] - 1)
    fig, axes = plt.subplots(3, 1, figsize=(10, 7.0), sharex=True)
    for ax, field, label in zip(axes, (vp, vs, rho), (r"$V_p$ (km/s)", r"$V_s$ (km/s)", r"density (g/cm$^3$)")):
        image = ax.imshow(field[::8, ::8].T, extent=(0.0, width_km, depth_km, 0.0), aspect="equal", cmap="viridis")
        fig.colorbar(image, ax=ax, label=label, pad=0.01)
        ax.set_ylabel("depth (km)")
        for window in marked:
            ax.add_patch(Rectangle((window.x_left_km, window.top_km), window.size_km, window.size_km, fill=False, edgecolor="white", linewidth=1.2))
    axes[-1].set_xlabel("x (km)")
    fig.savefig(figure, dpi=200, bbox_inches="tight")
    print(f"\nfigure written to {figure}")


if __name__ == "__main__":
    mode, args = sys.argv[1], sys.argv[2:]
    if mode == "inspect":
        numbers = [float(a) for a in args[4:]]
        inspect(*args[:4], marked=[Window(*numbers[i:i + 3]) for i in range(0, len(numbers), 3)])
    elif mode == "build":
        build(args[0], args[1], Window(*(float(a) for a in args[2:5])), args[5], [float(a) for a in args[6:]])
    elif mode == "check":
        check(args[0], args[1], Window(*(float(a) for a in args[2:5])), float(args[5]))
    else:
        raise SystemExit(f"unknown mode {mode!r}; see the module docstring")
