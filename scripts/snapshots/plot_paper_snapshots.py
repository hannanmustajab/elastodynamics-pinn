#!/usr/bin/env python3
"""Wavefield snapshots and centreline traces of a run for the manuscript, drawn on the machine that holds
the run. Networks: PirateNet, Vanilla (NeuralNet) and Fourier-feature (FourierNeuralNet).

Snapshots: same panels as compare_wavefields in the run script (columns prediction, Numerical, Difference; u_x
above u_y; colour range -0.2 to 0.2), with five changes:

    * the layer interfaces are drawn as thin grey lines that fade out where the displacement is
      large, so they read as lying behind the wavefield and do not cover it,
    * a star marks the centre of the initial displacement (the origin),
    * y points up. The run script draws the first y row at the top, which mirrors the medium; with
      the interfaces drawn, the panels have to match the orientation of the velocity-model figure.
    * the prediction column is titled with the network of the run instead of "PINN": PirateNet,
      Vanilla PINN or FF-PINN (author, 2026-10-12). The trace legends use the same name.
    * the difference is drawn up to the edges of the domain (2026-10-12). The run script sets a nine-point
      border of the difference to zero. At late times the largest differences lie on the edges (PirateNet
      3 x 128 with t_max = 2: 0.03 at x = -1.5 and x = 1.5 at t = 2.0), and the figure cited for the
      absorbing boundary has to show them. The check against the log keeps the border of the run script.

Traces: the two plots of compare_wavefields_cross_section, unchanged: u_x along x at y = 0 and
u_y along y at x = 0 (grid index 256), the network against the reference.

Fonts: the run script uses the scienceplots styles science, nature and no-latex. Version 2.1.1 on
the cluster draws them with Computer Modern, version 2.1.0 on the workstation with DejaVu Sans. The script sets
the fonts of version 2.1.1, so the figures of both machines look the same.

Everything is read from the run folder: the network class (global_AF.py), the newest checkpoint,
and from main.py the network size, the time scale t1, the layer values and the reference folders.
In this repository the run folder is <benchmark>/results/<run>; main.py and global_AF.py are then
read from <benchmark>/code/<run> (code_folder).

Checks: before drawing, the script recomputes an error the run script printed and stops if it
differs from the value in the log. This verifies checkpoint, network, ansatz and reference folder.
    * finished run (STAGE_DONE.json exists): "Average Relative L2 Error" at the end of the log,
      magnitude metric with border over all frames of the window;
    * run in progress: "Average RRMSE" of the monitoring step that wrote the newest checkpoint,
      the same metric over the monitoring time steps of get_time_steps.
That metric does not see which reference component is which, so for every plotted time the error
with the two reference components swapped must be the larger one.

Reference: by default the folders named in main.py (the finite-difference reference the run was
monitored with). The manuscript uses the DEVITO reference (devito_check.py, mode "reference"):
pass its folder with --reference, and the panels, traces and printed errors use it. The check
against the log always needs the reference of the run; --check-reference names it when the paths
of main.py do not exist on this machine (a run copied from another machine).

Usage:  python plot_paper_snapshots.py <run folder> <time> [<time> ...] [--reference DIR] [--check-reference DIR]
Output: <run folder>/experiments/full_time_ext_tmax_<T>/plots/paper/                    finished run
        <run folder>/experiments/full_time_ext_tmax_<T>/plots/paper_interim_epoch_<n>/  run in progress
            wavefield_t<time>.png, trace_ux_t<time>.png, trace_uy_t<time>.png
"""
import ast
import importlib.util
import math
import os
import re
import sys
from pathlib import Path
from typing import List, Tuple

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU only: the cards are busy with training runs

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
import scienceplots  # noqa: F401  registers the styles below
import torch

# The style of the run script with the fonts of scienceplots 2.1.1 (see above).
plt.style.use(["science", "nature", "no-latex"])
plt.rcParams.update({"font.size": 8, "font.family": "serif", "font.serif": ["cmr10", "Computer Modern Serif", "DejaVu Serif"],
                     "mathtext.fontset": "cm", "axes.formatter.use_mathtext": True})

N, SIGMA, BORDER, DT_FRAME = 512, 0.1, 9, 0.01
PEAK = SIGMA * math.exp(-0.5)
LOW, HIGH = -1.5, 1.5  # the domain of all three benchmarks
EXTENT = [LOW, HIGH, LOW, HIGH]
# Name of the prediction in panel titles and legends, by network class of the run.
NETWORK_LABELS = {"PirateNet": "PirateNet", "NeuralNet": "Vanilla PINN", "FourierNeuralNet": "FF-PINN"}
Field = Tuple[np.ndarray, np.ndarray]


def code_folder(run: Path) -> Path:
    """Folder with main.py and global_AF.py of a run: the run folder itself, or, in the layout of this repository
    (<benchmark>/results/<run> and <benchmark>/code/<run>), the code folder of the same run."""
    if (run / "main.py").exists():
        return run
    code = run.parent.parent / "code" / run.name
    if not (code / "main.py").exists():
        raise FileNotFoundError(f"no main.py in {run} or {code}")
    return code


def setting(tree: ast.Module, name: str):
    """Value of `name = <literal>` in the run script, which must bind the name exactly once."""
    bindings = [node for node in ast.walk(tree)
                if isinstance(node, ast.Name) and node.id == name and isinstance(node.ctx, ast.Store)]
    assignments = [node for node in ast.walk(tree) if isinstance(node, ast.Assign) and len(node.targets) == 1
                   and isinstance(node.targets[0], ast.Name) and node.targets[0].id == name]
    if len(bindings) != 1 or len(assignments) != 1:
        raise ValueError(f"expected one assignment of {name} in main.py, found {len(bindings)}")
    return ast.literal_eval(assignments[0].value)


def interfaces(tree: ast.Module) -> List[float]:
    """y positions where the material changes; the layers have equal thickness, as in get_material_properties."""
    values = [setting(tree, name) for name in ("lambda_vals", "mu_vals", "rho_vals")]
    if len({len(column) for column in values}) != 1:
        raise ValueError("lambda_vals, mu_vals and rho_vals differ in length")
    layers = list(zip(*values))
    edges = np.linspace(LOW, HIGH, len(layers) + 1)[1:-1]
    return [float(edge) for i, edge in enumerate(edges) if layers[i] != layers[i + 1]]


def load_network(run: Path, window: Path, source: str, tree: ast.Module) -> Tuple[torch.nn.Module, int]:
    """The network of the run with its newest checkpoint, and the number of epochs that checkpoint has seen."""
    spec = importlib.util.spec_from_file_location("run_network", code_folder(run) / "global_AF.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    layers, width = setting(tree, "n_layers"), setting(tree, "n_neurons")
    if "self.network_u = PirateNet(" in source:
        net = module.PirateNet(input_dim=3, output_dim=2, num_blocks=layers, hidden_dim=width,
                               s=setting(tree, "scale"), act=setting(tree, "activation"))
    elif "self.network_u = FourierNeuralNet(" in source:
        net = module.FourierNeuralNet(input_dimension=3, output_dimension=2, n_hidden_layers=layers, neurons=width,
                                      mapping_size=setting(tree, "mapping_size"), scale=setting(tree, "scale"))
    elif "self.network_u = NeuralNet(" in source:
        net = module.NeuralNet(input_dimension=3, output_dimension=2, n_hidden_layers=layers, neurons=width)
    else:
        raise ValueError("main.py builds none of PirateNet, FourierNeuralNet, NeuralNet")
    number = lambda path: int(re.search(r"(\d+)\.pth$", path.name).group(1))
    newest = max(window.glob("model/checkpoint_epoch_*.pth"), key=number)
    net.load_state_dict(torch.load(newest, map_location="cpu")["model_u_state_dict"])
    print(f"network: {type(net).__name__} {layers} x {width}; checkpoint: {newest.relative_to(run)}")
    return net.eval(), number(newest) - 1  # checkpoint n holds the weights after n - 1 epochs


def predict(net: torch.nn.Module, points: torch.Tensor, t: float, t1: float) -> Field:
    """(u_x, u_y) of the PINN at time t on the grid, indexed [x, y]: the ansatz of the run script."""
    x, y = points[:, 1], points[:, 2]
    gauss = torch.exp(-(x**2 + y**2) / (2 * SIGMA**2))
    rise, decay = math.tanh(2.5 * t / t1) ** 2, math.exp(-0.5 * (1.5 * t / t1) ** 2)
    points[:, 0] = t
    with torch.no_grad():
        raw = net(points)
    ux = raw[:, 0] * rise + x * gauss / PEAK * decay
    uy = raw[:, 1] * rise + y * gauss / PEAK * decay
    return ux.reshape(N, N).numpy(), uy.reshape(N, N).numpy()


def reference(folders: Tuple[Path, Path], frame: int) -> Field:
    """Reference (u_x, u_y) of a frame from the two folders given in this order."""
    ux, uy = (np.load(folder / f"snapshot_{frame:04d}.npy").astype(np.float32) for folder in folders)
    return ux, uy


def vector_error(pinn: Field, ref: Field) -> float:
    """Relative L2 error of the displacement field over both components and all grid points."""
    return math.sqrt(sum(np.sum((p - r) ** 2) for p, r in zip(pinn, ref)) / sum(np.sum(r**2) for r in ref))


def clear_border(field: np.ndarray) -> np.ndarray:
    field = field.copy()
    field[:BORDER, :] = 0
    field[:, :BORDER] = 0
    field[:, N - BORDER:] = 0
    field[N - BORDER:, :] = 0
    return field


def monitoring_frames(tmax: float) -> List[int]:
    """Reference frames of the monitoring step: get_time_steps of the run script."""
    times = [t for t in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.8, 0.9, 1.0, 1.2, 1.5, 1.8) if t <= tmax]
    if tmax > 0.9:
        times += [t for t in (tmax - 0.4, tmax - 0.2, tmax - 0.1, tmax) if t > 0.9]
    return sorted({int(round(t / DT_FRAME)) for t in times})


def check_against_log(run: Path, net: torch.nn.Module, folders: Tuple[Path, Path], points: torch.Tensor, t1: float,
                      tmax: float, epoch: int, finished: bool) -> None:
    """An error the run script printed for these weights must be reproduced from the checkpoint."""
    logs = [path.read_text() for path in sorted(run.glob("full_time_ext_*.txt"), key=lambda path: path.stat().st_mtime,
                                                reverse=True)]  # newest first; a resumed job writes a new log
    if finished:
        frames = list(range(int(round(tmax / DT_FRAME)) + 1))
        logged, tolerance = float(re.findall(r"Average Relative L2 Error: ([0-9.eE+-]+)", logs[0])[-1]), 1e-4
    else:
        frames = monitoring_frames(tmax)
        pattern = rf"Monitoring after epoch {epoch}:.*?Average RRMSE: ([0-9.eE+-]+)"
        found = next((match for match in (re.search(pattern, log, re.S) for log in logs) if match), None)
        if found is None:
            raise RuntimeError(f"no log has a monitoring step after epoch {epoch}")
        logged, tolerance = float(found.group(1)), 1.5e-4  # printed with four decimals
    total = 0.0
    for frame in frames:
        ux, uy = predict(net, points, frame * DT_FRAME, t1)
        rx, ry = reference(folders, frame)
        magnitude = np.sqrt(rx**2 + ry**2)
        total += np.linalg.norm(clear_border(np.sqrt(ux**2 + uy**2) - magnitude)) / np.linalg.norm(magnitude)
    value = total / len(frames)
    print(f"check: error over {len(frames)} frames {value:.6f}, log {logged:.6f}")
    if not abs(value - logged) <= tolerance:  # also stops on NaN
        raise RuntimeError("the error of the log is not reproduced; no figure written")


def draw_interfaces(ax: plt.Axes, field: np.ndarray, lines: List[float]) -> None:
    """Interfaces behind the wavefield: opacity 0.45 where the panel is at rest, none where |u| >= 0.04."""
    x = np.linspace(LOW, HIGH, N)
    for y in lines:
        row = int(round((y - LOW) / (HIGH - LOW) * (N - 1)))
        strength = np.abs(field[:, row])
        opacity = 0.45 * (1 - np.clip(0.5 * (strength[:-1] + strength[1:]) / 0.04, 0, 1))
        segments = [[(x[i], y), (x[i + 1], y)] for i in range(N - 1)]
        ax.add_collection(LineCollection(segments, colors=[(0, 0, 0, value) for value in opacity], linewidths=0.4,
                                         zorder=3))


def draw(pinn: Field, ref: Field, lines: List[float], label: str, path: Path) -> None:
    """Six panels: prediction (titled `label`), reference and difference, u_x above u_y."""
    fig, axes = plt.subplots(2, 3, gridspec_kw={"wspace": 0.1, "hspace": 0.1})
    for ax, title in zip(axes[0], [label, "Numerical", "Difference"]):
        ax.set_title(title, fontsize=8)
    for row in range(2):
        panels = (pinn[row], ref[row], pinn[row] - ref[row])
        for ax, field in zip(axes[row], panels):
            image = ax.imshow(field.T, vmin=-0.2, vmax=0.2, cmap="seismic", extent=EXTENT, origin="lower")
            draw_interfaces(ax, field, lines)
    for ax in axes.flatten():
        ax.plot(0, 0, marker="*", markersize=3.5, markerfacecolor="gold", markeredgecolor="black", markeredgewidth=0.3,
                zorder=4)
        ax.set_xlim(LOW, HIGH)
        ax.set_ylim(LOW, HIGH)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(image, cax=fig.add_axes([0.15, 0.05, 0.7, 0.03]), orientation="horizontal")
    fig.savefig(path, dpi=600)
    plt.close(fig)


def draw_trace(pinn: np.ndarray, ref: np.ndarray, component: str, label: str, path: Path) -> None:
    """One centreline trace, as in compare_wavefields_cross_section of the run script."""
    plt.figure(dpi=600)
    axis = np.linspace(LOW, HIGH, num=N)
    plt.plot(axis, pinn, label=label, color="blue", linewidth=1)
    plt.plot(axis, ref, label="Numerical", color="orange", linewidth=1)
    plt.title(f"$u_{component}$", fontsize=8)
    plt.xlabel(f"${component}$", fontsize=8)
    plt.xlim(LOW, HIGH)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.legend(fontsize=8, loc="upper right")
    plt.tight_layout()
    plt.savefig(path, dpi=600)
    plt.close()


def main() -> None:
    args, chosen = sys.argv[1:], {}
    for flag in ("--reference", "--check-reference"):
        if flag in args:
            position = args.index(flag)
            chosen[flag] = Path(args[position + 1])
            del args[position:position + 2]
    run, times = Path(args[0]).resolve(), [float(value) for value in args[1:]]
    torch.set_num_threads(4)
    source = (code_folder(run) / "main.py").read_text()
    tree = ast.parse(source)
    t1 = setting(tree, "t1_value")
    # In the convention of the PINN, snapshots_y holds u_x and snapshots_x holds u_y.
    pair = lambda folder: (folder / "snapshots_y", folder / "snapshots_x")
    check_folders = (pair(chosen["--check-reference"]) if "--check-reference" in chosen
                     else (Path(setting(tree, "input_directory_y")), Path(setting(tree, "input_directory_x"))))
    folders = pair(chosen["--reference"]) if "--reference" in chosen else check_folders
    print(f"reference of the figures: {folders[0].parent}")
    windows = list(run.glob("experiments/full_time_ext_tmax_*"))
    if len(windows) != 1:
        raise ValueError(f"expected one training window in {run}, found {len(windows)}")
    window = windows[0]
    tmax = float(window.name.rsplit("_", 1)[1])
    frames = int(round(tmax / DT_FRAME)) + 1
    for t in times:
        frame = int(round(t / DT_FRAME))
        if abs(frame * DT_FRAME - t) > 1e-9 or not 0 <= frame < frames:
            raise ValueError(f"t = {t} is not a reference time step of the window [0, {tmax}]")
    net, epoch = load_network(run, window, source, tree)
    label = NETWORK_LABELS[type(net).__name__]
    finished = (window / "STAGE_DONE.json").exists()
    grid = torch.linspace(LOW, HIGH, N)
    gx, gy = torch.meshgrid(grid, grid, indexing="ij")
    points = torch.stack([torch.zeros(N * N), gx.reshape(-1), gy.reshape(-1)], dim=1)
    check_against_log(run, net, check_folders, points, t1, tmax, epoch, finished)
    lines = interfaces(tree)
    print(f"interfaces at y = {lines}" if lines else "no interfaces")
    target = window / "plots" / ("paper" if finished else f"paper_interim_epoch_{epoch}")
    print(f"{'finished run' if finished else 'run in progress'}, weights after {epoch} epochs")
    target.mkdir(parents=True, exist_ok=True)
    for t in times:
        pinn, ref = predict(net, points, t, t1), reference(folders, int(round(t / DT_FRAME)))
        error = vector_error(pinn, ref)
        if not vector_error(pinn, ref[::-1]) > error:  # also stops on NaN
            raise RuntimeError(f"t = {t}: the reference components fit better when swapped; no figure written")
        path = target / f"wavefield_t{t:.2f}.png"
        draw(pinn, ref, lines, label, path)
        draw_trace(pinn[0][:, N // 2], ref[0][:, N // 2], "x", label, target / f"trace_ux_t{t:.2f}.png")
        draw_trace(pinn[1][N // 2, :], ref[1][N // 2, :], "y", label, target / f"trace_uy_t{t:.2f}.png")
        print(f"t = {t:.2f}: relative L2 error of the displacement field {100 * error:.2f}%, wrote {path}")


if __name__ == "__main__":
    main()
