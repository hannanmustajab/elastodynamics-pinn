"""Error of the trained networks against the DEVITO reference: the numbers of Table 3 of the paper.

Metric of the paper (Staub 2024, Eq. 3.11): at each reference time step the relative L2 error of the
displacement field, ||u_ref - u_pinn||_2 / ||u_ref||_2, with the norm taken over both components and
all 512 x 512 grid points; the reported error is the average over all reference time steps of the
training window (frames 0 to 100 for [0, 1], 0 to 200 for [0, 2]; frame i is t = 0.01 i).

Also computed: the magnitude metric the training scripts print ("Average Relative L2 Error": relative
error of the displacement magnitude with a nine-point border of the difference set to zero). Against
the finite-difference reference the runs were monitored with, it reproduces the value at the end of
each training log; against DEVITO it agrees with that value to about 1e-4.

Reference frames: snapshots_x holds the layer-normal component (u_y of the network) and snapshots_y
the layer-parallel component (u_x of the network), indexed [x, y]. For the final checkpoint of a run
the swapped pairing must give the larger error; the script stops otherwise.

This is the scoring script of the paper with the run table read from runs.csv and the networks
and reference frames read from the benchmark folders of this repository. The network construction,
the ansatz and the error computation are unchanged.

Files:
    runs.csv                                                  one row per run
    <benchmark>/code/<run>/global_AF.py                      network classes of the run
    <benchmark>/results/<run>/experiments/<window>/model/checkpoint_epoch_10001.pth
        checkpoint n holds the weights after n - 1 epochs, so 10001 is the network after 10,000 epochs
    <benchmark>/ground_truth/<reference folder>/snapshots_{x,y}/snapshot_%04d.npy   DEVITO reference
    rescored/error_vs_epoch.csv   one row per run: the error after 10,000 epochs (written)
    rescored/error_vs_time.csv    one row per run and reference time step (written)

Usage: python scripts/diagnostics/score_soap_runs.py [--references DIR] [--cpu] [run ...]
    Scores every run of runs.csv, or only the named runs, and then compares the errors with
    figures/data/ and with Table 3 of the paper; exits with status 1 if one differs.
    --references DIR reads the reference folders from DIR instead of the ground_truth folders
    (for example references recomputed with the scripts of scripts/diagnostics/README.md).
"""
import csv
import importlib.util
import math
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
RUN_TABLE = ROOT / "runs.csv"
REFERENCES = None   # None: <benchmark>/ground_truth/; --references DIR: DIR
FOLDERS = {"homogeneous": "devito_div_homogeneous", "two_layer": "devito_div_two_layer", "six_layer": "devito_div_six_layer_t200",
           "marmousi2_s010": "marmousi2_x09000_z1500_w2000_s010_t200_clamp"}
# Reference name -> benchmark folder whose ground_truth/ holds it.
BENCHMARK_OF = {"homogeneous": "homogeneous", "two_layer": "two_layer", "six_layer": "six_layer", "marmousi2_s010": "marmousi2"}
RESULTS = ROOT / "rescored"
OUTPUT = RESULTS / "error_vs_epoch.csv"
TIME_OUTPUT = RESULTS / "error_vs_time.csv"
FIELDS = ["benchmark", "network", "run", "epoch", "error_percent", "magnitude_error_percent", "frames"]
TIME_FIELDS = ["benchmark", "network", "run", "frame", "time", "error_percent", "energy_difference"]
# Errors of the paper, to four decimals (the tables round them to two).
EXPECTED = [ROOT / "figures" / "data" / "error_vs_epoch.csv", ROOT / "figures" / "data" / "marmousi2_error_vs_epoch.csv"]
TOLERANCE = 0.002   # percentage points; CPU, MPS and CUDA differ in the last digits of single precision
FINAL_EPOCH = 10000

N, T1, SIGMA, BORDER, DT_FRAME = 512, 0.2, 0.1, 9, 0.01
PEAK = SIGMA * math.exp(-0.5)


def read_runs() -> Tuple[Dict[str, tuple], Dict[str, Path]]:
    """Run name -> (benchmark, network label, results folder, network class, layers or blocks, width, reference,
    end of window), and run name -> code folder."""
    runs, code = {}, {}
    with RUN_TABLE.open() as handle:
        for row in csv.DictReader(handle):
            run, benchmark_folder = row["run"], ROOT / row["benchmark_folder"]
            label = f"{row['network']} {row['size']}"
            runs[run] = (row["benchmark"], label, benchmark_folder / "results" / run, row["class"], int(row["layers"]),
                         int(row["width"]), row["reference"], float(row["t_max"]))
            code[run] = benchmark_folder / "code" / run
    return runs, code


RUNS, CODE = read_runs()


def checkpoints(run: str) -> Dict[int, Path]:
    """Epoch -> checkpoint file of a run (only the final one is in this repository)."""
    found = {}
    for path in RUNS[run][2].glob("experiments/*/model/checkpoint_epoch_*.pth"):
        epoch = int(re.search(r"checkpoint_epoch_(\d+)\.pth", path.name).group(1)) - 1
        if epoch % 500 == 0:
            found[epoch] = path
    return found


def build(run: str) -> torch.nn.Module:
    """The network of a run, built from the network file of its folder."""
    _, _, _, class_name, layers, width, _, _ = RUNS[run]
    spec = importlib.util.spec_from_file_location("network_" + re.sub(r"\W", "_", run), CODE[run] / "global_AF.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Embedding scale 1 for every run, also the run with s = 2: the scale only sets the initial embedding matrix B,
    # a trained parameter that the checkpoint replaces.
    if class_name == "PirateNet":
        return module.PirateNet(input_dim=3, output_dim=2, num_blocks=layers, hidden_dim=width, s=1.0, act="sine")
    if class_name == "NeuralNet":
        return module.NeuralNet(input_dimension=3, output_dimension=2, n_hidden_layers=layers, neurons=width)
    return module.FourierNeuralNet(input_dimension=3, output_dimension=2, n_hidden_layers=layers, neurons=width,
                                   mapping_size=width // 2, scale=1)


def load_reference(name: str, frames: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reference (u_x, u_y) of the network convention, each of shape (frames, N, N)."""
    base = (ROOT / BENCHMARK_OF[name] / "ground_truth") if REFERENCES is None else REFERENCES

    def stack(component: str) -> torch.Tensor:
        files = [base / FOLDERS[name] / f"snapshots_{component}" / f"snapshot_{i:04d}.npy" for i in range(frames)]
        missing = [f for f in files if not f.exists()]
        if missing:
            raise FileNotFoundError(f"{len(missing)} reference frames missing, first {missing[0]}")
        return torch.from_numpy(np.stack([np.load(f) for f in files]).astype(np.float32)).to(device)
    return stack("y"), stack("x")


def predict(net: torch.nn.Module, points: torch.Tensor, t: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """(u_x, u_y) of the network with the hard initial-condition ansatz on the N x N grid, as in errors()."""
    x, y = points[:, 1], points[:, 2]
    gauss = torch.exp(-(x**2 + y**2) / (2 * SIGMA**2))
    points[:, 0] = t
    with torch.no_grad():
        raw = net(points)
    rise, decay = math.tanh(2.5 * t / T1) ** 2, math.exp(-0.5 * (1.5 * t / T1) ** 2)
    return ((raw[:, 0] * rise + x * gauss / PEAK * decay).reshape(N, N),
            (raw[:, 1] * rise + y * gauss / PEAK * decay).reshape(N, N))


def errors(net: torch.nn.Module, reference: Tuple[torch.Tensor, torch.Tensor], points: torch.Tensor) -> Tuple[float, float, float, list, list]:
    """Averages over the frames of the vector error, the vector error with swapped components and the magnitude
    error with border; then the vector error and the magnitude error of every frame."""
    ref_x, ref_y = reference
    vector, swapped, magnitude = [], [], []
    for i in range(len(ref_x)):
        ux, uy = predict(net, points, i * DT_FRAME)
        rx, ry = ref_x[i], ref_y[i]
        norm = torch.sqrt(torch.sum(rx**2 + ry**2))
        vector.append(float(torch.sqrt(torch.sum((ux - rx) ** 2 + (uy - ry) ** 2)) / norm))
        swapped.append(float(torch.sqrt(torch.sum((ux - ry) ** 2 + (uy - rx) ** 2)) / norm))
        difference = torch.sqrt(ux**2 + uy**2) - torch.sqrt(rx**2 + ry**2)
        difference[:BORDER, :] = 0
        difference[:, :BORDER] = 0
        difference[:, N - BORDER:] = 0
        difference[N - BORDER:, :] = 0
        magnitude.append(float(torch.sqrt(torch.sum(difference**2)) / norm))
    return float(np.mean(vector)), float(np.mean(swapped)), float(np.mean(magnitude)), vector, magnitude


def write_csv(path: Path, fields: List[str], rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def expected_errors() -> Dict[str, float]:
    """Run -> error after 10,000 epochs in percent, as in the source data of the paper's figures."""
    found = {}
    for path in EXPECTED:
        with path.open() as handle:
            found.update({row["run"]: float(row["error_percent"]) for row in csv.DictReader(handle) if int(row["epoch"]) == FINAL_EPOCH})
    return found


def table_errors() -> Dict[str, str]:
    """Run -> error printed in Table 3 of the paper (empty for the run that is only named in the text)."""
    with RUN_TABLE.open() as handle:
        return {row["run"]: row["table_error_percent"] for row in csv.DictReader(handle)}


def compare(scored: Dict[str, float]) -> bool:
    """Print each score next to the paper's values; True if every score matches them."""
    expected, table = expected_errors(), table_errors()
    print(f"\n{'run':48s} {'scored':>8s} {'paper':>8s} {'Table 3':>8s}")
    good = True
    for run, value in scored.items():
        # The table rounds to two decimals; 1.6745 is printed as 1.67, so allow half a unit plus the tolerance.
        ok = abs(value - expected[run]) <= TOLERANCE and (not table[run] or abs(value - float(table[run])) <= 0.005 + TOLERANCE)
        good = good and ok
        print(f"{run:48s} {value:8.4f} {expected[run]:8.4f} {table[run] or '-':>8s}  {'ok' if ok else 'DIFFERENT'}")
    return good


def main() -> None:
    args = sys.argv[1:]
    global REFERENCES
    if "--references" in args:
        position = args.index("--references")
        if position + 1 >= len(args) or args[position + 1].startswith("--"):
            raise SystemExit("--references needs a folder")
        REFERENCES = Path(args[position + 1]).resolve()
        del args[position:position + 2]
    use_cpu = "--cpu" in args
    unknown_flags = [a for a in args if a.startswith("--") and a != "--cpu"]
    if unknown_flags:
        raise SystemExit(f"unknown option {unknown_flags[0]}; see the docstring of this script")
    names = [a for a in args if not a.startswith("--")] or list(RUNS)
    print(f"reference frames from {'the ground_truth folders of the benchmarks' if REFERENCES is None else REFERENCES}")
    unknown = [name for name in names if name not in RUNS]
    if unknown:
        raise SystemExit(f"not in {RUN_TABLE.relative_to(ROOT)}: {', '.join(unknown)}")
    device = torch.device("cpu" if use_cpu else "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    grid = torch.linspace(-1.5, 1.5, N)
    gx, gy = torch.meshgrid(grid, grid, indexing="ij")
    points = torch.stack([torch.zeros(N * N), gx.reshape(-1), gy.reshape(-1)], 1).to(device)
    references = {}  # type: Dict[Tuple[str, int], Tuple[torch.Tensor, torch.Tensor]]
    epoch_rows, time_rows, scored = [], [], {}
    for run in names:
        benchmark, network, _, _, _, _, reference_name, t_end = RUNS[run]
        files = checkpoints(run)
        if FINAL_EPOCH not in files:
            raise SystemExit(f"{run}: no checkpoint after {FINAL_EPOCH} epochs")
        frames = int(round(t_end / DT_FRAME)) + 1
        if (reference_name, frames) not in references:
            references[(reference_name, frames)] = load_reference(reference_name, frames, device)
        net = build(run).to(device)
        net.load_state_dict(torch.load(files[FINAL_EPOCH], map_location="cpu")["model_u_state_dict"])
        net.eval()
        vector, swapped, magnitude, per_frame, per_frame_magnitude = errors(net, references[(reference_name, frames)], points)
        if not swapped > vector:
            raise RuntimeError(f"{run}: the swapped component pairing does not give the larger error ({swapped:.4f}, {vector:.4f})")
        scored[run] = 100 * vector
        epoch_rows.append(dict(benchmark=benchmark, network=network, run=run, epoch=FINAL_EPOCH, error_percent=f"{100 * vector:.4f}",
                               magnitude_error_percent=f"{100 * magnitude:.4f}", frames=frames))
        time_rows += [dict(benchmark=benchmark, network=network, run=run, frame=i, time=f"{i * DT_FRAME:.2f}", error_percent=f"{100 * v:.4f}",
                           energy_difference=f"{m**2:.6g}") for i, (v, m) in enumerate(zip(per_frame, per_frame_magnitude))]
        print(f"{run}: error {100 * vector:8.4f}%  (magnitude metric with border {100 * magnitude:8.4f}%)", flush=True)
        write_csv(OUTPUT, FIELDS, epoch_rows)   # after every run, so that an interrupted call keeps what it has scored
        write_csv(TIME_OUTPUT, TIME_FIELDS, time_rows)
    print(f"wrote {OUTPUT.relative_to(ROOT)} and {TIME_OUTPUT.relative_to(ROOT)}")
    if not compare(scored):
        raise SystemExit("at least one error differs from the paper")
    print("\nall errors agree with the paper")


if __name__ == "__main__":
    main()
