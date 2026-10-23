"""Check that regenerated reference wavefields match the ones the paper was scored against.

The reference frames of the paper are in <benchmark>/ground_truth/; scripts/diagnostics/README.md gives the
commands that compute them again. reference_fingerprints.csv holds, for every frame of the four
references, two numbers per component: the L2 norm and the sum of the 512 x 512 values times a fixed
weight pattern without symmetry (WEIGHT), so that a field that differs anywhere changes at least one
of them in general. "ok" means a largest difference below 1e-5 of the frame norm. Rounding
differences between machines should stay far below it (the finite-difference reference computed on
two machines differed by at most 1.5e-8; DEVITO was not compared across machines), while the
second-order finite-difference solution of make_reference.py (a different solver) gives 1.3e-4 for the
homogeneous medium. The fingerprints are a quick check only; the actual test is that
score_soap_runs.py reproduces the errors of the paper with the regenerated reference.

Usage:
    python scripts/diagnostics/check_reference.py <reference folder> [<reference folder> ...]
        compare each folder (named as in score_soap_runs.FOLDERS) with the stored fingerprints
    python scripts/diagnostics/check_reference.py --write <reference folder> [...]
        write the fingerprints of the given folders (how the stored file was made)
"""
import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

FINGERPRINTS = Path(__file__).resolve().parent / "reference_fingerprints.csv"
FIELDS = ["reference", "frame", "norm_x", "weighted_x", "norm_y", "weighted_y"]
TOLERANCE = 1e-5   # relative to the norm of the frame
_I, _J = np.meshgrid(np.arange(512.0), np.arange(512.0), indexing="ij")
WEIGHT = np.cos(0.1234 * _I) * np.cos(0.0567 * _J + 0.3) + np.sin(0.0891 * (_I + 2.0 * _J))
WEIGHT /= np.linalg.norm(WEIGHT)   # unit norm: a change of the field by d changes the weighted sum by at most d


def fingerprint(folder: Path) -> List[dict]:
    """One row per frame: norm and weighted sum of snapshots_x and snapshots_y, in double precision."""
    frames = sorted((folder / "snapshots_x").glob("snapshot_*.npy"))
    if not frames:
        raise FileNotFoundError(f"no frames in {folder / 'snapshots_x'}")
    rows = []
    for path in frames:
        x = np.load(path).astype(np.float64)
        y = np.load(folder / "snapshots_y" / path.name).astype(np.float64)
        rows.append(dict(reference=folder.name, frame=int(path.stem.split("_")[1]), norm_x=f"{np.linalg.norm(x):.10e}",
                         weighted_x=f"{np.sum(x * WEIGHT):.10e}", norm_y=f"{np.linalg.norm(y):.10e}", weighted_y=f"{np.sum(y * WEIGHT):.10e}"))
    return rows


def stored() -> Dict[Tuple[str, int], dict]:
    with FINGERPRINTS.open() as handle:
        return {(row["reference"], int(row["frame"])): row for row in csv.DictReader(handle)}


def check(folder: Path, known: Dict[Tuple[str, int], dict]) -> bool:
    """Print the largest relative difference of the folder from the stored fingerprints; True if within TOLERANCE."""
    worst, compared = 0.0, 0
    for row in fingerprint(folder):
        key = (row["reference"], row["frame"])
        if key not in known:
            continue   # a longer reference than the stored one; the extra frames are not used by the paper
        old = known[key]
        scale = max(float(old["norm_x"]), float(old["norm_y"]), 1e-30)
        for name in ("norm_x", "weighted_x", "norm_y", "weighted_y"):
            worst = max(worst, abs(float(row[name]) - float(old[name])) / scale)
        compared += 1
    expected = sum(1 for name, _ in known if name == folder.name)
    if expected == 0:
        print(f"{folder.name}: no stored fingerprints for a folder of this name (names as in score_soap_runs.FOLDERS)")
        return False
    if compared < expected:
        print(f"{folder.name}: only {compared} of the {expected} frames the paper uses")
        return False
    print(f"{folder.name}: {compared} frames, largest relative difference {worst:.2e} ({'ok' if worst < TOLERANCE else 'DIFFERENT'})")
    return worst < TOLERANCE


def main() -> None:
    args = sys.argv[1:]
    if args and args[0] == "--write":
        rows = [row for folder in args[1:] for row in fingerprint(Path(folder))]
        with FINGERPRINTS.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {FINGERPRINTS.name} ({len(rows)} frames)")
        return
    if not args:
        raise SystemExit(__doc__)
    known = stored()
    results = []
    for folder in args:
        try:
            results.append(check(Path(folder), known))
        except FileNotFoundError as error:
            print(f"{Path(folder).name}: {error}")
            results.append(False)
    if not all(results):
        raise SystemExit("at least one reference differs from the one of the paper")


if __name__ == "__main__":
    main()
