"""Make a copy of a run folder that trains again on this machine.

The files in <benchmark>/code/ are kept exactly as they were run, so their main.py reads the monitoring
reference from a path on the machine that trained them. This script copies the training files of one run
into a new folder and changes only the two lines that name that reference:

    input_directory_x = '<reference>/snapshots_x'
    input_directory_y = '<reference>/snapshots_y'

The monitoring reference is the one each run printed its errors against during training (README.md,
"Retraining"); the errors of the paper are computed afterwards against the DEVITO reference by
scripts/diagnostics/score_soap_runs.py. The copy holds no checkpoint, so it starts from a randomly
initialized network (seeds as in the paper). Start it in the new folder with
`python main.py > full_time_ext_<name>.txt`.

Usage: python scripts/prepare_run.py <benchmark>/code/<run> <new folder> <monitoring reference folder>
"""
import re
import shutil
import sys
from pathlib import Path

TRAINING_FILES = ("main.py", "global_AF.py", "soap.py", "marmousi2_crop.npz", "marmousi2_model.py")


def frames_needed(script: str) -> int:
    """The number of reference frames main.py loads at its start (its constant num_time_steps)."""
    found = re.findall(r"^num_time_steps = (\d+)", script, flags=re.MULTILINE)
    if len(found) != 1:
        raise ValueError("main.py does not set num_time_steps exactly once")
    return int(found[0])


def point_to(script: str, reference: Path) -> str:
    """main.py with the two reference lines replaced; each must occur exactly once."""
    for component in "xy":
        pattern = rf"^input_directory_{component} = '[^']*'$"
        if len(re.findall(pattern, script, flags=re.MULTILINE)) != 1:
            raise ValueError(f"main.py does not set input_directory_{component} exactly once")
        line = f"input_directory_{component} = {str(reference / f'snapshots_{component}')!r}"
        script = re.sub(pattern, lambda _: line, script, flags=re.MULTILINE)   # a function: backslashes stay literal
    return script


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    run, target, reference = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]).resolve()
    if not (run / "main.py").exists():
        raise SystemExit(f"{run} has no main.py")
    if target.exists() and any(target.iterdir()):
        raise SystemExit(f"{target} exists and is not empty")
    script = (run / "main.py").read_text()
    required = ["main.py", "global_AF.py", "soap.py"] + (["marmousi2_crop.npz"] if "medium_file = " in script else [])
    absent = [name for name in required if not (run / name).exists()]
    if absent:
        raise SystemExit(f"{run} lacks {', '.join(absent)}")
    frames = frames_needed(script)
    missing = [f"snapshots_{c}/snapshot_{i:04d}.npy" for c in "xy" for i in range(frames)
               if not (reference / f"snapshots_{c}" / f"snapshot_{i:04d}.npy").exists()]
    if missing:
        raise SystemExit(f"{reference} lacks {len(missing)} of the {2 * frames} frames main.py loads, first {missing[0]}")
    target.mkdir(parents=True, exist_ok=True)
    for name in TRAINING_FILES:
        if (run / name).exists():
            shutil.copy2(run / name, target / name)
    (target / "main.py").write_text(point_to(script, reference))
    print(f"copied the training files of {run.name} to {target}; monitoring reference {reference} ({frames} frames)")
    print(f"start with:  cd {target} && python main.py > full_time_ext_{run.name}.txt 2>&1")


if __name__ == "__main__":
    main()
