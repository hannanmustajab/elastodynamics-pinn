# Adaptive PINNs for elastic wave propagation in heterogeneous media: code, ground truth and results

Code, ground-truth wavefields, training logs and trained networks of the paper

> A. H. Mustajab, F. Wuttke, A. R. Khan and Z. H. Rizvi, *Adaptive Physics-Informed Neural Networks (PINNs) for
> Elastic Wave Propagation in Heterogeneous Media*, submitted to Scientific Reports.

## Layout

One folder per benchmark of the paper. Each has the same three parts:

| Folder | `code/` | `ground_truth/` | `results/` |
|---|---|---|---|
| [`homogeneous/`](homogeneous/) | 3 runs | DEVITO reference, t ≤ 1 | networks, logs, errors, figures |
| [`two_layer/`](two_layer/) | 3 runs | DEVITO reference, t ≤ 1 | networks, logs, errors, figures |
| [`six_layer/`](six_layer/) | 15 runs (t_max = 1 and 2) | DEVITO reference, t ≤ 2 | networks, logs, errors, figures |
| [`marmousi2/`](marmousi2/) | 3 runs | DEVITO reference, t ≤ 1 | networks, logs, errors, figures |

- `code/<run>/`: the training script of the run exactly as it was run (`main.py`: medium, network, sampling, loss,
  SOAP training with residual-based resampling, monitoring), the network classes (`global_AF.py`: Vanilla MLP
  `NeuralNet`, Fourier-feature MLP `FourierNeuralNet`, `PirateNet`), the SOAP optimizer (`soap.py`, unchanged from
  https://github.com/nikhilvyas/SOAP) and the launch script of the machine that trained it (`run.sh`).
- `ground_truth/<reference>/snapshots_{x,y}/snapshot_%04d.npy`: the DEVITO reference the paper scores against; one
  float32 512 x 512 array per frame, frame i at t = 0.01 i, indexed [x, y] on [-1.5, 1.5]^2. `snapshots_x` holds u_y
  and `snapshots_y` holds u_x.
- `results/<run>/`: the training log (`full_time_ext_*.txt`) and the network after 10,000 epochs
  (`experiments/<window>/model/checkpoint_epoch_10001.pth`, key `model_u_state_dict`).
- `results/errors.csv`: the time-averaged relative L2 error of each run after 10,000 epochs, with the value of Table 3;
  `results/error_vs_time.csv`: the error at every reference frame; `results/figures/`: the figures of the paper for
  this benchmark.

Shared files:

| Path | Content |
|---|---|
| `runs.csv` | All 24 runs: benchmark folder, network, size, t_max, GPU, value in Table 3 |
| `scripts/diagnostics/` | Scoring script, DEVITO and finite-difference reference solvers, Marmousi2 medium, reference check ([README](scripts/diagnostics/README.md)) |
| `scripts/*.py` | Figure scripts (scienceplots style); `scripts/snapshots/`: wavefield snapshots; `scripts/prepare_run.py`: copy a run for retraining |
| `figures/data/` | Source data of every plot of the paper; `figures/paper/`: the figures that combine benchmarks |
| `environment/` | Package lists of the two training machines; requirements of the reference solver |

`checkpoint_epoch_<n>.pth` holds the network after n - 1 epochs, so `checkpoint_epoch_10001.pth` is the network after
10,000 epochs, the one all results refer to.

## Check the errors of Table 3

```bash
pip install -r requirements.txt
python scripts/diagnostics/score_soap_runs.py
```

The script evaluates every stored network on the 512 x 512 grid at every frame of its benchmark's ground truth, writes
`rescored/`, prints each error next to the value of the paper and ends with `all errors agree with the paper` (exit
status 0) or names the run that differs. A run agrees if its error is within 0.002 percentage points of the
four-decimal value in `figures/data/` and rounds to the value of Table 3. Before release (2026-10-23; Apple MPS,
PyTorch 2.13) all 24 runs agreed to four decimals. A GPU is used when one is available; a CPU works.

To check the ground truth itself, compute it again with DEVITO (`pip install -r environment/requirements-reference.txt`)
and score against the new frames:

```bash
python scripts/diagnostics/devito_check.py reference references/devito_div_homogeneous 2 1.0 homogeneous
python scripts/diagnostics/devito_check.py reference references/devito_div_two_layer 2 1.0 two_layer
python scripts/diagnostics/devito_check.py reference references/devito_div_six_layer_t200 2 2.0 six_layer
python scripts/diagnostics/devito_marmousi2.py reference figures/data/marmousi2_crop_s010.npz \
    references/marmousi2_x09000_z1500_w2000_s010_t200_clamp 2 1.0 clamp
python scripts/diagnostics/check_reference.py references/devito_div_homogeneous references/devito_div_two_layer \
    references/devito_div_six_layer_t200 references/marmousi2_x09000_z1500_w2000_s010_t200_clamp
python scripts/diagnostics/score_soap_runs.py --references references
```

## Runs

| Table 3 column | Networks | Folder |
|---|---|---|
| Homogeneous | Vanilla 5 x 128, Fourier-feature 5 x 128, PirateNet 2 x 64 | `homogeneous/` |
| Two-layer | Vanilla 5 x 128, Fourier-feature 5 x 128, PirateNet 2 x 64 | `two_layer/` |
| Six-layer, t_max = 1 | Vanilla 5 x 128 and 9 x 128, Fourier-feature 5 x 128 and 9 x 128, PirateNet 2 x 64, 2 x 128, 3 x 128 | `six_layer/` (`*_T1.0_*`) |
| Six-layer, t_max = 2 | Vanilla 9 x 128, Fourier-feature 9 x 128, PirateNet 2 x 64, 3 x 64, 2 x 128, 3 x 128 | `six_layer/` (`*_T2.0_*`) |
| Marmousi2 | Vanilla 9 x 128, Fourier-feature 9 x 128, PirateNet 3 x 128 | `marmousi2/` |

Two runs are discussed in the text but not listed in Table 3: the Vanilla 5 x 128 run with one batch per epoch
(`six_layer/.../SixLayer_Vanilla_5x128_T1.0_soap_rad`, 22.45%; the table's Vanilla 5 x 128 is the two-batch run
`SixLayer_Vanilla_5x128_T1.0_2batch_soap_rad`) and the PirateNet 2 x 64 run with Fourier scale s = 2
(`SixLayer_Pirate_2x64_T2.0_scale2_soap_rad`).

The runs used two machines: a workstation with one NVIDIA RTX 5090 (Python 3.10, PyTorch 2.8.0, CUDA 12.8) and the nesh
cluster of Kiel University with NVIDIA H100 and V100 cards (Python 3.12, PyTorch 2.4.1, CUDA 11.8); `runs.csv` gives
the GPU of each run and `environment/` every package of both machines. The run scripts of the two machines differ
mainly in how they set the number of batches per epoch (a constant in the script, or, in some cluster runs, the fewest
that fit on the card, printed in the log). The settings of each run are printed at the top of its log and summarized
in the settings table of the paper.

`six_layer/.../SixLayer_Pirate_3x128_T2.0_soap_rad` was interrupted on the cluster after 5,450 epochs (cluster
maintenance) and continued on the workstation from its checkpoint after 5,000 epochs. Its folders hold the second job;
`first_job/` holds the script and log of the first. The two scripts differ only in the path of the monitoring reference.

## Figures

Run the scripts from the repository root; they write `figures/<name>.pdf` and `.png`. The versions printed in the
paper are in `<benchmark>/results/figures/` and `figures/paper/`.

| Figure of the paper | Script | Input |
|---|---|---|
| Velocity models | `scripts/make_velocity_models.py` | none |
| Error and training loss against epoch | `scripts/make_convergence_figures.py` | `figures/data/error_vs_epoch.csv`, `loss_vs_epoch.csv` (`--fetch` rebuilds the loss file from the logs in `<benchmark>/results/`) |
| Error against epoch, PirateNet sizes | `scripts/make_size_figures.py` | `figures/data/error_vs_epoch.csv` |
| Error against time | `scripts/make_error_time_figures.py` | `figures/data/error_vs_time.csv` |
| Six-layer traces at t = 0.9 | `scripts/make_trace_figures.py` | networks in `six_layer/results/`, `six_layer/ground_truth/` |
| Marmousi2 medium and errors; Marmousi2 traces | `scripts/make_marmousi2_figures.py` | `figures/data/marmousi2_*`; `--traces` recomputes the traces from the networks and the ground truth |
| Wavefield snapshots | `scripts/snapshots/plot_paper_snapshots.py`, `plot_row_snapshots.py`, `plot_marmousi2_snapshots.py --row` | a results folder, the ground truth (`--reference`) and the monitoring reference of the run (`--check-reference`) |

The error-against-epoch curves were computed from the checkpoints of every 500th epoch, of which only the final one is
stored here; the final value of each curve is the one `score_soap_runs.py` reproduces. The snapshot scripts first
recompute the final error the training log prints and stop if it differs, which checks results folder, code, network
and reference together. That check needs the reference the run was monitored with: for the Marmousi2 runs it is the
ground truth; for the layered media it is the finite-difference reference of the next section. Example:

```bash
python scripts/snapshots/plot_marmousi2_snapshots.py marmousi2/results/Marmousi2_s010_Pirate_3x128_T1.0_soap_rad 0.9 --row \
    --reference marmousi2/ground_truth/marmousi2_x09000_z1500_w2000_s010_t200_clamp \
    --check-reference marmousi2/ground_truth/marmousi2_x09000_z1500_w2000_s010_t200_clamp
```

The two schematic figures (training pipeline and PirateNet architecture) are drawings and not generated from data.

## Retraining

`main.py` of each run is kept as it was run and reads its monitoring reference from a path on the machine that trained
it. To train a run again, compute that reference with the finite-difference solver (numpy only, minutes per medium) and
make a copy of the run that points to it:

```bash
python scripts/diagnostics/make_reference.py references/reference_homogeneous 2 1.0 homogeneous
python scripts/diagnostics/make_reference.py references/reference_two_layer 2 1.0 two_layer
python scripts/diagnostics/make_reference.py references/multi_layered_fd_div_reference_t240 2 2.4 six_layer
python scripts/prepare_run.py homogeneous/code/Homogeneous_Pirate_2x64_soap_rad retrain/pirate references/reference_homogeneous
cd retrain/pirate && python main.py > full_time_ext_retrain.txt 2>&1
```

The Marmousi2 runs are monitored against their ground truth. The monitoring reference only affects the errors printed
during training, not the training itself. Training takes from about half an hour (two-layer medium, H100) to several
hours per run on the GPUs of `runs.csv` (32 GB or more); the peak GPU memory of each run is printed in its log. GPU
arithmetic is not bitwise deterministic, so a retrained network differs slightly from the stored one; the stored
networks are the ones the paper reports.

## Licence

Code: MIT (`LICENSE`); `soap.py` is copyright 2024 Nikhil Vyas, also MIT (`LICENSES/SOAP-MIT.txt`). Trained networks,
training logs, ground-truth wavefields and figure data: CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). The
Marmousi2 medium and its ground truth derive from the Marmousi2 model of the Allied Geophysical Laboratory, University
of Houston (Martin, Wiley and Marfurt 2006, CC BY 4.0).

## Citation

Please cite the paper and this repository (`CITATION.cff`). The archived version will have a DOI on Zenodo.
