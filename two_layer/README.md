# Two-layer medium

λ = 25, μ = 35 below y = 0 and λ = 20, μ = 30 above, ρ = 100; window t ∈ [0, 1].

| Folder | Content |
|---|---|
| `code/<run>/` | `main.py` (training script as run), `global_AF.py` (networks), `soap.py` (optimizer), `run.sh` |
| `ground_truth/devito_div_two_layer/` | DEVITO reference: `snapshots_x` (u_y) and `snapshots_y` (u_x), 101 frames, t = 0 to 1, 512 x 512 float32 |
| `results/<run>/` | training log and the network after 10,000 epochs (`experiments/*/model/checkpoint_epoch_10001.pth`) |
| `results/errors.csv` | time-averaged relative L2 error of each run (%), with the value of Table 3 |
| `results/error_vs_time.csv` | the error at every reference frame |
| `results/figures/` | the figures of the paper for this benchmark |

| Run | Network | t_max | GPU | Error (%) | Table 3 |
|---|---|---|---|---|---|
| `TwoLayer_Vanilla_5x128_T1.0_soap_rad` | Vanilla 5x128 | 1 | H100 (nesh cluster) | 5.7778 | 5.78 |
| `TwoLayer_FourierXB_5x128_T1.0_soap_rad` | Fourier-feature 5x128 | 1 | H100 (nesh cluster) | 5.7404 | 5.74 |
| `TwoLayer_Pirate_2x64_T1.0_soap_rad` | PirateNet 2x64 | 1 | V100 (nesh cluster) | 1.7767 | 1.78 |

Recompute the errors from the stored networks and the ground truth (from the repository root):

```bash
python scripts/diagnostics/score_soap_runs.py TwoLayer_Vanilla_5x128_T1.0_soap_rad TwoLayer_FourierXB_5x128_T1.0_soap_rad TwoLayer_Pirate_2x64_T1.0_soap_rad
```

Figures of the paper in `results/figures/`:

- `two_layer_wavefield_t05.png`
- `two_layer_wavefield_t08.png`
