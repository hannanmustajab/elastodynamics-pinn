# Marmousi2 crop

Part of the elastic Marmousi2 model (9 to 11 km along the profile, 1.5 to 3.5 km depth), smoothed with a Gaussian filter of width 0.01; window t ∈ [0, 1]. The medium file `marmousi2_crop.npz` is in each code folder.

| Folder | Content |
|---|---|
| `code/<run>/` | `main.py` (training script as run), `global_AF.py` (networks), `soap.py` (optimizer), `run.sh` |
| `ground_truth/marmousi2_x09000_z1500_w2000_s010_t200_clamp/` | DEVITO reference: `snapshots_x` (u_y) and `snapshots_y` (u_x), 101 frames, t = 0 to 1, 512 x 512 float32 |
| `results/<run>/` | training log and the network after 10,000 epochs (`experiments/*/model/checkpoint_epoch_10001.pth`) |
| `results/errors.csv` | time-averaged relative L2 error of each run (%), with the value of Table 3 |
| `results/error_vs_time.csv` | the error at every reference frame |
| `results/figures/` | the figures of the paper for this benchmark |

| Run | Network | t_max | GPU | Error (%) | Table 3 |
|---|---|---|---|---|---|
| `Marmousi2_s010_Vanilla_9x128_T1.0_soap_rad` | Vanilla 9x128 | 1 | V100 (nesh cluster) | 20.7138 | 20.71 |
| `Marmousi2_s010_FourierXB_9x128_T1.0_soap_rad` | Fourier-feature 9x128 | 1 | V100 (nesh cluster) | 18.2518 | 18.25 |
| `Marmousi2_s010_Pirate_3x128_T1.0_soap_rad` | PirateNet 3x128 | 1 | RTX 5090 | 1.6745 | 1.67 |

Recompute the errors from the stored networks and the ground truth (from the repository root):

```bash
python scripts/diagnostics/score_soap_runs.py Marmousi2_s010_Vanilla_9x128_T1.0_soap_rad Marmousi2_s010_FourierXB_9x128_T1.0_soap_rad Marmousi2_s010_Pirate_3x128_T1.0_soap_rad
```

Figures of the paper in `results/figures/`:

- `marmousi2_fourier9x128_wavefield_t09_row.png`
- `marmousi2_model_errors.pdf`
- `marmousi2_piratenet3x128_wavefield_t09_row.png`
- `marmousi2_traces_all_networks_t09.pdf`
- `marmousi2_vanilla9x128_wavefield_t09_row.png`
