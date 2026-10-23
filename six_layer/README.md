# Six-layer medium

λ from 20 to 70 and μ from 10 to 55 in six layers along y, ρ = 100; windows t ∈ [0, 1] (`*_T1.0_*`) and [0, 2] (`*_T2.0_*`).

| Folder | Content |
|---|---|
| `code/<run>/` | `main.py` (training script as run), `global_AF.py` (networks), `soap.py` (optimizer), `run.sh` |
| `ground_truth/devito_div_six_layer_t200/` | DEVITO reference: `snapshots_x` (u_y) and `snapshots_y` (u_x), 201 frames, t = 0 to 2; the t_max = 1 runs use frames 0 to 100, 512 x 512 float32 |
| `results/<run>/` | training log and the network after 10,000 epochs (`experiments/*/model/checkpoint_epoch_10001.pth`) |
| `results/errors.csv` | time-averaged relative L2 error of each run (%), with the value of Table 3 |
| `results/error_vs_time.csv` | the error at every reference frame |
| `results/figures/` | the figures of the paper for this benchmark |

| Run | Network | t_max | GPU | Error (%) | Table 3 |
|---|---|---|---|---|---|
| `SixLayer_Vanilla_5x128_T1.0_2batch_soap_rad` | Vanilla 5x128 | 1 | RTX 5090 | 11.9470 | 11.95 |
| `SixLayer_Vanilla_5x128_T1.0_soap_rad` | Vanilla (1 batch per epoch) 5x128 | 1 | H100 (nesh cluster) | 22.4453 | - |
| `SixLayer_FourierXB_5x128_T1.0_soap_rad` | Fourier-feature 5x128 | 1 | V100 (nesh cluster) | 9.1437 | 9.14 |
| `SixLayer_Vanilla_9x128_T1.0_soap_rad` | Vanilla 9x128 | 1 | RTX 5090 | 4.1168 | 4.12 |
| `SixLayer_FourierXB_9x128_T1.0_soap_rad` | Fourier-feature 9x128 | 1 | RTX 5090 | 5.4992 | 5.50 |
| `SixLayer_Pirate_2x64_T1.0_soap_rad` | PirateNet 2x64 | 1 | V100 (nesh cluster) | 3.3374 | 3.34 |
| `SixLayer_Pirate_2x128_T1.0_soap_rad` | PirateNet 2x128 | 1 | V100 (nesh cluster) | 0.3986 | 0.40 |
| `SixLayer_Pirate_3x128_T1.0_soap_rad` | PirateNet 3x128 | 1 | V100 (nesh cluster) | 0.3421 | 0.34 |
| `SixLayer_Vanilla_9x128_T2.0_soap_rad` | Vanilla 9x128 | 2 | H100 (nesh cluster) | 10.7033 | 10.70 |
| `SixLayer_FourierXB_9x128_T2.0_soap_rad` | Fourier-feature 9x128 | 2 | H100 (nesh cluster) | 12.2227 | 12.22 |
| `SixLayer_Pirate_2x64_T2.0_scale1_soap_rad` | PirateNet 2x64 | 2 | RTX 5090 | 9.8226 | 9.82 |
| `SixLayer_Pirate_3x64_T2.0_soap_rad` | PirateNet 3x64 | 2 | V100 (nesh cluster) | 10.8630 | 10.86 |
| `SixLayer_Pirate_2x128_T2.0_soap_rad` | PirateNet 2x128 | 2 | V100 (nesh cluster) | 4.5048 | 4.50 |
| `SixLayer_Pirate_3x128_T2.0_soap_rad` | PirateNet 3x128 | 2 | V100 (nesh cluster) to epoch 5450; RTX 5090 from epoch 5000 | 3.0008 | 3.00 |
| `SixLayer_Pirate_2x64_T2.0_scale2_soap_rad` | PirateNet (s=2) 2x64 | 2 | RTX 5090 | 10.5823 | - |

Recompute the errors from the stored networks and the ground truth (from the repository root):

```bash
python scripts/diagnostics/score_soap_runs.py SixLayer_Vanilla_5x128_T1.0_2batch_soap_rad SixLayer_Vanilla_5x128_T1.0_soap_rad SixLayer_FourierXB_5x128_T1.0_soap_rad SixLayer_Vanilla_9x128_T1.0_soap_rad SixLayer_FourierXB_9x128_T1.0_soap_rad SixLayer_Pirate_2x64_T1.0_soap_rad SixLayer_Pirate_2x128_T1.0_soap_rad SixLayer_Pirate_3x128_T1.0_soap_rad SixLayer_Vanilla_9x128_T2.0_soap_rad SixLayer_FourierXB_9x128_T2.0_soap_rad SixLayer_Pirate_2x64_T2.0_scale1_soap_rad SixLayer_Pirate_3x64_T2.0_soap_rad SixLayer_Pirate_2x128_T2.0_soap_rad SixLayer_Pirate_3x128_T2.0_soap_rad SixLayer_Pirate_2x64_T2.0_scale2_soap_rad
```

Figures of the paper in `results/figures/`:

- `error_vs_epoch_sizes.pdf`
- `sixlayer_piratenet3x128_wavefield_t05.png`
- `sixlayer_piratenet3x128_wavefield_t09.png`
- `sixlayer_piratenet_T2_wavefield_t12_row.png`
- `sixlayer_piratenet_T2_wavefield_t15_row.png`
- `sixlayer_piratenet_T2_wavefield_t20_row.png`
- `sixlayer_traces_all_networks_t09.pdf`
- `sixlayer_vanilla9x128_wavefield_t05.png`
- `sixlayer_vanilla9x128_wavefield_t09.png`
