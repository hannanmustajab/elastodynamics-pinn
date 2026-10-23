# Reference solutions and scoring

The errors of the paper compare each trained network with a numerical reference solution of the
same problem, computed with DEVITO. The frames the paper was scored against are stored in
`<benchmark>/ground_truth/` (about 1 GB in all); the scripts in this folder compute them again.
`reference_fingerprints.csv` lets you check that a recomputed reference equals the stored one.

## Files

| File | Purpose |
|---|---|
| `fd_two_forms.py` | Grid, time step, padded domain, layered media and initial displacement shared by all reference solvers; also a second-order finite-difference solver |
| `devito_check.py` | DEVITO reference of the homogeneous, two-layer and six-layer media (mode `reference`) |
| `marmousi2_model.py` | Builds the Marmousi2 crop as a cosine series of the Lamé parameters (`marmousi2_crop.npz`) |
| `devito_marmousi2.py` | DEVITO reference of the Marmousi2 crop (mode `reference`) |
| `make_reference.py` | Second-order finite-difference reference: the one the training runs printed their errors against during training |
| `test_fd_operators.py` | Checks the finite-difference operators on a manufactured solution and the transmission at one interface |
| `check_reference.py` | Compares a regenerated reference with `reference_fingerprints.csv` |
| `score_soap_runs.py` | Scores the final checkpoint of every run against the DEVITO reference and compares the errors with the paper |

## Computing the references

The references of the paper were computed with DEVITO 4.8.14 on the RTX 5090 workstation (its environments hold
4.8.14 and the development build 4.8.14+36.g3948ea77). Install it with `pip install devito==4.8.14`
(`environment/requirements-reference.txt`). Run the commands from the repository root; each one writes `snapshots_x/`,
`snapshots_y/` and `snapshots_u/` with one `.npy` file per frame (float32, 512 x 512, frame i at t = 0.01 i).

```bash
python scripts/diagnostics/devito_check.py reference references/devito_div_homogeneous 2 1.0 homogeneous
python scripts/diagnostics/devito_check.py reference references/devito_div_two_layer 2 1.0 two_layer
python scripts/diagnostics/devito_check.py reference references/devito_div_six_layer_t200 2 2.0 six_layer
python scripts/diagnostics/devito_marmousi2.py reference figures/data/marmousi2_crop_s010.npz \
    references/marmousi2_x09000_z1500_w2000_s010_t200_clamp 2 1.0 clamp
python scripts/diagnostics/check_reference.py references/devito_div_homogeneous references/devito_div_two_layer \
    references/devito_div_six_layer_t200 references/marmousi2_x09000_z1500_w2000_s010_t200_clamp
```

The arguments are the output folder, the grid refinement (2: grid spacing 3/1022, time step 5e-4), the end time and the medium.
The paper's Marmousi2 reference was computed to t = 2 (folder suffix `_t200`); the runs use frames 0 to 100 only, and
the frames up to t = 1 do not depend on the end time. `clamp` continues the medium unchanged along the normal outside the
domain, as the absorbing boundary condition of the training assumes.

All four references are computed on the domain [-2.67, 2.67]^2 with a fixed outer edge, so that no reflection from that
edge re-enters [-1.5, 1.5]^2 within the time window. They are not surrounded by an absorbing layer.

## Marmousi2 medium

`marmousi2_crop.npz` (in each Marmousi2 run folder and in `figures/data/marmousi2_crop_s010.npz`) holds the cosine
coefficients of λ and μ that both the training and the reference evaluate. It was built from the elastic Marmousi2 model
(Martin, Wiley and Marfurt 2006; Allied Geophysical Laboratory, University of Houston, https://www.agl.uh.edu/downloads/,
CC BY 4.0) with

```bash
python scripts/diagnostics/marmousi2_model.py build vp_marmousi-ii.segy.gz vs_marmousi-ii.segy.gz 9.0 1.5 2.0 <out_dir> 0.01
```

(window from 9 km along the profile and 1.5 km depth, 2 km wide; Gaussian filter width 0.01). Rebuilding it from the
downloaded files gives the file of the paper exactly:

| File | SHA-256 |
|---|---|
| `vp_marmousi-ii.segy.gz` | `2c7a0c4170c95a15620c2146573711c61a4cef8dc27338ef6c211a81ffa68920` |
| `vs_marmousi-ii.segy.gz` | `c88bec68fc4bd7c6e624737a20ecd3241c90e10512103e28e61b344fb5baed7d` |
| `marmousi2_x09000_z1500_w2000_s010.npz` (= `marmousi2_crop.npz`) | `773ed11f89acd41c82c10688294dd623565e55203ce8a3120998a95f4bc564c0` |

## Scoring

```bash
python scripts/diagnostics/score_soap_runs.py            # all runs of runs.csv, against <benchmark>/ground_truth/
python scripts/diagnostics/score_soap_runs.py --references references   # against recomputed references
python scripts/diagnostics/score_soap_runs.py SixLayer_Pirate_3x128_T1.0_soap_rad   # one run
```

The script writes `rescored/error_vs_epoch.csv` (the error after 10,000 epochs) and `rescored/error_vs_time.csv` (the error
at every reference time step). It then prints each error next to the value in `figures/data/` and in Table 3 and exits
with status 1 if one differs. On an Apple laptop GPU (MPS) the 23 runs of Table 3 took under 10 minutes; a CPU works but is slower
(`--cpu` forces it). `--references DIR` reads the references from another folder.

## Conventions

- Arrays are indexed `[x, y]`; the layers vary along `y`.
- `snapshots_x` holds the layer-normal component (u_y of the network) and `snapshots_y` the layer-parallel component
  (u_x of the network). The scoring script pairs them accordingly and checks that the swapped pairing gives the larger error.
- The error of the paper is the relative L2 error of the displacement vector over the 512 x 512 grid, averaged over all
  reference frames of the training window. The "magnitude metric" printed next to it is the one the training scripts print.
