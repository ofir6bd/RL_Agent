# RL_Agent_Doser

Reinforcement-learning agent for fractionated radiotherapy planning on the
OpenKBP dataset. The pipeline mirrors the diagram in `rl_pipeline2.html`.

## Layout

```
configs/default.yaml                    hyperparameters
scripts/preprocess.py                   raw CSV  ->  data/processed/<pt>/*.npy
scripts/compute_dose_influence_matrix.py  pyRadPlan -> data/processed/<pt>/dose_influence_matrix.npz (sparse CSR)
scripts/inspect_numpy_arrays.py         quick sanity-check for cached *.npy files
src/config.py                           constants & defaults
src/data/                               CSV parsing + resampling
src/env/                                DoseEnv (35 fractions) + reward
src/models/                             3D-CNN encoder + actor / critic heads
src/agents/ppo.py                       PPO with GAE
src/utils/metrics.py                    DVH, D95, MAE, OAR means
train.py                                training entrypoint
evaluate.py                             deterministic eval on test patients
```

## First-time setup

Create and activate a local virtual environment in this folder, then install
the project dependencies. Do this once, before anything else.

### Windows (PowerShell)

```powershell
# 1) create the venv (folder named .venv at the project root)
python -m venv .venv

# 2) allow script execution for this session, then activate
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1

# 3) upgrade pip and install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

To leave the environment later: `deactivate`.
To re-enter it in a new shell: `.\.venv\Scripts\Activate.ps1`.

> Tip: in VS Code press `Ctrl+Shift+P` → **Python: Select Interpreter** and
> pick the interpreter inside `.venv` so the editor and integrated terminal
> use the same environment.

## Download the OpenKBP dataset

The `data/` folder is git-ignored, so the raw CSVs are **not** included in this
repository. They come from the public OpenKBP grand-challenge dataset hosted
at <https://github.com/ababier/open-kbp>.

Each patient is a folder (`pt_1`, `pt_2`, ...) containing CSVs for the CT,
dose, structure masks, possible-dose mask, and voxel dimensions. The dataset
ships pre-split into `train/`, `validation/`, and `test/`. The preprocessing
script expects that same layout under `data/original/`:

```
data/
  original/
    train/
      pt_1/
        ct.csv
        dose.csv
        voxel_dimensions.csv
        possible_dose_mask.csv
        PTV56.csv  PTV63.csv  PTV70.csv
        Brainstem.csv  Mandible.csv  SpinalCord.csv
        LeftParotid.csv  RightParotid.csv
      pt_2/
        ...
    validation/
      pt_201/
        ...
    test/
      pt_241/
        ...
```

### Download the patients

```powershell
# from a scratch folder OUTSIDE this project
git clone https://github.com/ababier/open-kbp.git

# copy each split into this project's data/original/<split>/
New-Item -ItemType Directory -Force -Path ".\data\original\train",".\data\original\validation",".\data\original\test" | Out-Null
Copy-Item -Recurse "..\open-kbp\provided-data\train-pats\pt_*"      ".\data\original\train\"
Copy-Item -Recurse "..\open-kbp\provided-data\validation-pats\pt_*" ".\data\original\validation\"
Copy-Item -Recurse "..\open-kbp\provided-data\test-pats\pt_*"       ".\data\original\test\"
```

Preprocessing mirrors the same split layout under `data/processed/`:

```
data/processed/
  train/      pt_*/...npy
  validation/ pt_*/...npy
  test/       pt_*/...npy
```

`train.py` reads from `data/processed/train/` and `evaluate.py` reads from
`data/processed/validation/` (the "eval" patients) by default. The split
names are configurable via `train_split` and `eval_split` in
[configs/default.yaml](configs/default.yaml).

## Run order (after the env is activated)

```powershell
# 1) install deps (only the first time, or when requirements.txt changes)
pip install -r requirements.txt

# 2) one-time preprocessing  (raw CSVs -> data/processed/<split>/<pt>/*.npy)
python scripts/preprocess.py --raw data/original --out data/processed

# 3) one-time dose-influence-matrix per patient (slow; uses pyRadPlan)
python scripts/compute_dose_influence_matrix.py --processed data/processed

# 4) one-time per-patient warm-start beamlet plan (a* via FISTA-NNLS).
#    train.py supervised-pretrains the actor on these so PPO doesn't
#    start from a uniform-init policy that sprays dose into OARs.
python scripts/compute_warmstart_actions.py --config configs/default.yaml

# 5) train PPO (uses data/processed/train/ by default)
python train.py --config configs/default.yaml

# 6) evaluate the best checkpoint on the eval split (validation by default)
python evaluate.py --config configs/default.yaml --ckpt runs/best.pt
#   override the split if you want, e.g. final test:
# python evaluate.py --config configs/default.yaml --ckpt runs/best.pt --split test
```




## Current baseline reward setup

The current default baseline in [configs/default.yaml](configs/default.yaml) is:

- `lambda_ptv: 1.5`
- `lambda_oar: 1.0`
- `oar_voxel_subweight: 1.0`
- `oar_mean_subweight: 0.2`
- `oar_dmax_subweight: 1.0`

`evaluate.py` also prints a **CONFIG SNAPSHOT** block at the end of each run,
so the exact reward and tolerance settings are saved with the metrics output.

## Command formatting note (PowerShell)

Run plain commands in the terminal (no markdown syntax), e.g.:

```powershell
python train.py --config configs/default.yaml
python evaluate.py --config configs/default.yaml --ckpt runs/best.pt
```

Do **not** paste markdown-style text like `[train.py](...)` or `<best.pt>` into PowerShell.

> Note on D95: previously the dose-influence-matrix downsampler used
> integer-floor source-to-target indexing, so when pyRadPlan's native
> dose grid had any axis coarser than the target grid (typically the
> z-axis at 47 slices vs target 64) up to 25 % of PTV voxels ended up
> unreachable and `D95_PTV*` was pinned to 0 Gy regardless of the
> agent. The fix in `scripts/compute_dose_influence_matrix.py`
> (`_downsample_matrix`) does a two-pass source-side aggregation +
> nearest-source target-side fallback, after which PTV reachability is
> 99.7-100 % across all 206 patients. `evaluate.py` still reports
> `D95r_PTV*` (D95 over the *reachable* voxels) and `%reach_PTV*` so
> any future coverage regression is immediately visible.



 
# charts for a specific patient
python evaluate.py --config configs/default.yaml --ckpt runs/best.pt --charts-patient pt_201

# or just pick whichever patient comes first
python evaluate.py --config configs/default.yaml --ckpt runs/best.pt --charts-patient first