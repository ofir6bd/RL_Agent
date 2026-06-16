"""Quick sanity-check for cached .npy files.

Examples
--------
# summary of every .npy in one patient
python scripts/inspect_numpy_arrays.py data/processed/pt_1

# a single file
python scripts/inspect_numpy_arrays.py data/processed/pt_1/ct.npy

# show a middle axial slice of the CT
python scripts/inspect_numpy_arrays.py data/processed/pt_1/ct.npy --show
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np


def summarise(path: Path) -> None:
    array = np.load(path, allow_pickle=True)
    if array.dtype.kind in ("U", "S", "O"):
        print(f"{path.name:25s}  dtype={array.dtype}  "
              f"shape={array.shape}  values={array.tolist()}")
        return
    n_nonzero = int((array != 0).sum()) if array.size else 0
    print(f"{path.name:25s}  dtype={array.dtype}  "
          f"shape={tuple(array.shape)}  "
          f"min={array.min():.4g}  max={array.max():.4g}  "
          f"mean={array.mean():.4g}  "
          f"nonzero={n_nonzero}/{array.size}")


def show_slice(path: Path) -> None:
    import matplotlib.pyplot as plt
    array = np.load(path)
    if array.ndim == 3:
        plt.imshow(array[array.shape[0] // 2], cmap="gray")
    elif array.ndim == 4:
        plt.imshow(array[0, array.shape[1] // 2], cmap="gray")
    else:
        print(f"{path.name}: cannot visualise shape {array.shape}")
        return
    plt.title(f"{path.name}  middle slice")
    plt.colorbar()
    plt.show()


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "target", help=".npy file or folder containing .npy files",
    )
    arg_parser.add_argument(
        "--show", action="store_true",
        help="display the middle slice with matplotlib",
    )
    args = arg_parser.parse_args()

    target_path = Path(args.target)
    if target_path.is_dir():
        for npy_file in sorted(target_path.glob("*.npy")):
            summarise(npy_file)
    else:
        summarise(target_path)
        if args.show:
            show_slice(target_path)


if __name__ == "__main__":
    main()
