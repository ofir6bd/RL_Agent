"""Run once: parse raw OpenKBP CSVs and resample every patient to a fixed
cubic grid, caching the result as .npy under data/processed/<split>/<pt>/.

The raw folder is expected to be split into ``train/``, ``validation/`` and
``test/`` subfolders (matching the OpenKBP ``provided-data`` layout). The
processed tree mirrors the same split names.

Usage
-----
python scripts/preprocess.py --raw data/original --out data/processed --grid 64
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
from tqdm import tqdm

# allow `python scripts/preprocess.py` from project root
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.data.preprocess import preprocess_patient


DEFAULT_SPLITS = ("train", "validation", "test")


def _iter_patients(raw_root: Path, splits):
    """Yield (split, raw_patient_path) for every pt_* under each split."""
    for split in splits:
        split_dir = raw_root / split
        if not split_dir.is_dir():
            continue
        for patient_dir in sorted(split_dir.iterdir()):
            if patient_dir.is_dir() and patient_dir.name.startswith("pt"):
                yield split, patient_dir


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--raw", default="data/original",
        help="folder containing <split>/pt_* sub-folders",
    )
    arg_parser.add_argument("--out", default="data/processed")
    arg_parser.add_argument("--grid", type=int, default=64)
    arg_parser.add_argument("--n_beams", type=int, default=9)
    arg_parser.add_argument(
        "--splits", nargs="*", default=list(DEFAULT_SPLITS),
        help="which split folders to process",
    )
    args = arg_parser.parse_args()

    raw_root = Path(args.raw)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    split_patient_pairs = list(_iter_patients(raw_root, args.splits))
    if not split_patient_pairs:
        raise SystemExit(
            f"No pt_* folders found under "
            f"{raw_root}/{{{','.join(args.splits)}}}"
        )

    for split, raw_patient_dir in tqdm(split_patient_pairs, desc="preprocess"):
        output_patient_dir = out_root / split / raw_patient_dir.name
        preprocess_patient(
            raw_patient_dir, output_patient_dir,
            grid=args.grid, n_beams=args.n_beams,
        )


if __name__ == "__main__":
    main()
