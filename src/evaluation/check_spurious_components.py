"""One-off diagnostic: are 95HD/ASD inflated by spurious small blobs?

Checks the hypothesis that the U-Net baseline's poor 95HD/ASD (despite
good Dice) is caused by small false-positive connected components far
from the main structure - a handful of stray pixels barely dents Dice
but can massively inflate boundary-distance metrics. Not part of the
core reproduction pipeline - a post-hoc analysis script to understand a
specific result.

Run from the repo root:
    python -m src.evaluation.check_spurious_components --pred-dir outputs/<run>/test_predictions
"""
import argparse
import glob
import os

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import label


def analyze_volume(pred_path, num_classes=4):
    pred = sitk.GetArrayFromImage(sitk.ReadImage(pred_path)).astype(np.int32)
    results = []
    for c in range(1, num_classes):
        mask = pred == c
        total_px = int(mask.sum())
        if total_px == 0:
            results.append((c, 0, 0, 0.0))
            continue
        labeled, n_components = label(mask)
        if n_components <= 1:
            results.append((c, total_px, n_components, 0.0))
            continue
        sizes = np.bincount(labeled.ravel())[1:]  # skip background label 0
        main_size = sizes.max()
        stray_px = total_px - main_size
        results.append((c, total_px, n_components, stray_px / total_px))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pred-dir", type=str, required=True)
    parser.add_argument("--num-classes", type=int, default=4)
    args = parser.parse_args()

    pred_files = sorted(glob.glob(os.path.join(args.pred_dir, "*_pred.nii.gz")))
    if not pred_files:
        print(f"no *_pred.nii.gz files found in {args.pred_dir}")
        return

    print(f"{'case':30s} {'class':>5s} {'total_px':>9s} {'n_components':>13s} {'stray_frac':>11s}")
    multi_component_count = 0
    total_checked = 0
    for path in pred_files:
        case = os.path.basename(path).replace("_pred.nii.gz", "")
        for c, total_px, n_components, stray_frac in analyze_volume(path, args.num_classes):
            total_checked += 1
            if n_components > 1:
                multi_component_count += 1
                print(f"{case:30s} {c:5d} {total_px:9d} {n_components:13d} {stray_frac:10.2%}")

    print(f"\n{multi_component_count}/{total_checked} (case, class) pairs have >1 connected component")


if __name__ == "__main__":
    main()
