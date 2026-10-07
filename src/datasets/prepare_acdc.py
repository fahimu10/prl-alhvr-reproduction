"""Build the ACDC .h5 dataset from the official raw download.

Generates the dataset from the official ACDC release rather than relying
on a third-party preprocessed mirror.

WHY THIS ISN'T JUST THE REFERENCE SCRIPT
----------------------------------------
ALHVR ships `dataloaders/acdc_data_processing.py`, but it cannot produce
the dataset its own code consumes:
  * it writes ONLY per-slice files - no per-volume .h5, which val/test need
  * it emits no data_list/*.list split files at all
  * it indexes slices with `range(image.shape[0])`, i.e. 0-based, but the
    authors' own shipped all_slices.list starts at slice_1 (zero entries
    end in _slice_0). So that script did not generate their data either.
  * its paths are hardcoded to the authors' machine
This script follows its normalisation exactly (per-volume min-max to
[0,1], then cast) while producing all three artefacts, with 1-based slice
numbering so filenames match the authors' lists.

FRAME RENAMING - the subtle part
--------------------------------
Raw ACDC annotates two frames per patient at the ED and ES timepoints,
named by their true frame index: patient001 has frame01 and frame12. The
dataset the code expects renames these to frame01/frame02. Verified over
all 100 training patients:
  * every patient has exactly two *_gt.nii.gz files
  * those two frame numbers always equal (ED, ES) from Info.cfg
  * ED < ES in every case
  * ED is NOT always 1 - patient090 has ED:4, ES:11
So the mapping is by sorted frame order (ED -> frame01, ES -> frame02).
Assuming "the file literally named frame01" would silently mislabel
patient090.

SPLITS ARE COPIED, NEVER REGENERATED
------------------------------------
The train/val/test lists come from the authors' own repo
(train_ALHVR/data/acdc/). Regenerating splits would break comparability
with the paper even if the ratios matched.

Run from the repo root:
    python -m src.datasets.prepare_acdc \\
        --raw-dir data/acdc_raw/training \\
        --out-dir data/acdc_processed/data

Then optionally compare against another processed copy:
    python -m src.datasets.prepare_acdc --out-dir data/acdc_processed/data \\
        --compare-to path/to/other/data --compare-only
"""
import argparse
import glob
import os
import re
import shutil
import sys

import h5py
import numpy as np
import SimpleITK as sitk

# ACDC's NIfTI headers carry an sform whose scales disagree with the
# qform, so ITK prints "unexpected scales in sform" twice per file read -
# ~2400 lines that bury the actual output. It concerns only the spatial
# transform metadata, not the voxel array we read, and the reference
# pipeline ignores that metadata too. Silenced so real problems stay
# visible; remove this line if you need to debug NIfTI headers.
sitk.ProcessObject.SetGlobalWarningDisplay(False)

LIST_FILES = ["train.list", "val.list", "test.list", "train_slices.list", "all_slices.list"]
DEFAULT_LISTS_FROM = "reference/alhvr-official/train_ALHVR/data/acdc"


def read_info_cfg(patient_dir):
    text = open(os.path.join(patient_dir, "Info.cfg")).read()
    ed = int(re.search(r"ED:\s*(\d+)", text).group(1))
    es = int(re.search(r"ES:\s*(\d+)", text).group(1))
    return ed, es


def annotated_frames(patient_dir):
    """Return the two annotated frame numbers, sorted (ED first, then ES).

    Cross-checked against Info.cfg so a mismatch fails loudly rather than
    producing a quietly mislabelled dataset.
    """
    gts = glob.glob(os.path.join(patient_dir, "*_gt.nii.gz"))
    frames = sorted(int(re.search(r"frame0*(\d+)_gt", os.path.basename(g)).group(1)) for g in gts)
    ed, es = read_info_cfg(patient_dir)
    if frames != sorted([ed, es]):
        raise ValueError(f"{patient_dir}: gt frames {frames} != Info.cfg (ED,ES) {(ed, es)}")
    if len(frames) != 2:
        raise ValueError(f"{patient_dir}: expected exactly 2 annotated frames, found {frames}")
    return frames


def process_patient(patient_dir, out_dir, image_dtype):
    patient = os.path.basename(patient_dir)
    n_slices = 0
    for out_idx, frame_num in enumerate(annotated_frames(patient_dir), start=1):
        img_path = os.path.join(patient_dir, f"{patient}_frame{frame_num:02d}.nii.gz")
        gt_path = os.path.join(patient_dir, f"{patient}_frame{frame_num:02d}_gt.nii.gz")

        image = sitk.GetArrayFromImage(sitk.ReadImage(img_path))
        label = sitk.GetArrayFromImage(sitk.ReadImage(gt_path))
        if image.shape != label.shape:
            raise ValueError(f"{img_path}: image {image.shape} != label {label.shape}")

        # Per-volume min-max to [0,1], exactly as the reference script does,
        # BEFORE slicing - so every slice inherits volume-level normalisation.
        image = (image - image.min()) / (image.max() - image.min())
        image = image.astype(image_dtype)
        label = label.astype(np.uint8)

        name = f"{patient}_frame{out_idx:02d}"

        # per-volume file (val/test read these from the root directly)
        with h5py.File(os.path.join(out_dir, f"{name}.h5"), "w") as f:
            f.create_dataset("image", data=image, compression="gzip")
            f.create_dataset("label", data=label, compression="gzip")

        # per-slice files, 1-BASED to match the authors' list files
        for z in range(image.shape[0]):
            slice_name = f"{name}_slice_{z + 1}"
            with h5py.File(os.path.join(out_dir, "slices", f"{slice_name}.h5"), "w") as f:
                f.create_dataset("image", data=image[z], compression="gzip")
                f.create_dataset("label", data=label[z], compression="gzip")
            n_slices += 1
    return n_slices


def copy_lists(lists_from, out_dir):
    dest = os.path.join(out_dir, "data_list")
    os.makedirs(dest, exist_ok=True)
    for name in LIST_FILES:
        src = os.path.join(lists_from, name)
        if not os.path.exists(src):
            raise SystemExit(f"missing split list {src}\n"
                             f"clone the reference: git clone https://github.com/ziziyao/ALHVR reference/alhvr-official")
        shutil.copyfile(src, os.path.join(dest, name))
    print(f"copied {len(LIST_FILES)} split lists from {lists_from}")


def verify(out_dir):
    """Every name referenced by the split lists must exist on disk."""
    ok = True
    dl = os.path.join(out_dir, "data_list")

    with open(os.path.join(dl, "all_slices.list")) as f:
        slice_names = [l.strip() for l in f if l.strip()]
    missing = [n for n in slice_names if not os.path.exists(os.path.join(out_dir, "slices", f"{n}.h5"))]
    print(f"  slices referenced by all_slices.list : {len(slice_names)}, missing: {len(missing)}")
    if missing:
        ok = False
        print(f"    e.g. {missing[:5]}")

    vol_names = []
    for split in ("train.list", "val.list", "test.list"):
        with open(os.path.join(dl, split)) as f:
            vol_names += [l.strip() for l in f if l.strip()]
    missing_vol = [n for n in vol_names if not os.path.exists(os.path.join(out_dir, f"{n}.h5"))]
    print(f"  volumes referenced by train/val/test  : {len(vol_names)}, missing: {len(missing_vol)}")
    if missing_vol:
        ok = False
        print(f"    e.g. {missing_vol[:5]}")

    on_disk = len(glob.glob(os.path.join(out_dir, "slices", "*.h5")))
    print(f"  slice files on disk                   : {on_disk} (expected {len(slice_names)})")
    if on_disk != len(slice_names):
        ok = False
    return ok


def compare(out_dir, other_dir, tol=1e-5, limit=None):
    """Compare our output against another processed copy (e.g. a third-party mirror)."""
    ours = sorted(glob.glob(os.path.join(out_dir, "slices", "*.h5")))
    if limit:
        ours = ours[:limit]
    checked = img_bad = lbl_bad = absent = 0
    worst = 0.0
    for p in ours:
        q = os.path.join(other_dir, "slices", os.path.basename(p))
        if not os.path.exists(q):
            absent += 1
            continue
        with h5py.File(p, "r") as a, h5py.File(q, "r") as b:
            ia, ib = a["image"][:].astype(np.float64), b["image"][:].astype(np.float64)
            la, lb = a["label"][:], b["label"][:]
            if ia.shape != ib.shape:
                img_bad += 1
                continue
            worst = max(worst, float(np.abs(ia - ib).max()))
            if not np.allclose(ia, ib, atol=tol):
                img_bad += 1
            if not np.array_equal(la, lb):
                lbl_bad += 1
        checked += 1
    print(f"  compared {checked} slice files against {other_dir}")
    print(f"    not present there : {absent}")
    print(f"    image mismatches  : {img_bad}  (max abs diff {worst:.3e}, tol {tol})")
    print(f"    label mismatches  : {lbl_bad}")
    return img_bad == 0 and lbl_bad == 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", default="data/acdc_raw/training",
                        help="official ACDC 'training' dir containing patient001..patient100")
    parser.add_argument("--out-dir", required=True, help="dataset root to create (the --root-path for training)")
    parser.add_argument("--lists-from", default=DEFAULT_LISTS_FROM,
                        help="dir holding the authors' split lists; they are copied, never regenerated")
    parser.add_argument("--image-dtype", choices=["float32", "float64"], default="float32",
                        help="float32 matches the reference script's cast; the loader casts to float32 either way")
    parser.add_argument("--compare-to", default=None, help="another processed root to diff against")
    parser.add_argument("--compare-limit", type=int, default=None, help="only compare the first N slice files")
    parser.add_argument("--compare-only", action="store_true", help="skip processing, just compare/verify")
    args = parser.parse_args()

    os.makedirs(os.path.join(args.out_dir, "slices"), exist_ok=True)

    if not args.compare_only:
        patients = sorted(glob.glob(os.path.join(args.raw_dir, "patient*")))
        patients = [p for p in patients if os.path.isdir(p)]
        if len(patients) != 100:
            print(f"WARNING: expected 100 training patients, found {len(patients)} in {args.raw_dir}")
        total = 0
        for i, pdir in enumerate(patients, 1):
            total += process_patient(pdir, args.out_dir, np.dtype(args.image_dtype))
            if i % 20 == 0 or i == len(patients):
                print(f"  processed {i}/{len(patients)} patients, {total} slices")
        print(f"wrote {total} slices and {2 * len(patients)} volumes to {args.out_dir}")
        copy_lists(args.lists_from, args.out_dir)

    print("verifying against the split lists:")
    ok = verify(args.out_dir)

    if args.compare_to:
        print(f"comparing to {args.compare_to}:")
        ok = compare(args.out_dir, args.compare_to, limit=args.compare_limit) and ok

    print("\nOK" if ok else "\nPROBLEMS FOUND - see above")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
