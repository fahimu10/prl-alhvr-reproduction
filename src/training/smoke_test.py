"""Dataset loader smoke test + single-batch overfit test for the U-Net baseline.

Run from the repo root:
    python -m src.training.smoke_test --root-path data/acdc_processed/data

Picks CUDA first if available (see src/utils/device.py), then MPS, then
CPU.

It exists to catch a data/model/loss wiring bug in seconds, before it
wastes hours of GPU time. Run it before starting a long training job.
"""
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import numpy as np
from scipy.ndimage import zoom

from src.datasets.acdc import ACDCDataset, Compose, RandomGenerator
from src.losses.dice import DiceLoss
from src.models.unet import UNet
from src.utils.device import get_device

DATA_ROOT = Path(__file__).resolve().parents[2] / "data" / "acdc_processed" / "data"
PATCH_SIZE = (256, 256)
NUM_CLASSES = 4


class ResizeOnly:
    """Resize to output_size via nearest-neighbor zoom, no augmentation.

    TEST-ONLY scaffolding - deliberately NOT in src/datasets/acdc.py, which
    is kept a pure port of the reference dataset.py. Nothing in the
    reproduction path uses this. The overfit check needs a deterministic
    fixed batch to measure memorization capacity; RandomGenerator's
    rotation/flip would make the batch vary run to run (its RNG is numpy's,
    which torch.manual_seed does not cover).
    """

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        x, y = image.shape
        image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))
        return {"image": image, "label": label}


def dataset_smoke_test(root_dir):
    print("=== dataset loader smoke test ===")
    train_ds = ACDCDataset(
        base_dir=root_dir,
        split="train",
        transform=Compose([RandomGenerator(PATCH_SIZE)]),
    )
    val_ds = ACDCDataset(base_dir=root_dir, split="val")
    print(f"train slices: {len(train_ds)} (expect 1312)")
    print(f"val volumes: {len(val_ds)} (expect 20)")
    assert len(train_ds) == 1312
    assert len(val_ds) == 20

    loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=0)
    batch = next(iter(loader))
    image, label = batch["image"], batch["label"]
    print(f"batch image shape/dtype: {tuple(image.shape)} {image.dtype}")
    print(f"batch label shape/dtype: {tuple(label.shape)} {label.dtype}")
    assert image.shape == (8, 1, *PATCH_SIZE)
    assert label.shape == (8, *PATCH_SIZE)
    assert not torch.isnan(image).any(), "NaNs in image batch"
    uniq = torch.unique(label)
    print(f"label values in batch: {uniq.tolist()}")
    assert set(uniq.tolist()).issubset({0, 1, 2, 3})

    # Edge case: find an actual all-background slice (apex/base slices with
    # no cardiac structure - expected, not corruption) and a slice with all
    # 4 classes present, and confirm the loader/transform handle both fine.
    # "_slice_1" is not all-background for every patient, so search by
    # content instead of filename.
    bg_case = full_class_case = None
    for idx, case in enumerate(train_ds.sample_list):
        labels = set(torch.unique(train_ds[idx]["label"]).tolist())
        if bg_case is None and labels == {0}:
            bg_case = case
        if full_class_case is None and labels == {0, 1, 2, 3}:
            full_class_case = case
        if bg_case and full_class_case:
            break
    assert bg_case is not None, "expected at least one all-background slice in train_slices.list"
    assert full_class_case is not None, "expected at least one slice with all 4 classes present"
    print(f"edge case (background-only): {bg_case}")
    print(f"edge case (all 4 classes): {full_class_case}")

    print("dataset loader smoke test PASSED\n")


def standard_dice_score(preds, label, num_classes):
    """Per-class Dice with the textbook 2x numerator, for verification only.

    Deliberately NOT DiceLoss's formula - this is an independent check of
    segmentation quality. Also returns per-class target pixel counts: a
    tiny-footprint class can legitimately score lower on a memorized batch
    without anything being broken, and the counts make that visible.
    """
    scores, pixel_counts = [], []
    for c in range(num_classes):
        p = (preds == c).float()
        t = (label == c).float()
        denom = p.sum() + t.sum()
        scores.append(((2 * (p * t).sum() / denom).item()) if denom > 0 else float("nan"))
        pixel_counts.append(int(t.sum().item()))
    return scores, pixel_counts


def overfit_test(root_dir, device, iters):
    print("=== single-batch overfit test (U-Net baseline) ===")
    torch.manual_seed(0)
    # ResizeOnly + shuffle=False so the batch is identical across machines:
    # RandomGenerator's RNG is numpy/random, which torch.manual_seed does
    # not cover, and a random rotation can make a small structure harder to
    # memorize - noise this check does not want.
    overfit_ds = ACDCDataset(
        base_dir=root_dir,
        split="train",
        transform=Compose([ResizeOnly(PATCH_SIZE)]),
    )
    loader = DataLoader(overfit_ds, batch_size=8, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    image = batch["image"].to(device)
    label = batch["label"].to(device)

    model = UNet(in_chns=1, class_num=NUM_CLASSES).to(device)
    # Same optimizer settings as the real recipe, so this is representative.
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    dice_loss = DiceLoss(n_classes=NUM_CLASSES)

    ce_history = []
    for it in range(iters):
        model.train()
        optimizer.zero_grad()
        outputs = model(image)
        outputs_soft = F.softmax(outputs, dim=1)
        loss_dice = dice_loss(outputs_soft, label.unsqueeze(1))
        loss_ce = F.cross_entropy(outputs, label.long())
        (loss_dice + loss_ce).backward()
        optimizer.step()
        ce_history.append(loss_ce.item())
        if it % 100 == 0 or it == iters - 1:
            print(f"  iter {it:4d}  dice_term {loss_dice.item():.4f}  ce_term {loss_ce.item():.4f}")

    # Why not assert the combined loss -> 0: ALHVR's DiceLoss omits the 2x
    # numerator, so a perfect prediction floors that term at ~0.5. Ported
    # faithfully, not a bug. The pass/fail check
    # therefore uses CE, which does converge, plus a correctly-normalized
    # Dice computed post-hoc.
    model.eval()
    with torch.no_grad():
        preds = torch.argmax(model(image), dim=1)
    class_dice, pixel_counts = standard_dice_score(preds, label, NUM_CLASSES)
    mean_dice = sum(class_dice) / len(class_dice)
    print(f"ce_term[0]={ce_history[0]:.4f}  ce_term[-1]={ce_history[-1]:.4f}")
    print(f"post-training per-class Dice (standard formula): {[f'{d:.3f}' for d in class_dice]}")
    print(f"target pixel count per class: {pixel_counts}")
    print(f"mean Dice across classes: {mean_dice:.3f}")

    # CE is the primary signal: a real wiring bug (label misalignment,
    # swapped channels, wrong target shape) stops it converging at all.
    # Dice is checked as a mean rather than a per-class minimum, because a
    # tiny-footprint class can sit low without anything being wrong.
    assert ce_history[-1] < 0.05, (
        f"expected cross-entropy to collapse toward 0 on a memorized single "
        f"batch, got {ce_history[-1]:.4f} - check data/model/loss wiring"
    )
    assert mean_dice > 0.75, (
        f"expected mean Dice across classes > 0.75 after overfitting a single "
        f"batch, got {mean_dice:.3f} ({class_dice}) - check data/model/loss wiring"
    )
    print("single-batch overfit test PASSED\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-path", default=str(DATA_ROOT))
    parser.add_argument("--iters", type=int, default=1200)
    args = parser.parse_args()

    device = get_device()
    print(f"device: {device}\n")

    dataset_smoke_test(args.root_path)
    overfit_test(args.root_path, device, iters=args.iters)


if __name__ == "__main__":
    main()
