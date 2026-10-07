"""U-Net baseline training (Part 1 reproduction) - fully supervised, no unlabeled data.

The official ALHVR repo ships only the full semi-supervised method
(train_ALHVR_acdc.py) - there is no labeled-only baseline script to copy.
This script is therefore CONSTRUCTED, but every component is taken from
ALHVR's own code, with no borrowing from any other codebase:

  - Model: the plain `UNet` class from ALHVR's networks/unet.py (the file
    defines both `UNet` and the ALHVR-specific `UNet_fea_aux`; the
    baseline is the former, ported unchanged).
  - Data: `ACDCDataset` / `RandomGenerator`, ported from ALHVR's
    dataloaders/dataset.py. `BaseDataSets`' own `num=` parameter
    (`self.sample_list = self.sample_list[:num]`) is what restricts
    training to the labeled subset - the mechanism exists in the
    reference dataset class itself.
  - Labeled-subset size: ALHVR's own hardcoded `patients_to_slices`
    table, verbatim (--labelnum 7 -> 136 slices = 10%).
  - Loss: `DiceLoss` (ported verbatim, including its non-standard
    missing-2x numerator) + F.cross_entropy on raw logits - exactly the
    `loss_dice + loss_ce` combination train_ALHVR_acdc.py forms for the
    labeled portion of its batches, with softmaxed input to the Dice term
    and raw logits to CE, as there.
  - Optimizer: SGD(lr=base_lr, momentum=0.9, weight_decay=0.0001), and no
    LR schedule - train_ALHVR_acdc.py sets base_lr once and never updates
    param_group['lr'] anywhere.
  - Validation: every 200 iterations via the ported val_2d, checkpoint
    saved on improved mean val Dice - matching train_ALHVR_acdc.py's loop.
  - Determinism block, seed=1337: same as train_ALHVR_acdc.py's __main__.

batch_size=16 and max_iterations=30000 are ALHVR's own argparse defaults
in train_ALHVR_acdc.py, and are independently confirmed by the paper's
Section 4.2: "for segmentation on the ACDC and AbdomenCT-1K datasets, we
use U-Net as the backbone, with a batch size of 16 and 30k training
iterations."

`drop_last=True` mirrors the reference's batch sampling: ALHVR's
TwoStreamBatchSampler discards the trailing partial batch, so full-size
batches only. A plain shuffled DataLoader is used here because the
two-stream labeled/unlabeled sampler has no meaning for a supervised-only
baseline.

Run from the repo root:
    python -m src.training.train_unet_baseline --root-path data/acdc_processed/data --labelnum 7

Each run writes to its own directory (see src/utils/run_naming.py); the
matching evaluate_acdc command is printed when training ends.
"""
import argparse
import logging
import os
import random
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.acdc import (ACDCDataset, Compose, RandomGenerator,
                               patients_to_slices)
from src.evaluation.val_2d import test_single_volume
from src.losses.dice import DiceLoss
from src.models.unet import UNet
from src.utils.device import get_device
from src.utils.metrics import MetricsLogger, write_config
from src.utils.run_naming import make_run_name

NUM_CLASSES = 4

# 10% / 20% / 100% labeled, per ALHVR's own patients_to_slices table.
LABELNUM_TO_PERCENT = {7: "10pct", 14: "20pct", 70: "100pct"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--exp", type=str, default="unet_baseline")
    parser.add_argument("--labelnum", type=int, default=7, choices=sorted(LABELNUM_TO_PERCENT), help="7=10%%, 14=20%%, 70=100%%")
    parser.add_argument("--max-iterations", type=int, default=30000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--base-lr", type=float, default=0.01)
    parser.add_argument("--patch-size", type=int, nargs=2, default=[256, 256])
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES)
    parser.add_argument("--use-dino", type=int, default=0,
                        help="PART 2: 1 = add a frozen DINOv2 branch fused into the encoder's first level. 0 = the reference encoder (Part 1)")
    parser.add_argument("--dino-weights", type=str, default=None,
                        help="local DINOv2 weights file; omit to use the timm/HF cache")
    parser.add_argument("--width", type=int, default=16,
                        help="PART 2 capacity experiment: base channel width. 16 is ALHVR's own setting and the only valid value for Part 1")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--deterministic", type=int, default=1)
    parser.add_argument("--val-interval", type=int, default=200)
    return parser.parse_args()


def set_determinism(seed):
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def validate(model, valloader, num_classes, device, patch_size):
    model.eval()
    metric_sum = np.zeros((num_classes - 1, 2))
    for sampled_batch in valloader:
        metric_i = test_single_volume(
            sampled_batch["image"], sampled_batch["label"], model, num_classes, device, patch_size
        )
        metric_sum += np.array(metric_i)
    metric_mean = metric_sum / len(valloader)
    mean_dice = float(np.mean(metric_mean, axis=0)[0])
    mean_hd95 = float(np.mean(metric_mean, axis=0)[1])
    return mean_dice, mean_hd95, metric_mean


def train(args, snapshot_path, device, metrics):
    train_ds = ACDCDataset(
        base_dir=args.root_path,
        split="train",
        num=patients_to_slices(args.labelnum),
        transform=Compose([RandomGenerator(tuple(args.patch_size))]),
    )
    val_ds = ACDCDataset(base_dir=args.root_path, split="val")
    logging.info(f"labeled training slices: {len(train_ds)}, val volumes: {len(val_ds)}")

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4,
        pin_memory=(device.type == "cuda"), worker_init_fn=worker_init_fn, drop_last=True,
    )
    valloader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=1)

    model = UNet(in_chns=1, class_num=args.num_classes, width=args.width,
                         use_dino=bool(args.use_dino),
                         dino_weights=args.dino_weights).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=1e-4)
    dice_loss = DiceLoss(n_classes=args.num_classes)

    iter_num = 0
    max_epoch = args.max_iterations // len(trainloader) + 1
    best_dice = 0.0

    for _ in tqdm(range(max_epoch), ncols=70):
        for sampled_batch in trainloader:
            image = sampled_batch["image"].to(device)
            label = sampled_batch["label"].to(device)

            model.train()
            outputs = model(image)
            outputs_soft = F.softmax(outputs, dim=1)
            loss_dice = dice_loss(outputs_soft, label.unsqueeze(1))
            loss_ce = F.cross_entropy(outputs, label.long())
            loss = loss_dice + loss_ce

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            iter_num += 1

            if iter_num % 50 == 0:
                logging.info(f"iter {iter_num}: loss {loss.item():.4f} dice_term {loss_dice.item():.4f} ce_term {loss_ce.item():.4f}")
                metrics.log(iter_num, "train", loss=loss.item(),
                            dice_term=loss_dice.item(), ce_term=loss_ce.item())

            if iter_num > 0 and iter_num % args.val_interval == 0:
                mean_dice, mean_hd95, per_class = validate(model, valloader, args.num_classes, device, tuple(args.patch_size))
                # Per-class is logged, not just the average across classes:
                # Part 2's hypothesis is about RV specifically, and this curve
                # cannot be recovered afterwards from the mean alone.
                pc = "  ".join(f"c{c+1} dice {per_class[c][0]:.4f} hd95 {per_class[c][1]:.2f}"
                               for c in range(args.num_classes - 1))
                logging.info(f"iter {iter_num}: val_mean_dice {mean_dice:.4f} val_mean_hd95 {mean_hd95:.4f} | {pc}")
                metrics.log(iter_num, "val", mean_dice=mean_dice, mean_hd95=mean_hd95)
                for c in range(args.num_classes - 1):
                    metrics.log(iter_num, "val", **{f"dice_c{c+1}": per_class[c][0],
                                                    f"hd95_c{c+1}": per_class[c][1]})
                if mean_dice > best_dice:
                    best_dice = mean_dice
                    torch.save(model.state_dict(), os.path.join(snapshot_path, f"iter_{iter_num}_dice_{round(best_dice, 4)}.pth"))
                    torch.save(model.state_dict(), os.path.join(snapshot_path, "best_model.pth"))
                    logging.info(f"iter {iter_num}: new best val_mean_dice {best_dice:.4f}, checkpoint saved")

            if iter_num >= args.max_iterations:
                break
        if iter_num >= args.max_iterations:
            break

    torch.save(model.state_dict(), os.path.join(snapshot_path, "final_model.pth"))
    logging.info(f"training finished, best val_mean_dice {best_dice:.4f}")
    return best_dice


def main():
    args = parse_args()
    if args.deterministic:
        set_determinism(args.seed)

    # Run-directory naming convention: see src/utils/run_naming.py.
    pct = LABELNUM_TO_PERCENT[args.labelnum]
    # A non-default width or the DINOv2 branch must be visible in the run
    # name, or a Part 2 run would shadow the reported Part 1 baseline in
    # the figures.
    _exp = args.exp + (f"-w{args.width}" if args.width != 16 else "")
    _exp += "-dino" if args.use_dino else ""
    run_name = make_run_name(_exp, pct, args.batch_size, args.seed)
    snapshot_path = os.path.join(args.output_dir, run_name)
    os.makedirs(snapshot_path, exist_ok=True)

    logging.basicConfig(
        filename=os.path.join(snapshot_path, "log.txt"), level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d] %(message)s", datefmt="%H:%M:%S",
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(f"run dir: {snapshot_path}")
    logging.info(str(args))

    device = get_device()
    logging.info(f"device: {device}")

    write_config(os.path.join(snapshot_path, "config.json"), args)

    start = time.time()
    with MetricsLogger(os.path.join(snapshot_path, "metrics.csv")) as metrics:
        train(args, snapshot_path, device, metrics)
    logging.info(f"total training time: {(time.time() - start) / 60:.1f} min")

    # Echo the eval command so the timestamped path never has to be retyped.
    logging.info(
        "evaluate with:\n"
        f"  python -m src.evaluation.evaluate_acdc --root-path {args.root_path} "
        f"--checkpoint {os.path.join(snapshot_path, 'best_model.pth')} "
        f"--save-predictions {os.path.join(snapshot_path, 'test_predictions')}"
    )


if __name__ == "__main__":
    main()
