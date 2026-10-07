"""CPS (cross pseudo supervision) baseline on ACDC - Table 2 row 2 (target 85.34 Dice).

CONSTRUCTION NOTE

CPS is a *different paper's* method that ALHVR only cites for comparison,
so there is no CPS script in the ALHVR repository to port. Because the
official ALHVR repository is the sole implementation reference for this
project, CPS is not imported from the CPS authors' codebase; it is
constructed from ALHVR's own components, with the construction stated
explicitly below.

Three things make the construction well-founded rather than guesswork:

  1. ALHVR is *itself* built on the CPS framework. The paper's Table 1 is
     captioned "training on the ACDC dataset with 10% labeled data using
     the CPS framework", and Fig. 1 analyses CPS predictions directly.
  2. The paper describes CPS (Sec. 2.1): two networks "initialized with
     different parameters and trained independently".
  3. Every mechanical part already exists in our ported ALHVR code and is
     reused unchanged: the UNet, TwoStreamBatchSampler (8 labeled +
     8 unlabeled per batch), DiceLoss + CE supervised loss, the Gaussian
     consistency ramp-up, SGD settings, validation cadence, determinism.

So CPS here is exactly `train_alhvr.py` with the two high-value-region
modules (CG-CPCL, DTCT) replaced by plain cross pseudo supervision, and
nothing else altered.

Three choices that the sources do not pin down, made explicitly:

  a. **Plain `UNet`, not `UNet_fea_aux`.** CPS has no feature-perturbation
     mechanism; the auxiliary decoder exists solely to produce the
     perturbed predictions ALHVR's region partition needs. Using it would
     add a component CPS does not specify.
  b. **Cross-entropy for the cross-supervision term.** ALHVR's own
     analogous cross-network term (`loss_pro_ce`) uses `F.cross_entropy`
     against the other network's pseudo-labels, so CE is the idiom of the
     codebase we are building on.
  c. **Cross supervision applied to the unlabeled half only.** Every
     unsupervised loss in `train_ALHVR_acdc.py` is computed on
     `[labeled_bs:]`; matching that keeps the labeled/unlabeled split
     consistent across all our Part 1 methods.

Loss:  L = L_sup_A + L_sup_B + lambda(t) * ( CE(A_unlab, argmax B) + CE(B_unlab, argmax A) )

Run from the repo root:
    python -m src.training.train_cps --root-path data/acdc_processed/data --labelnum 7
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
                               TwoStreamBatchSampler, patients_to_slices)
from src.evaluation.val_2d import test_single_volume
from src.losses.dice import DiceLoss
from src.models.unet import UNet
from src.utils.device import get_device
from src.utils.metrics import MetricsLogger, write_config
from src.utils.ramps import sigmoid_rampup
from src.utils.run_naming import make_run_name

LABELNUM_TO_PERCENT = {7: "10pct", 14: "20pct", 70: "100pct"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--exp", type=str, default="cps")
    parser.add_argument("--labelnum", type=int, default=7, choices=sorted(LABELNUM_TO_PERCENT))
    parser.add_argument("--max-iterations", type=int, default=30000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--labeled-bs", type=int, default=8)
    parser.add_argument("--base-lr", type=float, default=0.01)
    parser.add_argument("--patch-size", type=int, nargs=2, default=[256, 256])
    parser.add_argument("--num-classes", type=int, default=4)
    parser.add_argument("--use-dino", type=int, default=0,
                        help="PART 2: 1 = add a frozen DINOv2 branch fused into the encoder's first level. 0 = the reference encoder (Part 1)")
    parser.add_argument("--dino-weights", type=str, default=None,
                        help="local DINOv2 weights file; omit to use the timm/HF cache")
    parser.add_argument("--width", type=int, default=16,
                        help="PART 2 capacity experiment: base channel width. 16 is ALHVR's own setting and the only valid value for Part 1")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--deterministic", type=int, default=1)
    parser.add_argument("--val-interval", type=int, default=200)
    # ramp-up identical to train_ALHVR_acdc.py's own defaults
    parser.add_argument("--consistency", type=float, default=0.1)
    parser.add_argument("--consistency-rampup", type=float, default=200.0)
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
    return (float(np.mean(metric_mean, axis=0)[0]),
            float(np.mean(metric_mean, axis=0)[1]),
            metric_mean)


def train(args, snapshot_path, device, metrics):
    labeled_bs = args.labeled_bs
    num_classes = args.num_classes
    patch_size = tuple(args.patch_size)

    db_train = ACDCDataset(
        base_dir=args.root_path, split="train", num=None,
        transform=Compose([RandomGenerator(patch_size)]),
    )
    db_val = ACDCDataset(base_dir=args.root_path, split="val")

    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args.labelnum)
    logging.info(f"total slices: {total_slices}, labeled slices: {labeled_slice}")
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size - labeled_bs
    )

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4,
                             pin_memory=(device.type == "cuda"), worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)

    # Two independently initialised networks - PyTorch's default init makes
    # them differ, which is the whole basis of cross pseudo supervision.
    model1 = UNet(in_chns=1, class_num=num_classes, width=args.width,
                         use_dino=bool(args.use_dino),
                         dino_weights=args.dino_weights).to(device)
    model2 = UNet(in_chns=1, class_num=num_classes, width=args.width,
                         use_dino=bool(args.use_dino),
                         dino_weights=args.dino_weights).to(device)
    optimizer1 = torch.optim.SGD(model1.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=0.0001)
    optimizer2 = torch.optim.SGD(model2.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=0.0001)
    dice_loss = DiceLoss(n_classes=num_classes)

    iter_num = 0
    max_epoch = args.max_iterations // len(trainloader) + 1
    best_perf1, best_perf2 = 0.0, 0.0
    logging.info(f"{len(trainloader)} iterations per epoch, {max_epoch} epochs")

    iterator = tqdm(range(max_epoch), ncols=70)
    for _ in iterator:
        for sampled_batch in trainloader:
            volume_batch = sampled_batch["image"].to(device)
            label_batch = sampled_batch["label"].to(device)

            model1.train()
            model2.train()

            outputs1 = model1(volume_batch)
            outputs2 = model2(volume_batch)
            outputs_soft1 = F.softmax(outputs1, dim=1)
            outputs_soft2 = F.softmax(outputs2, dim=1)

            # supervised loss on the labeled half - identical to ALHVR's
            loss_dice1 = dice_loss(outputs_soft1[:labeled_bs], label_batch[:labeled_bs].unsqueeze(1))
            loss_dice2 = dice_loss(outputs_soft2[:labeled_bs], label_batch[:labeled_bs].unsqueeze(1))
            loss_ce1 = F.cross_entropy(outputs1[:labeled_bs], label_batch[:labeled_bs].long())
            loss_ce2 = F.cross_entropy(outputs2[:labeled_bs], label_batch[:labeled_bs].long())
            loss_sup1 = loss_dice1 + loss_ce1
            loss_sup2 = loss_dice2 + loss_ce2

            # cross pseudo supervision on the unlabeled half: each network's
            # hard prediction becomes the other's target. detach() because the
            # pseudo-label is a fixed target, not a path for gradient.
            pseudo1 = torch.argmax(outputs_soft1[labeled_bs:].detach(), dim=1)
            pseudo2 = torch.argmax(outputs_soft2[labeled_bs:].detach(), dim=1)
            loss_cps1 = F.cross_entropy(outputs1[labeled_bs:], pseudo2)
            loss_cps2 = F.cross_entropy(outputs2[labeled_bs:], pseudo1)

            consistency_weight = args.consistency * sigmoid_rampup(iter_num // 150, args.consistency_rampup)

            loss1 = loss_sup1 + consistency_weight * loss_cps1
            loss2 = loss_sup2 + consistency_weight * loss_cps2
            loss = loss1 + loss2

            optimizer1.zero_grad()
            optimizer2.zero_grad()
            loss.backward()
            optimizer1.step()
            optimizer2.step()
            iter_num += 1

            if iter_num % 50 == 0:
                logging.info(
                    f"iter {iter_num}: loss {loss.item():.4f} "
                    f"sup1 {loss_sup1.item():.4f} sup2 {loss_sup2.item():.4f} "
                    f"cps1 {loss_cps1.item():.4f} cps2 {loss_cps2.item():.4f} "
                    f"cw {consistency_weight:.4f}"
                )
                metrics.log(iter_num, "train", loss=loss.item(),
                            sup1=loss_sup1.item(), sup2=loss_sup2.item(),
                            cps1=loss_cps1.item(), cps2=loss_cps2.item(),
                            consistency_weight=consistency_weight)

            if iter_num > 0 and iter_num % args.val_interval == 0:
                for tag, model, prev_best in (("1", model1, best_perf1), ("2", model2, best_perf2)):
                    mean_dice, mean_hd95, per_class = validate(model, valloader, num_classes, device, patch_size)
                    pc = "  ".join(f"c{c+1} dice {per_class[c][0]:.4f} hd95 {per_class[c][1]:.2f}"
                                   for c in range(num_classes - 1))
                    logging.info(f"iter {iter_num}: model{tag} val_mean_dice {mean_dice:.4f} "
                                 f"val_mean_hd95 {mean_hd95:.4f} | {pc}")
                    metrics.log(iter_num, f"val/model{tag}",
                                mean_dice=mean_dice, mean_hd95=mean_hd95)
                    for c in range(num_classes - 1):
                        metrics.log(iter_num, f"val/model{tag}",
                                    **{f"dice_c{c+1}": per_class[c][0],
                                       f"hd95_c{c+1}": per_class[c][1]})
                    if mean_dice > prev_best:
                        torch.save(model.state_dict(), os.path.join(snapshot_path, f"best_model{tag}.pth"))
                        logging.info(f"iter {iter_num}: model{tag} new best val_mean_dice {mean_dice:.4f}, checkpoint saved")
                        if tag == "1":
                            best_perf1 = mean_dice
                        else:
                            best_perf2 = mean_dice
                    model.train()

            if iter_num >= args.max_iterations:
                break
        if iter_num >= args.max_iterations:
            iterator.close()
            break

    torch.save(model1.state_dict(), os.path.join(snapshot_path, "final_model1.pth"))
    torch.save(model2.state_dict(), os.path.join(snapshot_path, "final_model2.pth"))
    logging.info(f"training finished, best val_mean_dice model1 {best_perf1:.4f} model2 {best_perf2:.4f}")
    return max(best_perf1, best_perf2)


def main():
    args = parse_args()
    if args.deterministic:
        set_determinism(args.seed)

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
    logging.info(
        "evaluate with:\n"
        f"  python -m src.evaluation.evaluate_acdc --root-path {args.root_path} "
        f"--checkpoint {os.path.join(snapshot_path, 'best_model1.pth')} "
        f"--save-predictions {os.path.join(snapshot_path, 'test_predictions')}"
    )


if __name__ == "__main__":
    main()
