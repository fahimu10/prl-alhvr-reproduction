"""ALHVR training on ACDC - full method and both Table 5 ablations.

Ported from ALHVR's train_ALHVR_acdc.py. Two networks with identical
structure and independent parameters; each decoder is run on clean and on
perturbed features, giving `out` and `out_aux` per network.

Region partition (paper Eqs. 11-14). An adaptive threshold gamma is the
mean of the lowest beta-fraction of the two networks' averaged perturbed
confidences, recomputed every batch:
    Omega_1 reliable stable    - both confident            (not trained on)
    Omega_2 reliable unstable  - exactly one confident     -> CG-CPCL
    Omega_3 unreliable stable  - neither confident         -> DTCT

  CG-CPCL (Eqs. 5-6, 15-20): class prototypes are built from each
  network's CLEAN decoder features via masked average pooling, then each
  network's PERTURBED features are scored by cosine similarity against
  the OTHER network's prototypes - the "cross" in cross-prototype - and
  supervised by the other network's pseudo-labels, masked to Omega_2.

  DTCT (Eqs. 21-25): per pixel, whichever unperturbed prediction is more
  confident becomes the teacher; that is sharpened and used as an MSE
  target for the perturbed predictions, masked to Omega_3, plus an entropy
  regulariser on the same region.

Ablations come from --use-cgcpcl / --use-dtct (both on = full ALHVR):
    both on   -> ALHVR             (paper Table 5: 90.56 Dice)
    cgcpcl    -> CG-CPCL only      (paper Table 5: 89.32 Dice)
    dtct      -> DTCT only         (paper Table 5: 89.19 Dice)
Turning BOTH off does NOT reproduce Table 5's 86.61 "neither" row: the
released script has no standalone cross pseudo-supervision term, so with
both modules off it is supervised-only.

Run from the repo root:
    python -m src.training.train_alhvr --root-path data/acdc_processed/data --labelnum 7

Each run writes to its own self-describing, sortable directory:
    outputs/<exp>_<pct>_bs<batch_size>_seed<seed>_<YYYYmmdd-HHMMSS>/
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
from src.models.unet import UNet_fea_aux
from src.utils.device import get_device
from src.utils.metrics import MetricsLogger, write_config
from src.utils.prototype import calDist_2D, getPrototype_2D
from src.utils.ramps import sigmoid_rampup
from src.utils.run_naming import make_run_name

LABELNUM_TO_PERCENT = {7: "10pct", 14: "20pct", 70: "100pct"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--exp", type=str, default="alhvr")
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
    # ALHVR hyperparameters (defaults are train_ALHVR_acdc.py's own)
    parser.add_argument("--consistency", type=float, default=0.1)
    parser.add_argument("--consistency-rampup", type=float, default=200.0)
    parser.add_argument("--temperature", type=float, default=0.1, help="sharpening T; exponent used is 1/T")
    parser.add_argument("--temperature-schedule", choices=["constant", "linear", "step"],
                        default="constant",
                        help="PART 2: T warm-up. 'constant' is the reference")
    parser.add_argument("--temperature-start", type=float, default=0.5,
                        help="PART 2: starting (gentler) T for linear/step schedules")
    parser.add_argument("--temperature-rampup", type=int, default=0,
                        help="PART 2: iterations over which T anneals; 0 = half of --max-iterations")
    parser.add_argument("--proportion", type=float, default=0.8, help="beta: fraction of lowest confidences forming gamma")
    parser.add_argument("--scaler", type=float, default=1.0, help="prototype cosine-similarity multiplier")
    # ablation switches
    parser.add_argument("--use-cgcpcl", type=int, default=1, help="1=on (Omega_2 prototype loss)")
    parser.add_argument("--use-dtct", type=int, default=1, help="1=on (Omega_3 teacher-competition loss)")
    parser.add_argument("--use-class-threshold", type=int, default=0,
                        help="PART 2: 1=class-aware gamma_c, 0=reference global gamma")
    parser.add_argument("--class-threshold-scope", choices=["all", "foreground"], default="all",
                        help="PART 2: which classes get their own gamma_c")
    parser.add_argument("--class-threshold-mode", choices=["offset", "quantile", "inverse"], default="offset",
                        help="PART 2: how gamma_c is derived; see generate_threshold_per_class")
    parser.add_argument("--class-threshold-alpha", type=float, default=0.5,
                        help="PART 2, 'inverse' mode only: fraction of the remaining headroom "
                             "(1 - gamma) the hardest class's threshold moves upward; 1.0 would "
                             "put it at gamma=1, i.e. the whole class low-confidence")
    return parser.parse_args()


def set_determinism(seed):
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def generate_threshold(con, proportion):
    """Adaptive threshold gamma (Eq. 11): mean of the lowest K confidences."""
    k = int(con.numel() * proportion)
    lowestk_con, _ = torch.topk(con.view(-1), k, largest=False)
    return torch.mean(lowestk_con)


def generate_threshold_per_class(con, cls, proportion, num_classes,
                                 mode="quantile", scope="all", alpha=0.25):
    """PART 2 CONTRIBUTION - class-aware threshold gamma_c.

    Replaces Eq. 11's single pooled threshold with one per predicted class.
    The confidence map and the region comparisons are unchanged; only how
    the threshold is derived differs.

        mode="quantile"   gamma_c = mean of the lowest `proportion` fraction
                          of class c's own confidences
        mode="offset"     gamma_c = global gamma, shifted DOWN by how much
                          less confident class c is than the pooled average
        mode="inverse"    gamma_c = global gamma, shifted UP by alpha times
                          the same quantity - see below
        scope="all"       every class, background included
        scope="foreground" background keeps the global gamma; the Part 2
                          proposal specifies this variant, and background is
                          ~88% of pixels so the choice is not cosmetic

    "quantile" AND "offset" BOTH PERFORM WORSE THAN THE GLOBAL THRESHOLD
    (RV, the class this targeted, is harmed rather than helped). Kept so
    the negative result stays reproducible.

    Why they fail, from the thresholds the runs logged: background lands at
    ~0.994, tracking the reference's global gamma, while every foreground
    class sits 0.08-0.20 BELOW it. Thresholds are monotone, so a lower bar
    marks FEWER of a class's pixels as low-confidence. Both modes therefore
    move foreground pixels out of Omega_2/Omega_3 and into Omega_1, which
    carries no unsupervised loss at all - the opposite of the intent. Any
    threshold derived from a class's own statistics does this: it normalises
    away the difficulty signal it is trying to exploit.

    "inverse" tests that diagnosis by flipping the sign - an under-confident
    class gets a HIGHER bar, so more of its pixels enter the regions the
    modules act on:

        gamma_c = gamma_global + alpha * (1 - gamma_global) * deficit_c

    The step is measured in units of the REMAINING HEADROOM (1 - gamma), not
    raw confidence, and `deficit_c` is normalised to [-1, 1] across classes.
    An unscaled shift, alpha * (mu_global - mu_c), pins every foreground
    class at the clamp within 200 iterations: gamma_global is ~0.98 even
    that early, so the headroom is ~0.01 while the raw confidence gap is
    ~0.1 - any alpha large enough to matter saturates. The scale that
    matters here is tiny and that is not a flaw: across the whole beta sweep
    the reference's gamma moves only 0.9912 -> 0.9955, and that 0.004 window
    is worth +0.78 Dice.

    Background is more confident than average, so its deficit is negative and
    it gets a LOWER bar - the mode adds signal on hard classes and removes
    background noise from Omega_3 in one step.

    Watch `train/frac_low_con_c1`: past ~0.5 the run is degenerate.

    See the README for the results of each mode.

    Args:
        con: (N,H,W) confidence map, Eq. 11's (max_aux1 + max_aux2) / 2
        cls: (N,H,W) class per pixel, argmax of the averaged auxiliary
             softmax - groups pixels only, never thresholds them
    Returns:
        (num_classes,) tensor of thresholds, indexable by class id.

    Two NaN guards: a class absent from the batch falls back to the global
    gamma, and k is clamped to >= 1 so a tiny class cannot ask for topk(0).
    """
    flat_con = con.reshape(-1)
    flat_cls = cls.reshape(-1)
    global_gamma = generate_threshold(con, proportion)
    mu_global = flat_con.mean()
    gammas = con.new_empty(num_classes)

    if mode == "inverse":
        # Per-class under-confidence, normalised by the largest magnitude in
        # this batch so the values are comparable across iterations as the
        # absolute confidences drift upward during training.
        deficit = con.new_zeros(num_classes)
        for c in range(num_classes):
            sel_c = flat_con[flat_cls == c]
            if sel_c.numel():
                deficit[c] = mu_global - sel_c.mean()
        scale = deficit.abs().max()
        deficit = deficit / scale if scale > 0 else deficit

    for c in range(num_classes):
        if scope == "foreground" and c == 0:
            gammas[c] = global_gamma
            continue
        sel = flat_con[flat_cls == c]
        if sel.numel() == 0:
            gammas[c] = global_gamma
            continue
        if mode == "offset":
            gammas[c] = (global_gamma - (mu_global - sel.mean())).clamp(0.0, 1.0)
        elif mode == "inverse":
            # deficits are normalised to [-1, 1] across classes, so the
            # largest shift is exactly alpha * (1 - global_gamma) and
            # saturation cannot occur.
            gammas[c] = global_gamma + alpha * (1.0 - global_gamma) * deficit[c]
        else:
            k = max(1, int(sel.numel() * proportion))
            lowest, _ = torch.topk(sel, k, largest=False)
            gammas[c] = lowest.mean()
    return gammas


def temperature_at(iter_num, args):
    """Sharpening temperature for this iteration (Part 2, T scheduling).

    The exponent in `sharpening` is 1/T, so a LARGER T is GENTLER: T=0.5
    gives exponent 2, T=0.1 gives exponent 10. Both schedules therefore
    start at `--temperature-start` (gentle) and end at `--temperature`
    (the reference's 0.1), which is what "gentler early sharpening" means.

        constant - the reference: T fixed at --temperature throughout
        linear   - T falls linearly from start to end over the rampup,
                   then holds
        step     - T is start until the rampup, then end

    `--temperature-rampup` defaults to half of max_iterations. Only has any
    effect when DTCT is enabled; sharpening exists nowhere else.
    """
    if args.temperature_schedule == "constant":
        return args.temperature
    t0, t1 = args.temperature_start, args.temperature
    r = args.temperature_rampup or max(1, args.max_iterations // 2)
    if args.temperature_schedule == "step":
        return t0 if iter_num < r else t1
    return t0 + (t1 - t0) * min(1.0, iter_num / r)      # linear


def sharpening(P, temperature):
    """Probability sharpening (Eq. 22). Note the exponent is 1/temperature."""
    T = 1 / temperature
    return P ** T / (P ** T + (1 - P) ** T)


def compute_prototypes(fea1, fea2, max_index1, max_index2, num_classes, flag=1):
    """flag=1: prototypes + upsampled features. flag=0: upsampled features only."""
    if flag == 0:
        fts1 = F.interpolate(fea1, size=max_index1.shape[-2:], mode='bilinear')
        fts2 = F.interpolate(fea2, size=max_index2.shape[-2:], mode='bilinear')
        return fts1, fts2

    one_hot1 = torch.nn.functional.one_hot(max_index1, num_classes=num_classes).permute(0, 3, 1, 2)
    one_hot2 = torch.nn.functional.one_hot(max_index2, num_classes=num_classes).permute(0, 3, 1, 2)

    fts1 = F.interpolate(fea1, size=max_index1.shape[-2:], mode='bilinear')
    fts2 = F.interpolate(fea2, size=max_index2.shape[-2:], mode='bilinear')

    prototypes1 = getPrototype_2D(fts1, one_hot1)
    prototypes2 = getPrototype_2D(fts2, one_hot2)
    return prototypes1, prototypes2, fts1, fts2


def compute_prototype_loss(fts1, fts2, prototypes1, prototypes2, max_index1, max_index2,
                           high_low_con_mask1, high_low_con_mask2, scaler):
    """CG-CPCL loss: each network's perturbed features vs the other's prototypes."""
    pro_cos1 = torch.stack([calDist_2D(fts1, p2, scaler=scaler) for p2 in prototypes2], dim=1)
    pro_cos2 = torch.stack([calDist_2D(fts2, p1, scaler=scaler) for p1 in prototypes1], dim=1)

    loss_pro_ce1 = F.cross_entropy(pro_cos1, max_index2, reduction='none')
    loss_pro_ce2 = F.cross_entropy(pro_cos2, max_index1, reduction='none')

    loss_pro_ce1 = torch.sum(high_low_con_mask1.unsqueeze(1) * loss_pro_ce1) / (
        torch.sum(high_low_con_mask1.unsqueeze(1)) + 1e-16)
    loss_pro_ce2 = torch.sum(high_low_con_mask2.unsqueeze(1) * loss_pro_ce2) / (
        torch.sum(high_low_con_mask2.unsqueeze(1)) + 1e-16)

    loss_pro_cos1 = ((1 - pro_cos1) * high_low_con_mask1.unsqueeze(1)).mean()
    loss_pro_cos2 = ((1 - pro_cos2) * high_low_con_mask2.unsqueeze(1)).mean()
    return loss_pro_ce1, loss_pro_ce2, loss_pro_cos1, loss_pro_cos2


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

    model1 = UNet_fea_aux(in_chns=1, class_num=num_classes, width=args.width,
                         use_dino=bool(args.use_dino),
                         dino_weights=args.dino_weights).to(device)
    model2 = UNet_fea_aux(in_chns=1, class_num=num_classes, width=args.width,
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

            outputs1, outputs_aux1, fea1, fea_aux1 = model1(volume_batch)
            outputs2, outputs_aux2, fea2, fea_aux2 = model2(volume_batch)

            outputs_soft1 = F.softmax(outputs1, dim=1)
            outputs_soft2 = F.softmax(outputs2, dim=1)
            outputs_soft_aux1 = F.softmax(outputs_aux1, dim=1)
            outputs_soft_aux2 = F.softmax(outputs_aux2, dim=1)

            # supervised loss on the labeled half (Eqs. 3-4)
            loss_dice1 = dice_loss(outputs_soft1[:labeled_bs], label_batch[:labeled_bs].unsqueeze(1))
            loss_dice2 = dice_loss(outputs_soft2[:labeled_bs], label_batch[:labeled_bs].unsqueeze(1))
            loss_ce1 = F.cross_entropy(outputs1[:labeled_bs], label_batch[:labeled_bs].long())
            loss_ce2 = F.cross_entropy(outputs2[:labeled_bs], label_batch[:labeled_bs].long())
            loss_sup1 = loss_dice1 + loss_ce1
            loss_sup2 = loss_dice2 + loss_ce2

            consistency_weight = args.consistency * sigmoid_rampup(iter_num // 150, args.consistency_rampup)

            max_value_aux1, max_index_aux1 = outputs_soft_aux1[labeled_bs:].max(dim=1)
            max_value_aux2, max_index_aux2 = outputs_soft_aux2[labeled_bs:].max(dim=1)
            max_value1, max_index1 = outputs_soft1[labeled_bs:].max(dim=1)
            max_value2, max_index2 = outputs_soft2[labeled_bs:].max(dim=1)

            # region partition (Eqs. 11-14). `thr` is a scalar for the
            # reference's global gamma and a per-pixel map for Part 2's
            # gamma_c; the comparisons below are identical either way.
            con_avg = (max_value_aux1 + max_value_aux2) * 0.5
            if args.use_class_threshold:
                cls_avg = ((outputs_soft_aux1[labeled_bs:] + outputs_soft_aux2[labeled_bs:]) * 0.5).argmax(dim=1)
                gamma_c = generate_threshold_per_class(con_avg, cls_avg, args.proportion,
                                                       num_classes, args.class_threshold_mode,
                                                       args.class_threshold_scope,
                                                       args.class_threshold_alpha)
                threshold = gamma_c.mean()          # logged scalar, comparable across variants
                thr = gamma_c[cls_avg]              # (N,H,W) per-pixel threshold
            else:
                threshold = generate_threshold(con_avg, args.proportion)
                gamma_c = None
                thr = threshold
            low_con_mask = ((max_value_aux1 < thr) & (max_value_aux2 < thr)).to(torch.int32)
            high_low_con_mask1 = ((max_value_aux1 <= thr) & (max_value_aux2 > thr)).to(torch.int32)
            high_low_con_mask2 = ((max_value_aux2 <= thr) & (max_value_aux1 > thr)).to(torch.int32)

            loss_focus1 = torch.zeros((), device=device)
            loss_focus2 = torch.zeros((), device=device)
            if args.use_dtct:
                # DTCT: more confident unperturbed prediction becomes teacher (Eq. 21)
                new_outputs_soft = torch.where((max_value1 > max_value2).unsqueeze(1),
                                               outputs_soft1[labeled_bs:], outputs_soft2[labeled_bs:])
                temperature = temperature_at(iter_num, args)
                sharpened = sharpening(new_outputs_soft, temperature)
                mse_dist1 = (outputs_soft_aux1[labeled_bs:] - sharpened) ** 2
                mse_dist2 = (outputs_soft_aux2[labeled_bs:] - sharpened) ** 2
                mse1 = torch.sum(low_con_mask.unsqueeze(1) * mse_dist1) / (torch.sum(low_con_mask.unsqueeze(1)) + 1e-16)
                mse2 = torch.sum(low_con_mask.unsqueeze(1) * mse_dist2) / (torch.sum(low_con_mask.unsqueeze(1)) + 1e-16)

                preds1 = outputs_soft1[labeled_bs:] * low_con_mask.unsqueeze(1)
                preds_aux1 = outputs_soft_aux1[labeled_bs:] * low_con_mask.unsqueeze(1)
                preds2 = outputs_soft2[labeled_bs:] * low_con_mask.unsqueeze(1)
                preds_aux2 = outputs_soft_aux2[labeled_bs:] * low_con_mask.unsqueeze(1)

                uncertainty1 = -1.0 * torch.sum(preds1 * torch.log(preds1 + 1e-6), dim=1, keepdim=True)
                uncertainty2 = -1.0 * torch.sum(preds2 * torch.log(preds2 + 1e-6), dim=1, keepdim=True)
                uncertainty_aux1 = -1.0 * torch.sum(preds_aux1 * torch.log(preds_aux1 + 1e-6), dim=1, keepdim=True)
                uncertainty_aux2 = -1.0 * torch.sum(preds_aux2 * torch.log(preds_aux2 + 1e-6), dim=1, keepdim=True)

                loss_focus1 = mse1 + torch.mean(uncertainty1) + torch.mean(uncertainty_aux1)
                loss_focus2 = mse2 + torch.mean(uncertainty2) + torch.mean(uncertainty_aux2)

            loss_pro1 = torch.zeros((), device=device)
            loss_pro2 = torch.zeros((), device=device)
            if args.use_cgcpcl:
                # CG-CPCL: prototypes from clean features, scored on perturbed features
                prototypes1, prototypes2, _, _ = compute_prototypes(
                    fea1[labeled_bs:], fea2[labeled_bs:], max_index1, max_index2, num_classes, 1)
                fts_aux1, fts_aux2 = compute_prototypes(
                    fea_aux1[labeled_bs:], fea_aux2[labeled_bs:], max_index_aux1, max_index_aux2, num_classes, 0)
                loss_pro_ce1, loss_pro_ce2, loss_pro_cos1, loss_pro_cos2 = compute_prototype_loss(
                    fts_aux1, fts_aux2, prototypes1, prototypes2, max_index1, max_index2,
                    high_low_con_mask1, high_low_con_mask2, args.scaler)
                loss_pro1 = loss_pro_ce1 + loss_pro_cos1
                loss_pro2 = loss_pro_ce2 + loss_pro_cos2

            loss1 = loss_sup1 + consistency_weight * (loss_focus1 + loss_pro1)
            loss2 = loss_sup2 + consistency_weight * (loss_focus2 + loss_pro2)
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
                    f"pro1 {loss_pro1.detach().item():.4f} pro2 {loss_pro2.detach().item():.4f} "
                    f"focus1 {loss_focus1.detach().item():.4f} focus2 {loss_focus2.detach().item():.4f} "
                    f"cw {consistency_weight:.4f} gamma {threshold.item():.4f}"
                )
                # The region-partition occupancies are recorded here and not
                # in log.txt: they show what fraction of pixels lands in the
                # low- and mixed-confidence regions over training, which is
                # what gamma_c changes. Reducing the masks costs a device
                # sync, so it happens only on logging iterations.
                metrics.log(
                    iter_num, "train",
                    loss=loss.item(),
                    sup1=loss_sup1.item(), sup2=loss_sup2.item(),
                    pro1=loss_pro1.detach().item(), pro2=loss_pro2.detach().item(),
                    focus1=loss_focus1.detach().item(), focus2=loss_focus2.detach().item(),
                    consistency_weight=consistency_weight,
                    gamma=threshold.item(),
                    temperature=temperature_at(iter_num, args),
                    frac_low_con=low_con_mask.float().mean().item(),
                    frac_high_low1=high_low_con_mask1.float().mean().item(),
                    frac_high_low2=high_low_con_mask2.float().mean().item(),
                )
                if gamma_c is not None:
                    # Per-class thresholds and per-class low-confidence
                    # occupancy: the primary evidence for whether gamma_c
                    # changes which pixels the modules act on.
                    metrics.log(iter_num, "train",
                                **{f"gamma_c{c}": gamma_c[c].item() for c in range(num_classes)})
                    metrics.log(iter_num, "train",
                                **{f"frac_low_con_c{c}":
                                   (low_con_mask * (cls_avg == c)).float().sum().item()
                                   / max(1, (cls_avg == c).sum().item())
                                   for c in range(num_classes)})

            if iter_num > 0 and iter_num % args.val_interval == 0:
                for tag, model, opt_best in (("1", model1, best_perf1), ("2", model2, best_perf2)):
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
                    if mean_dice > opt_best:
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
    # A suffix, not a separate token: --exp already carries the method name.
    variant_suffix = {(1, 1): "", (1, 0): "-cgcpcl", (0, 1): "-dtct", (0, 0): "-suponly"}[
        (int(bool(args.use_cgcpcl)), int(bool(args.use_dtct)))
    ]
    variant = args.exp + variant_suffix
    # Every non-default Part 2 setting (beta, T, gamma_c, width, DINOv2) must
    # be visible in the run name: otherwise a sweep run is indistinguishable
    # from the reported baseline and the figure scripts, which pick the
    # newest run per method, would silently swap it in.
    if abs(args.proportion - 0.8) > 1e-9:
        variant += f"-b{round(args.proportion * 100):02d}"
    if args.temperature_schedule != "constant":
        variant += f"-t{args.temperature_schedule[:3]}"
    elif abs(args.temperature - 0.1) > 1e-9:
        variant += f"-t{round(args.temperature * 100):02d}"
    if args.use_class_threshold:
        # "all" keeps the original suffixes so existing run names stay
        # stable; "foreground" adds "fg".
        variant += {"offset": "-gammac", "quantile": "-gammacq",
                    "inverse": f"-gammaci{round(args.class_threshold_alpha * 100):02d}"}[args.class_threshold_mode]
        if args.class_threshold_scope == "foreground":
            variant += "fg"
    _exp = variant + (f"-w{args.width}" if args.width != 16 else "")
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
    logging.info(f"variant: {variant} (cgcpcl={bool(args.use_cgcpcl)}, dtct={bool(args.use_dtct)}, "
                 f"class_threshold={bool(args.use_class_threshold)}"
                 f"{'/' + args.class_threshold_mode + '/' + args.class_threshold_scope if args.use_class_threshold else ''})")
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
        f"--model unet_fea_aux "
        f"--save-predictions {os.path.join(snapshot_path, 'test_predictions')}"
    )


if __name__ == "__main__":
    main()
