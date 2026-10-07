"""Final test-set evaluation: Dice, Jaccard, 95HD, ASD per class + mean.

Ported from ALHVR's test_acdc.py (same metrics, same per-class loop, same
zoom-to-256-then-back inference pattern), adapted to evaluate either
backbone (`--model unet` for the U-Net baseline and CPS, `unet_fea_aux`
for ALHVR) and for our own dataset layout (data_list/test.list,
per-volume .h5 at the root).

This is the script that produces the numbers to compare against the
paper's reported values (the acceptance criterion is within +/-5% of
these). Training-time validation (src/evaluation/val_2d.py) only tracks
Dice+HD95 for checkpoint selection - this is the full metric set, run
once against the held-out test split after training.

Run from the repo root:
    python -m src.evaluation.evaluate_acdc --root-path data/acdc_processed/data --checkpoint outputs/<run>/best_model.pth
"""
import argparse
import json
import os

import h5py
import numpy as np
import SimpleITK as sitk
import torch
from medpy import metric
from scipy.ndimage import zoom
from tqdm import tqdm

from src.models.unet import UNet, UNet_fea_aux
from src.utils.device import get_device

# Mirrors the reference's net_factory dispatch: test_acdc.py takes a
# --model flag, defaulting to the ALHVR backbone.
NET_FACTORY = {"unet": UNet, "unet_fea_aux": UNet_fea_aux}


def calculate_metric_percase(pred, gt):
    pred = pred.copy()
    gt = gt.copy()
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if np.sum(pred) == 0:
        return 0.0, 0.0, 0.0, 0.0
    dice = metric.binary.dc(pred, gt)
    jc = metric.binary.jc(pred, gt)
    hd95 = metric.binary.hd95(pred, gt)
    asd = metric.binary.asd(pred, gt)
    return dice, jc, hd95, asd


def test_single_volume(case, model, root_path, save_path, num_classes, device, patch_size=(256, 256)):
    with h5py.File(os.path.join(root_path, f"{case}.h5"), "r") as h5f:
        image = h5f["image"][:]
        label = h5f["label"][:]

    prediction = np.zeros_like(label)
    model.eval()
    for ind in range(image.shape[0]):
        slice_ = image[ind, :, :]
        x, y = slice_.shape
        slice_ = zoom(slice_, (patch_size[0] / x, patch_size[1] / y), order=0)
        inp = torch.from_numpy(slice_).unsqueeze(0).unsqueeze(0).float().to(device)
        with torch.no_grad():
            out_main = model(inp)
            if len(out_main) > 1:
                out_main = out_main[0]
            out = torch.argmax(torch.softmax(out_main, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            prediction[ind] = pred

    per_class_metrics = [
        calculate_metric_percase(prediction == c, label == c) for c in range(1, num_classes)
    ]

    if save_path is not None:
        for arr, suffix in ((image, "img"), (prediction, "pred"), (label, "gt")):
            itk_img = sitk.GetImageFromArray(arr.astype(np.float32))
            itk_img.SetSpacing((1, 1, 10))
            sitk.WriteImage(itk_img, os.path.join(save_path, f"{case}_{suffix}.nii.gz"))

    return per_class_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root-path", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num-classes", type=int, default=4)
    parser.add_argument("--width", type=int, default=None,
                        help="base channel width; read from the run's config.json "
                             "beside the checkpoint when omitted")
    parser.add_argument("--use-dino", type=int, default=None,
                        help="PART 2: 1 if the checkpoint carries the frozen DINOv2 "
                             "branch; read from the run's config.json when omitted")
    parser.add_argument("--model", choices=sorted(NET_FACTORY), default="unet",
                        help="'unet' for the supervised baseline, 'unet_fea_aux' for ALHVR checkpoints")
    parser.add_argument("--save-predictions", type=str, default=None, help="dir to write .nii.gz volumes to, omit to skip")
    parser.add_argument("--results-json", type=str, default=None,
                        help="where to write results.json; defaults to alongside the checkpoint")
    args = parser.parse_args()

    with open(os.path.join(args.root_path, "data_list", "test.list")) as f:
        cases = sorted(line.strip().split(".")[0] for line in f if line.strip())

    if args.save_predictions:
        os.makedirs(args.save_predictions, exist_ok=True)

    device = get_device()
    print(f"device: {device}")
    # Architecture must match the checkpoint or the state_dict load fails,
    # so width and the DINOv2 flag default to the run's own config.json.
    cfg_path = os.path.join(os.path.dirname(os.path.abspath(args.checkpoint)), "config.json")
    cfg = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {}
    width = args.width if args.width is not None else cfg.get("width", 16)
    use_dino = bool(args.use_dino if args.use_dino is not None else cfg.get("use_dino", 0))
    # dino_pretrained=False: the frozen DINOv2 weights are already inside the
    # checkpoint, so evaluation needs no timm/HF download and works offline.
    model = NET_FACTORY[args.model](in_chns=1, class_num=args.num_classes, width=width,
                                    use_dino=use_dino, dino_pretrained=False).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    print(f"loaded checkpoint: {args.checkpoint} (arch: {args.model})")

    num_fg_classes = args.num_classes - 1
    metric_names = ["dice", "jaccard", "hd95", "asd"]

    # Keep every case, not just the running total. Per-case values make a
    # PAIRED comparison between methods possible (same 40 volumes, n=40),
    # which is far more sensitive than comparing means across a handful of
    # seeds - and they cannot be recovered without re-running evaluation.
    per_case = {}
    for case in tqdm(cases):
        m = np.asarray(test_single_volume(
            case, model, args.root_path, args.save_predictions, args.num_classes, device))
        per_case[case] = m  # (num_fg_classes, 4)

    totals = np.sum(list(per_case.values()), axis=0)
    means = totals / len(cases)  # (num_fg_classes, [dice, jc, hd95, asd])

    print(f"\nper-class metrics (n={len(cases)} test volumes):")
    for c in range(num_fg_classes):
        row = ", ".join(f"{name}={means[c, i]:.4f}" for i, name in enumerate(metric_names))
        print(f"  class {c + 1}: {row}")

    overall = means.mean(axis=0)
    print("\nmean across classes:")
    for i, name in enumerate(metric_names):
        print(f"  {name}: {overall[i]:.4f}")

    # results.json is written ALWAYS (not gated on --save-predictions) and next
    # to the checkpoint by default, so every evaluated run leaves a
    # machine-readable record.
    results_path = args.results_json or os.path.join(
        os.path.dirname(os.path.abspath(args.checkpoint)), "results.json")
    payload = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "root_path": os.path.abspath(args.root_path),
        "model": args.model,
        "num_classes": args.num_classes,
        "n_test_volumes": len(cases),
        "metric_names": metric_names,
        "class_names": {"1": "RV", "2": "Myo", "3": "LV"},
        "per_class_mean": {str(c + 1): dict(zip(metric_names, means[c].tolist()))
                           for c in range(num_fg_classes)},
        "mean_across_classes": dict(zip(metric_names, overall.tolist())),
        "per_case": {case: {str(c + 1): dict(zip(metric_names, m[c].tolist()))
                            for c in range(num_fg_classes)}
                     for case, m in per_case.items()},
    }
    with open(results_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwrote {results_path}")

    if args.save_predictions:
        with open(os.path.join(args.save_predictions, "performance.txt"), "w") as f:
            f.write(f"checkpoint: {args.checkpoint}\n")
            f.write(f"per-class ({metric_names}):\n{means}\n")
            f.write(f"mean across classes ({metric_names}): {overall}\n")


if __name__ == "__main__":
    main()
