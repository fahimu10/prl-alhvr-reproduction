"""Report figures for the Part 1 reproduction.

Produces PDF (for LaTeX) and PNG (for quick viewing) into reports/figures/:

  fig1_val_curves.pdf   validation Dice over training, per method
  fig2_table2.pdf       our Table 2 result against the paper's targets
  fig3_per_class_hd.pdf per-class 95HD

Run from the repo root:
    python -m src.evaluation.make_figures

Reads validation curves from each run's metrics.csv (see src/utils/metrics.py)
and final metrics from each run's results.json (written by evaluate_acdc.py).
Nothing measured is transcribed into this file, so the figures cannot drift
from the runs they claim to describe. Only PAPER - the published Table 2
targets we are comparing against - is a constant here.

Design notes: method identity is fixed to one colour across every figure
(a reader who learns "ALHVR is green" keeps that), the paper's target is
drawn as a reference marker rather than a second bar series, and no chart
uses two y-scales.
"""
import argparse
import json
import os
from glob import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.utils.metrics import read_metrics

# --- palette: categorical slots 1-3, fixed per method, never recycled -------
INK, MUTED, GRID, AXIS = "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7"
METHODS = {                                   # label -> (colour, run-dir prefix)
    "U-Net baseline": ("#2a78d6", "unet_baseline"),
    "CPS":            ("#eb6834", "cps"),
    "ALHVR":          ("#1baf7a", "alhvr_"),
}
PAPER = {  # Table 2, ACDC 10% labelled
    "U-Net baseline": dict(dice=81.59, jaccard=70.74, hd95=8.07, asd=2.35),
    "CPS":            dict(dice=85.34, jaccard=75.50, hd95=8.78, asd=2.37),
    "ALHVR":          dict(dice=90.56, jaccard=83.21, hd95=2.57, asd=0.78),
}
CLASS_NAMES = ["RV", "Myocardium", "LV"]


def load_results(run_dir):
    """Our measured metrics for a run, from its results.json.

    Dice and Jaccard are stored as fractions and scaled to percent to match
    the paper's Table 2; 95HD and ASD are absolute distances already.
    """
    path = os.path.join(run_dir, "results.json")
    if not os.path.exists(path):
        return None
    r = json.load(open(path))
    mean = r["mean_across_classes"]
    per_class = r["per_class_mean"]
    return {
        "summary": dict(dice=100 * mean["dice"], jaccard=100 * mean["jaccard"],
                        hd95=mean["hd95"], asd=mean["asd"]),
        "per_class_hd": [per_class[str(c)]["hd95"] for c in (1, 2, 3)],
    }


def style():
    plt.rcParams.update({
        "figure.dpi": 140, "savefig.dpi": 300, "savefig.bbox": "tight",
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "legend.fontsize": 8.5, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "grid.color": GRID, "grid.linewidth": 0.6,
        "legend.frameon": False, "figure.facecolor": "white",
    })


def val_dice_curve(run_dir):
    """(iterations, mean val Dice) for a run, from its metrics.csv.

    For the two-network methods this is model1, matching the network the
    reported checkpoint is selected from.
    """
    series = read_metrics(os.path.join(run_dir, "metrics.csv"))
    for tag in ("val/mean_dice", "val/model1/mean_dice"):
        if tag in series:
            return series[tag]
    raise KeyError(f"{run_dir}/metrics.csv has no validation Dice series "
                   f"(tags: {sorted(series)})")


def find_runs(outputs, seed):
    """The run to plot per method: given seed, then most complete, newest.

    Deterministic on every machine: the seed is named explicitly, candidates
    are sorted, and equal-length runs of one config tie-break to the newest
    (the earlier is a superseded attempt). Ties are never broken on best
    validation Dice, which would cherry-pick the result.
    """
    picked = {}
    for label, (_, prefix) in METHODS.items():
        candidates = []
        for d in sorted(glob(os.path.join(outputs, f"{prefix}*/"))):
            csv_path = os.path.join(d, "metrics.csv")
            if f"_seed{seed}_" not in d or not os.path.exists(csv_path):
                continue
            n = len(val_dice_curve(d.rstrip("/"))[0])
            candidates.append((n, d.rstrip("/")))
        if candidates:
            picked[label] = max(candidates)[1]
    return picked


def fig_val_curves(runs, out):
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.grid(axis="y", zorder=0)
    for label, (colour, _) in METHODS.items():
        if label not in runs:
            continue
        it, dice = val_dice_curve(runs[label])
        ax.plot(it, dice, color=colour, lw=0.7, alpha=0.25, zorder=2)
        k = 9                                 # rolling mean over the noise
        if len(dice) > k:
            sm = np.convolve(dice, np.ones(k) / k, mode="valid")
            ax.plot(it[k - 1:], sm, color=colour, lw=2, label=label, zorder=3)
        else:
            ax.plot(it, dice, color=colour, lw=2, label=label, zorder=3)
    ax.set_xlabel("training iteration")
    ax.set_ylabel("validation Dice")
    ax.set_title("Validation Dice over training (ACDC, 10% labelled)", loc="left")
    # Zoom to where the curves live; a 0-1 axis hides the separation.
    ax.set_ylim(0.55, 0.95)
    ax.legend(loc="lower right")
    fig.savefig(out + ".pdf"); fig.savefig(out + ".png"); plt.close(fig)


def fig_table2(results, out):
    """Deviation from the paper's value, against the +-5% acceptance band.

    Plotted as deviation rather than absolute values: on a zero-based axis
    the three methods' Dice scores are nearly indistinguishable, and this
    puts all four metrics on one scale with the acceptance criterion as a
    visible band.
    """
    metrics = [("dice", "Dice"), ("jaccard", "Jaccard"),
               ("hd95", "95HD"), ("asd", "ASD")]
    labels = [m for m in METHODS if m in results]
    if not labels:
        return False
    fig, ax = plt.subplots(figsize=(6.6, 3.6))

    ax.axhspan(-5, 5, color="#1baf7a", alpha=0.10, zorder=1)
    ax.axhline(0, color=AXIS, lw=1, zorder=2)
    ax.grid(axis="y", zorder=0)

    x, w = np.arange(len(metrics)), 0.24
    for i, m in enumerate(labels):
        ours = results[m]["summary"]
        devs = [100 * (ours[k] - PAPER[m][k]) / PAPER[m][k] for k, _ in metrics]
        ax.bar(x + (i - (len(labels) - 1) / 2) * (w + 0.03), devs, width=w,
               color=METHODS[m][0], label=m, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels([n for _, n in metrics])
    ax.set_ylabel("deviation from paper (%)")
    ax.set_title("Reproduction accuracy: deviation from the paper's Table 2",
                 loc="left")
    ax.text(0.02, 0.945, "shaded band = ±5% acceptance criterion",
            transform=ax.transAxes, fontsize=8, color=MUTED)
    ax.text(0.02, 0.885,
            "95HD / ASD: lower is better, so negative is favourable",
            transform=ax.transAxes, fontsize=8, color=MUTED)
    ax.legend(loc="lower left", ncol=3)
    ax.margins(y=0.20)
    fig.savefig(out + ".pdf"); fig.savefig(out + ".png"); plt.close(fig)
    return True


def fig_per_class_hd(results, out):
    labels = [m for m in METHODS if m in results]
    if not labels:
        return False
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.grid(axis="y", zorder=0)
    x, w = np.arange(len(CLASS_NAMES)), 0.24
    for i, m in enumerate(labels):
        ax.bar(x + (i - (len(labels) - 1) / 2) * (w + 0.03), results[m]["per_class_hd"], width=w,
               color=METHODS[m][0], label=m, zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(CLASS_NAMES)
    ax.set_ylabel("95HD (lower is better)")
    ax.set_title("Per-class boundary error: LV fragmentation is what ALHVR fixes",
                 loc="left")
    ax.legend()
    ax.margins(y=0.16)
    fig.savefig(out + ".pdf"); fig.savefig(out + ".png"); plt.close(fig)
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outputs", default="outputs")
    ap.add_argument("--figdir", default="reports/figures")
    ap.add_argument("--seed", type=int, default=1337,
                    help="which seed's run to plot (the reported seed)")
    args = ap.parse_args()

    style()
    os.makedirs(args.figdir, exist_ok=True)
    runs = find_runs(args.outputs, args.seed)
    print("using runs:")
    for k, v in runs.items():
        print(f"  {k:16s} {v}")

    results = {}
    for label, d in runs.items():
        r = load_results(d)
        if r is None:
            print(f"  WARNING {label}: no results.json - run evaluate_acdc.py; "
                  f"fig2/fig3 will omit this method")
        else:
            results[label] = r

    fig_val_curves(runs, os.path.join(args.figdir, "fig1_val_curves"))
    ok2 = fig_table2(results, os.path.join(args.figdir, "fig2_table2"))
    ok3 = fig_per_class_hd(results, os.path.join(args.figdir, "fig3_per_class_hd"))
    print(f"\nwrote fig1{', fig2' if ok2 else ''}{', fig3' if ok3 else ''} "
          f"(pdf + png) to {args.figdir}/")


if __name__ == "__main__":
    main()
