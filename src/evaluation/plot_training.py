"""Training-behaviour figures, built from each run's metrics.csv.

Two kinds of output, written to reports/figures/training/:

  training_<run>.pdf      per-run diagnostics sheet - what one run did
  compare_*.pdf           the same quantity across methods, side by side

These are diagnostics, not report figures. They answer "is this training
healthy, and what is it doing?" - the report figures in make_figures.py
answer "did it reproduce?".

Run from the repo root, after training:
    python -m src.evaluation.plot_training
    python -m src.evaluation.plot_training --run outputs/alhvr_.../

Input is metrics.csv only (src/utils/metrics.py). Panels whose series are
absent from a run are skipped rather than drawn empty, so the same script
handles the single-network baseline, two-network CPS, and ALHVR with its
extra threshold and region-occupancy series.

Design notes
------------
- Method colour is fixed and matches make_figures.py, so it means the same
  thing in every figure.
- Per-class comparisons use one panel per class rather than a second colour
  ramp, so class is encoded by position and method keeps the hue.
- In single-run loss panels hue encodes the loss component and line style
  the network (model1 solid, model2 dashed) - a composite encoding rather
  than eight competing hues. Component hues are Okabe-Ito, CVD-safe.
- No panel uses two y-scales; log y is used where the range demands it.
"""
import argparse
import csv
import os
from glob import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.utils.metrics import read_metrics

INK, MUTED, GRID, AXIS = "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7"

METHODS = {                                   # label -> (colour, run-dir prefix)
    "U-Net baseline": ("#2a78d6", "unet_baseline"),
    "CPS":            ("#eb6834", "cps"),
    "ALHVR":          ("#1baf7a", "alhvr_"),
}
# Okabe-Ito, colourblind-safe by construction.
COMPONENT = {
    "loss":    "#0072b2",   # blue      - total
    "sup":     "#d55e00",   # vermillion- supervised
    "cps":     "#009e73",   # green     - cross pseudo supervision
    "pro":     "#cc79a7",   # purple    - CG-CPCL prototype term
    "focus":   "#e69f00",   # orange    - DTCT term
    "dice_term": "#009e73",
    "ce_term":   "#cc79a7",
}
CLASS_NAMES = {1: "RV", 2: "Myocardium", 3: "LV"}


def style():
    plt.rcParams.update({
        "figure.dpi": 130, "savefig.dpi": 300, "savefig.bbox": "tight",
        "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
        "legend.fontsize": 7.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "grid.color": GRID, "grid.linewidth": 0.6,
        "legend.frameon": False, "figure.facecolor": "white",
    })


def smooth(y, k=9):
    """Rolling mean, returned with the offset needed to align x."""
    if len(y) <= k:
        return None, 0
    return np.convolve(y, np.ones(k) / k, mode="valid"), k - 1


def net_series(series, base):
    """[(network label, line style, (steps, values)), ...] for a metric.

    Handles the three shapes uniformly: an unsuffixed series (baseline),
    model1/model2 pairs (CPS, ALHVR), and absence.
    """
    out = []
    if base in series:
        out.append(("", "-", series[base]))
    for n, ls in (("1", "-"), ("2", "--")):
        tag = base.replace("val/", f"val/model{n}/") if base.startswith("val/") else f"{base}{n}"
        if tag in series:
            out.append((f"model{n}", ls, series[tag]))
    return out


# --- panels -----------------------------------------------------------------
# Each returns False when the run has no data for it, so the sheet adapts to
# the method instead of showing empty axes.

def panel_losses(ax, s):
    drawn = False
    for base, label in (("train/loss", "total"), ("train/sup", "supervised"),
                        ("train/cps", "cross pseudo-sup"), ("train/pro", "CG-CPCL"),
                        ("train/focus", "DTCT"), ("train/dice_term", "Dice term"),
                        ("train/ce_term", "CE term")):
        key = base.split("/")[-1]
        for net, ls, (x, y) in net_series(s, base):
            sm, off = smooth(y, 15)
            ax.plot(x, y, color=COMPONENT[key], lw=0.5, alpha=0.18, zorder=2)
            ax.plot(x[off:] if sm is not None else x, sm if sm is not None else y,
                    color=COMPONENT[key], lw=1.5, ls=ls, zorder=3,
                    label=f"{label} {net}".strip())
            drawn = True
    if not drawn:
        return False
    ax.set_yscale("symlog", linthresh=0.1)
    ax.set_xlabel("iteration"); ax.set_ylabel("loss (symlog)")
    ax.set_title("Training loss components", loc="left")
    ax.legend(ncol=2, loc="upper right")
    return True


def panel_val_dice(ax, s):
    series = net_series(s, "val/mean_dice")
    if not series:
        return False
    for net, ls, (x, y) in series:
        ax.plot(x, y, color="#0072b2", lw=0.6, alpha=0.25, zorder=2)
        sm, off = smooth(y)
        ax.plot(x[off:] if sm is not None else x, sm if sm is not None else y,
                color="#0072b2", lw=1.8, ls=ls, zorder=3,
                label=f"{net} (smoothed)".strip())
        ax.plot(x, np.maximum.accumulate(y), color="#d55e00", lw=1.2, ls=ls,
                zorder=4, label=f"{net} best so far".strip())
        # The reported number is a best-checkpoint score, so the running
        # max - not the final value - is what training delivers.
        ax.annotate(f"{np.max(y):.4f}", xy=(x[np.argmax(y)], np.max(y)),
                    xytext=(-4, 6), textcoords="offset points",
                    fontsize=7.5, color="#d55e00", ha="right")
    ax.set_xlabel("iteration"); ax.set_ylabel("validation Dice")
    ax.set_title("Validation Dice + running best", loc="left")
    ax.legend(loc="lower right")
    return True


def panel_val_per_class(ax, s, metric="dice"):
    drawn = False
    for c, name in CLASS_NAMES.items():
        for net, ls, (x, y) in net_series(s, f"val/{metric}_c{c}"):
            if net == "model2":
                continue                     # model1 only; model2 doubles the ink
            colour = ["#0072b2", "#d55e00", "#009e73"][c - 1]
            sm, off = smooth(y)
            ax.plot(x, y, color=colour, lw=0.5, alpha=0.2, zorder=2)
            ax.plot(x[off:] if sm is not None else x, sm if sm is not None else y,
                    color=colour, lw=1.6, zorder=3, label=name)
            drawn = True
    if not drawn:
        return False
    ax.set_xlabel("iteration")
    if metric == "dice":
        ax.set_ylabel("validation Dice")
        ax.set_title("Per-class Dice (model1)", loc="left")
        ax.legend(loc="lower right")
    else:
        ax.set_ylabel("95HD (log)"); ax.set_yscale("log")
        ax.set_title("Per-class 95HD (model1, log scale)", loc="left")
        ax.legend(loc="upper right")
    return True


def panel_schedule(ax, s):
    if "train/consistency_weight" not in s:
        return False
    x, y = s["train/consistency_weight"]
    ax.plot(x, y, color="#0072b2", lw=1.8)
    ax.set_xlabel("iteration"); ax.set_ylabel("consistency weight")
    ax.set_title("Consistency ramp-up schedule", loc="left")
    ax.annotate(f"plateau {y[-1]:.3f}", xy=(x[-1], y[-1]), xytext=(-6, -12),
                textcoords="offset points", fontsize=7.5, color=MUTED, ha="right")
    return True


def panel_gamma(ax, s):
    if "train/gamma" not in s:
        return False
    x, y = s["train/gamma"]
    ax.plot(x, y, color="#cc79a7", lw=0.6, alpha=0.3, zorder=2)
    sm, off = smooth(y, 15)
    if sm is not None:
        ax.plot(x[off:], sm, color="#cc79a7", lw=1.8, zorder=3)
    ax.set_xlabel("iteration"); ax.set_ylabel(r"adaptive threshold $\gamma$")
    ax.set_title(r"$\gamma$ trajectory - the single global cut Part 2 replaces",
                 loc="left")
    ax.annotate(f"{y[0]:.3f} → {y[-1]:.3f}", xy=(0.98, 0.06),
                xycoords="axes fraction", fontsize=7.5, color=MUTED, ha="right")
    return True


def panel_occupancy(ax, s):
    tags = [("train/frac_low_con", "both networks low", "#0072b2"),
            ("train/frac_high_low1", "net1 low, net2 high", "#d55e00"),
            ("train/frac_high_low2", "net2 low, net1 high", "#009e73")]
    if not any(t in s for t, _, _ in tags):
        return False
    for tag, label, colour in tags:
        if tag not in s:
            continue
        x, y = s[tag]
        sm, off = smooth(y, 15)
        ax.plot(x, y, color=colour, lw=0.5, alpha=0.2, zorder=2)
        ax.plot(x[off:] if sm is not None else x, sm if sm is not None else y,
                color=colour, lw=1.6, zorder=3, label=label)
    ax.set_xlabel("iteration"); ax.set_ylabel("fraction of unlabeled pixels")
    ax.set_title("Region-partition occupancy", loc="left")
    ax.legend(loc="upper right")
    return True


def panel_time(ax, csv_path):
    """Iterations against wall clock, from the recorded time_s column.

    This is what you need when sizing a Slurm --time request, and a run
    that slows down partway shows up as a bend here rather than as a job
    that silently hit its limit.
    """
    steps, secs = [], []
    with open(csv_path, newline="") as fh:
        for row in csv.DictReader(fh):
            if row["time_s"]:
                steps.append(float(row["step"])); secs.append(float(row["time_s"]))
    if not steps:
        return False
    steps, secs = np.array(steps), np.array(secs)
    order = np.argsort(secs)
    steps, secs = steps[order], secs[order]
    ax.plot(secs / 3600.0, steps, color="#0072b2", lw=1.8)
    ax.set_xlabel("wall clock (hours)"); ax.set_ylabel("iteration")
    ax.set_title("Progress vs wall clock", loc="left")
    rate = steps[-1] / (secs[-1] / 3600.0) if secs[-1] > 0 else 0
    ax.annotate(f"{rate:,.0f} iter/h\ntotal {secs[-1]/3600:.2f} h",
                xy=(0.03, 0.80), xycoords="axes fraction", fontsize=7.5, color=MUTED)
    return True


def run_sheet(run_dir, out_path):
    """Multi-panel diagnostics for one run; panels without data are dropped."""
    csv_path = os.path.join(run_dir, "metrics.csv")
    s = read_metrics(csv_path)

    panels = [("losses", lambda ax: panel_losses(ax, s)),
              ("val_dice", lambda ax: panel_val_dice(ax, s)),
              ("per_class_dice", lambda ax: panel_val_per_class(ax, s, "dice")),
              ("per_class_hd", lambda ax: panel_val_per_class(ax, s, "hd95")),
              ("schedule", lambda ax: panel_schedule(ax, s)),
              ("gamma", lambda ax: panel_gamma(ax, s)),
              ("occupancy", lambda ax: panel_occupancy(ax, s)),
              ("time", lambda ax: panel_time(ax, csv_path))]

    # Probe which panels have data on a throwaway figure, so the grid is
    # sized correctly before anything is drawn for real.
    probe = plt.figure()
    live = []
    for name, fn in panels:
        ax = probe.add_subplot(111)
        if fn(ax) is True:
            live.append((name, fn))
        probe.clf()
    plt.close(probe)

    ncol = 2
    nrow = int(np.ceil(len(live) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(11, 3.1 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, (_, fn) in zip(axes, live):
        ax.grid(axis="y", zorder=0)
        fn(ax)
    for ax in axes[len(live):]:
        ax.axis("off")

    fig.suptitle(f"Training diagnostics — {os.path.basename(run_dir)}",
                 x=0.008, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(out_path + ".pdf"); fig.savefig(out_path + ".png")
    plt.close(fig)
    return [n for n, _ in live]


# --- cross-method comparisons ----------------------------------------------

def compare_curve(runs, out_path, tag_options, ylabel, title, logy=False):
    fig, ax = plt.subplots(figsize=(6.4, 3.5))
    ax.grid(axis="y", zorder=0)
    drawn = False
    for label, (colour, _) in METHODS.items():
        if label not in runs:
            continue
        s = read_metrics(os.path.join(runs[label], "metrics.csv"))
        for tag in tag_options:
            if tag in s:
                x, y = s[tag]
                ax.plot(x, y, color=colour, lw=0.6, alpha=0.22, zorder=2)
                sm, off = smooth(y)
                ax.plot(x[off:] if sm is not None else x, sm if sm is not None else y,
                        color=colour, lw=2, zorder=3, label=label)
                drawn = True
                break
    if not drawn:
        plt.close(fig)
        return False
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("training iteration"); ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
    ax.legend(loc="best")
    fig.savefig(out_path + ".pdf"); fig.savefig(out_path + ".png")
    plt.close(fig)
    return True


def compare_per_class(runs, out_path, metric, ylabel, title, logy=False):
    """One panel per class - class is encoded by position, method by colour."""
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), sharex=True,
                             sharey=not logy)
    drawn = False
    for ax, (c, name) in zip(axes, CLASS_NAMES.items()):
        ax.grid(axis="y", zorder=0)
        for label, (colour, _) in METHODS.items():
            if label not in runs:
                continue
            s = read_metrics(os.path.join(runs[label], "metrics.csv"))
            for tag in (f"val/{metric}_c{c}", f"val/model1/{metric}_c{c}"):
                if tag in s:
                    x, y = s[tag]
                    sm, off = smooth(y)
                    ax.plot(x, y, color=colour, lw=0.5, alpha=0.18, zorder=2)
                    ax.plot(x[off:] if sm is not None else x,
                            sm if sm is not None else y,
                            color=colour, lw=1.8, zorder=3, label=label)
                    drawn = True
                    break
        if logy:
            ax.set_yscale("log")
        ax.set_title(name, loc="left")
        ax.set_xlabel("iteration")
    if not drawn:
        plt.close(fig)
        return False
    axes[0].set_ylabel(ylabel)
    axes[-1].legend(loc="best")
    fig.suptitle(title, x=0.008, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path + ".pdf"); fig.savefig(out_path + ".png")
    plt.close(fig)
    return True


def find_runs(outputs, seed):
    """Newest run per method for the given seed, deterministic.

    Newest wins because these figures describe the training you just did;
    an older run of the same config is a superseded attempt.
    """
    picked = {}
    for label, (_, prefix) in METHODS.items():
        candidates = sorted(
            d.rstrip("/") for d in glob(os.path.join(outputs, f"{prefix}*/"))
            if f"_seed{seed}_" in d and os.path.exists(os.path.join(d, "metrics.csv")))
        if candidates:
            picked[label] = candidates[-1]
    return picked


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outputs", default="outputs")
    ap.add_argument("--figdir", default="reports/figures/training")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--run", action="append", default=None,
                    help="plot this run directory (repeatable); default is the "
                         "newest run per method for --seed")
    args = ap.parse_args()

    style()
    os.makedirs(args.figdir, exist_ok=True)

    if args.run:
        runs = {os.path.basename(r.rstrip("/")): r.rstrip("/") for r in args.run}
    else:
        runs = find_runs(args.outputs, args.seed)
    if not runs:
        raise SystemExit(f"no runs with metrics.csv under {args.outputs} "
                         f"for seed {args.seed}")

    print("per-run diagnostics sheets:")
    for label, d in runs.items():
        name = os.path.basename(d)
        drawn = run_sheet(d, os.path.join(args.figdir, f"training_{name}"))
        print(f"  {label:16s} {name}\n{'':18s}panels: {', '.join(drawn)}")

    if not args.run:
        print("\ncross-method comparisons:")
        made = [
            ("compare_val_dice", compare_curve(
                runs, os.path.join(args.figdir, "compare_val_dice"),
                ("val/mean_dice", "val/model1/mean_dice"), "validation Dice",
                "Validation Dice over training")),
            ("compare_val_hd95", compare_curve(
                runs, os.path.join(args.figdir, "compare_val_hd95"),
                ("val/mean_hd95", "val/model1/mean_hd95"), "95HD (log)",
                "Validation 95HD over training", logy=True)),
            ("compare_sup_loss", compare_curve(
                runs, os.path.join(args.figdir, "compare_sup_loss"),
                ("train/sup1", "train/loss"), "supervised loss",
                "Supervised loss - the labeled-data fit each method achieves")),
            ("compare_per_class_dice", compare_per_class(
                runs, os.path.join(args.figdir, "compare_per_class_dice"),
                "dice", "validation Dice", "Per-class validation Dice by method")),
            ("compare_per_class_hd95", compare_per_class(
                runs, os.path.join(args.figdir, "compare_per_class_hd95"),
                "hd95", "95HD (log)", "Per-class validation 95HD by method",
                logy=True)),
        ]
        for name, ok in made:
            print(f"  {'ok  ' if ok else 'skip'} {name}")

    print(f"\nwrote figures (pdf + png) to {args.figdir}/")


if __name__ == "__main__":
    main()
