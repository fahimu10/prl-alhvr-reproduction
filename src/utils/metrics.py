"""Structured metric logging: one machine-readable record per scalar.

Each run directory gets, alongside the human-readable log.txt:

    metrics.csv   step,time_s,tag,value   - every scalar, as it is computed
    config.json   the full argparse namespace of the run

Figures read these files and never parse log.txt. Long format (one scalar
per row) because metrics arrive on different cadences - losses every 50
iterations, validation every val_interval, per-class values only at
validation - so a wide table would be mostly empty cells. Tags are
slash-namespaced ("train/loss", "val/model1/mean_dice") and pivot with:

    df.pivot(index="step", columns="tag", values="value")
"""
import csv
import json
import os
import time

import numpy as np

FIELDS = ("step", "time_s", "tag", "value")


class MetricsLogger:
    """Append-only CSV of (step, time_s, tag, value) scalar records.

    Flushes on every write: runs last hours on a shared cluster and can be
    killed by a time limit, and buffered rows would be lost precisely when
    the partial history is most wanted.
    """

    def __init__(self, path):
        self.path = path
        fresh = not os.path.exists(path) or os.path.getsize(path) == 0
        self._fh = open(path, "a", newline="")
        self._writer = csv.writer(self._fh)
        if fresh:
            self._writer.writerow(FIELDS)
            self._fh.flush()
        self._t0 = time.time()

    def log(self, step, prefix, **scalars):
        """Record scalars under `prefix`, e.g. log(50, "train", loss=1.9).

        Values may be floats or 0-dim tensors; pass .item(), never a batch.
        """
        elapsed = round(time.time() - self._t0, 2)
        for name, value in scalars.items():
            tag = f"{prefix}/{name}" if prefix else name
            self._writer.writerow((step, elapsed, tag, f"{float(value):.6g}"))
        self._fh.flush()

    def close(self):
        self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def write_config(path, args):
    """Persist the argparse namespace as JSON next to the metrics.

    Comparing runs across a sweep needs each run's configuration in the
    same machine-readable form as its metrics.
    """
    with open(path, "w") as fh:
        json.dump({k: v for k, v in vars(args).items()}, fh, indent=2, default=str)


def read_metrics(path):
    """Read metrics.csv into {tag: (steps, values)}, both float arrays.

    Sorted by step so callers can plot directly. Kept here rather than in
    the plotting scripts so every consumer reads the format the same way.
    """
    series = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            series.setdefault(row["tag"], ([], []))
            series[row["tag"]][0].append(float(row["step"]))
            series[row["tag"]][1].append(float(row["value"]))
    out = {}
    for tag, (steps, values) in series.items():
        steps, values = np.asarray(steps), np.asarray(values)
        order = np.argsort(steps, kind="stable")
        out[tag] = (steps[order], values[order])
    return out
