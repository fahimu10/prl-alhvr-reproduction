"""Consistency-weight ramp-up, ported from ALHVR's utils/ramps.py.

Only sigmoid_rampup is used: train_ALHVR_acdc.py's
get_current_consistency_weight calls it to realise the paper's Gaussian
warm-up lambda(t) = 0.1 * exp(-5 * (1 - t/t_max)^2) (Eq. 10, §4.2).
Original file carries its own CC BY-NC 4.0 header from Curious AI Ltd.
"""
import numpy as np


def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    current = np.clip(current, 0.0, rampup_length)
    phase = 1.0 - current / rampup_length
    return float(np.exp(-5.0 * phase * phase))
