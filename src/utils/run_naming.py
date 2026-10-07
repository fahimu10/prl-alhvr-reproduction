"""Single source of truth for run-directory names.

    <method>_<pct>_bs<batch_size>_seed<seed>_<YYYYmmdd-HHMMSS>
    e.g. alhvr-cgcpcl_10pct_bs16_seed1337_20260818-234610

Method first so `ls` groups a method's runs together and `outputs/alhvr-*`
selects the ablations; timestamp last so repeated runs of one config never
overwrite each other. The analysis scripts parse these names to group runs
by method and seed, so all three trainers must agree on the format.
"""
import time


def make_run_name(method, pct, batch_size, seed, timestamp=None):
    """`method` should already include any ablation suffix, e.g. 'alhvr-cgcpcl'."""
    ts = timestamp or time.strftime("%Y%m%d-%H%M%S")
    return f"{method}_{pct}_bs{batch_size}_seed{seed}_{ts}"
