import torch
from torch import nn


def get_lr_schedule_lambda(cfg):
    if cfg._target_ == "exponential":
        return lambda epoch: cfg.gamma**epoch
    elif cfg._target_ == "constant":
        return lambda _: 1
    elif cfg._target_ == "warmup_then":
        after = get_lr_schedule_lambda(cfg.after_warmup)
        s, t, n = cfg.warmup_start, cfg.warmup_end, cfg.warmup_steps

        def sched(epoch):
            if epoch < cfg.warmup_steps:
                # log linear warmup
                # tgt = s * ((t / s) ** (epoch / n))
                tgt = s + (t - s) * (epoch / n)
            else:
                tgt = after(epoch)
            return tgt / cfg.lr

        return sched

    else:
        raise ValueError(f"{cfg._target_} lr scheduler not found")
