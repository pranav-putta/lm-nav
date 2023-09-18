import torch
from torch import nn

def get_lr_schedule_lambda(cfg):
    if cfg._target_ == "exponential":
        return lambda epoch: cfg.gamma ** epoch
    elif cfg._target_ == "constant":
        return lambda _: 1
    else:
        raise ValueError(f"{cfg._target_} lr scheduler not found")


