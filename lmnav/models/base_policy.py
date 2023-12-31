import pdb
from habitat_baselines.rl.ddppo.ddp_utils import rank0_only
from omegaconf import OmegaConf, omegaconf
import torch
from torch import nn
from hydra.utils import get_class, instantiate

from lmnav.config.default_structured_configs import BaseModelConfig


def load_policy_artifact(writer, store, cfg):
    """load a policy through artifact. should be used by wrapping in a partial fn"""
    if store is None:
        ckpt_path = writer.load_model(cfg)
    else:
        if rank0_only():
            ckpt_path = writer.load_model(cfg)
            store.set("policy_ckpt", ckpt_path)
        else:
            store.wait(["policy_ckpt"])
            ckpt_path = store.get("policy_ckpt").decode("utf-8")
        # remove key after all processes consume; important in case this key is
        # used again.
        torch.distributed.barrier()
        store.delete_key("policy_ckpt")

    print(f"Loading policy ({cfg.name}:{cfg.version}) from config: {ckpt_path}")
    ckpt_state_dict = torch.load(ckpt_path, map_location="cpu")

    return ckpt_state_dict


def instantiate_model(cfg, load_ckpts=True, writer=None, store=None):

    # first check if cfg is a model
    assert isinstance(cfg, omegaconf.DictConfig) and cfg.get("is_model", False)

    # legacy config changes
    if cfg.get("freeze_vit", None) is not None:
        cfg["freeze_backbone"] = cfg["freeze_vit"]

    ckpt_state_dict = None
    if cfg.load_artifact is not None:
        # todo: figure out how to merge configs
        ckpt_state_dict = load_policy_artifact(writer, store, cfg.load_artifact)
        if cfg.use_artifact_config:
            cfg = OmegaConf.create(ckpt_state_dict["config"]).train.policy

    # recursively instantiate all models
    inputs = {}
    for k, v in cfg.items():
        if (
            not isinstance(v, omegaconf.DictConfig)
            or cfg[k].get("_target_", None) is None
        ):
            inputs[k] = v
            continue
        if v.get("is_model", False):
            inputs[k] = instantiate_model(
                v, load_ckpts=load_ckpts, writer=writer, store=store
            )
        else:
            inputs[k] = instantiate(v)

    try:
        model = get_class(cfg._target_)(**inputs)
    except Exception as e:
        print(f"An error ocurred while loading the model: {e}")
        import pdb

        pdb.set_trace()

    if ckpt_state_dict is not None:
        state_dict = {
            k[len("module.") :]: v for k, v in ckpt_state_dict["model"].items()
        }
        model.load_state_dict(state_dict, strict=False)

    return model
