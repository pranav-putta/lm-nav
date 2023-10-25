import os

from lmnav.config.default_structured_configs import *
from lmnav.dataset.habitat_datasets import *

from hydra import initialize, compose


def get_config(experiment):
    overrides = f'+experiment="{experiment}"'

    with initialize(version_base=None, config_path="."):
        cfg = compose(config_name="default.yaml", overrides=[overrides])

    if cfg.exp.name is None:
        cfg.exp.name = os.path.basename(experiment)

    return cfg
