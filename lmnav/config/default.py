import habitat
import habitat_baselines
from lmnav.config.default_structured_configs import *

from hydra import initialize, compose

def get_config(experiment):
    overrides = f"+experiment={experiment}"

    with initialize(version_base=None, config_path='.'):
        return compose(config_name='default.yaml', overrides=[overrides])
