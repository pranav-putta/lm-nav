from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore
    
from typing import Optional
from omegaconf import MISSING
import torch

### LOGGER CONFIGS ###j
@dataclass
class LoggerConfig:
    _target_: str = MISSING
    project: str = 'lmnav'
    name: Optional[str] = MISSING
    group: str = MISSING
    tags: Optional[list] = None
    notes: Optional[str] = None
    resume_id: Optional[str] = None
    

@dataclass
class WBLoggerConfig(LoggerConfig):
    _target_: str = 'wb'

@dataclass
class ConsoleLoggerConfig(LoggerConfig):
    _target_: str = 'console'

@dataclass
class ExperimentConfig:
    name: str = MISSING
    root_dir: str = 'experiments/'
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger: LoggerConfig = MISSING
    

### GENERATOR CONFIGS ###
@dataclass
class BaseActorConfig:
    name: str = MISSING

@dataclass
class OldEAIPolicyActorConfig(BaseActorConfig):
    name: str = "old_eai_policy"
    ckpt: str = ""


@dataclass
class BaseFilterMethodConfig:
    name: str = MISSING

@dataclass
class DTGFitlerMethodConfig(BaseFilterMethodConfig):
    name: str = "filter_methods.dtg"
    dtg_threshold: float = 1.0
    
@dataclass
class EpisodeGeneratorConfig:
    num_episodes: int = MISSING
    artifact: str = MISSING
    actor: BaseActorConfig = MISSING
    filter_method: BaseFilterMethodConfig = MISSING
    deterministic: bool = MISSING
    ckpt_freq: int = 1
    

cs = ConfigStore.instance()
cs.store(group='exp', name='base', node=ExperimentConfig)

cs.store(group='logger', name='base', node=LoggerConfig)
cs.store(group='logger', name='wb', node=WBLoggerConfig)
cs.store(group='logger', name='console', node=ConsoleLoggerConfig)

cs.store(group='generator', name='base', node=EpisodeGeneratorConfig)
cs.store(group='generator/filter_method', name='base', node=BaseFilterMethodConfig)
cs.store(group='generator/filter_method', name='dtg', node=DTGFitlerMethodConfig)

cs.store(group='actor', name='base', node=BaseActorConfig)
cs.store(group='actor', name='old_eai_policy', node=OldEAIPolicyActorConfig)


