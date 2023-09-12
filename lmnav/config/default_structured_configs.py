from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore
    
from typing import Optional
from omegaconf import MISSING
import torch

### LOGGER CONFIGS ###j
@dataclass
class LoggerConfig:
    project: str = 'lmnav'
    name: Optional[str] = MISSING
    group: str = MISSING
    tags: Optional[list] = MISSING
    notes: Optional[str] = MISSING
    

@dataclass
class WBLoggerConfig(LoggerConfig):
    _target_: str = 'lmnav.common.writer.WandBWriter'
    resume_id: Optional[str] = MISSING

@dataclass
class ConsoleLoggerConfig(LoggerConfig):
    _target_: str = 'lmnav.common.writer.ConsoleWriter'

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
class EpisodeGeneratorConfig:
    num_episodes: int = MISSING
    artifact: str = MISSING
    dtg_threshold: float = 1.0
    actor: BaseActorConfig = MISSING
    

cs = ConfigStore.instance()
cs.store(group='exp', name='base', node=ExperimentConfig)

cs.store(group='logger', name='base', node=LoggerConfig)
cs.store(group='logger', name='wb', node=WBLoggerConfig)
cs.store(group='logger', name='console', node=ConsoleLoggerConfig)

cs.store(group='generator', name='base', node=EpisodeGeneratorConfig)

cs.store(group='actor', name='base', node=BaseActorConfig)
cs.store(group='actor', name='old_eai_policy', node=OldEAIPolicyActorConfig)


