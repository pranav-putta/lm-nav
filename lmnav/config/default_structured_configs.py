from dataclasses import dataclass 
from hydra.core.config_store import ConfigStore
    
from typing import Optional
from omegaconf import MISSING
import torch

### LOGGER CONFIGS ###
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

### ARTIFACT CONFIG ###
@dataclass
class ArtifactConfig:
    name: str = MISSING
    version: str = MISSING
    dirpath: str = MISSING
    

### POLICY CONFIGS ###
@dataclass
class BasePolicyConfig:
    _target_: str = MISSING

@dataclass
class OldEAIPolicyConfig(BasePolicyConfig):
    _target_: str = "old_eai_policy"

@dataclass
class BaseNavLLaMAPolicyConfig(BasePolicyConfig):
    _target_: str = "lin_nav_llama"
    model_type: str = "pretrain_vicuna"
    image_size: int = 224
    drop_path_rate: int = 0
    use_grad_checkpoint: bool = False
    vit_precision: str = "fp16"
    num_query_token: int = 32

    freeze_vit: bool = True
    freeze_qformer: bool = True
    freeze_lora: bool = True
    freeze_qformer_compression: bool = True
    freeze_llama_proj: bool = False
    low_resource: bool = False
    lora_config: dict = MISSING
    qformer_compressor_cfg: dict = MISSING

    llama_model: str = "meta-llama/Llama-2-7b-chat-hf"
    equip_audio_branch: bool = False
    vis_processor: dict = MISSING
    

### FILTER CONFIGS ###
@dataclass
class BaseFilterMethodConfig:
    _target_: str = MISSING

@dataclass
class DTGFitlerMethodConfig(BaseFilterMethodConfig):
    _target_: str = "filter_methods.dtg"
    dtg_threshold: float = 1.0
    
### GENERATOR CONFIGS ###
@dataclass
class EpisodeGeneratorConfig:
    num_episodes: int = MISSING
    policy: BasePolicyConfig = MISSING
    filter_method: BaseFilterMethodConfig = MISSING
    deterministic: bool = MISSING
    ckpt_freq: int = 1
    store_artifact: ArtifactConfig = MISSING
    
### DATASET CONFIGS ###
@dataclass
class BaseDatasetConfig:
    _target_: str = MISSING
    artifact: ArtifactConfig = MISSING

@dataclass
class OfflineEpisodeDatasetConfig(BaseDatasetConfig):
    _target_: str = "datasets.offline_episode"
    
    
### TRAINER CONFIGS ###
   
@dataclass
class BaseRunnerConfig:
    policy: BasePolicyConfig = MISSING
    dataset: BaseDatasetConfig = MISSING
    
    pretrained_artifact: ArtifactConfig = MISSING
    
    num_envs: int = MISSING

@dataclass
class BaseLRConfig:
    lr: float = MISSING

@dataclass
class ConstantLRConfig(BaseLRConfig):
    pass

@dataclass
class ExponentialLRConfig(BaseLRConfig):
    _target_: str = 'torch.optim.lr_scheduler.ExponentialLR'
    lr: float = MISSING
    gamma: float = MISSING


@dataclass
class TrainRunnerConfig(BaseRunnerConfig):
    epochs: int = MISSING
    batch_size: int = MISSING
    ckpt_freq: int = 50
    episodes_per_batch: int = MISSING
    max_trajectory_length: int = MISSING
    grad_accums: int = MISSING
    lr_schedule: BaseLRConfig = MISSING

@dataclass
class BCTrainRunnerConfig(TrainRunnerConfig):
    bc_epochs: int = 10 

@dataclass
class EvalRunnerConfig(BaseRunnerConfig):
    ckpt: str = MISSING
    num_episodes: int = MISSING
    save_videos: bool = MISSING
    deterministic: bool = MISSING
    dtg_threshold: float = 1.0
    

cs = ConfigStore.instance()
cs.store(group='exp', name='base', node=ExperimentConfig)

cs.store(group='logger', name='base', node=LoggerConfig)
cs.store(group='logger', name='wb', node=WBLoggerConfig)
cs.store(group='logger', name='console', node=ConsoleLoggerConfig)

cs.store(group='generator', name='base', node=EpisodeGeneratorConfig)
cs.store(group='generator/filter_method', name='base', node=BaseFilterMethodConfig)
cs.store(group='generator/filter_method', name='dtg', node=DTGFitlerMethodConfig)

cs.store(group='policy', name='base', node=BasePolicyConfig)
cs.store(group='policy/old_eai_policy', name='old_eai_policy', node=OldEAIPolicyConfig)
cs.store(group='policy/nav_llama', name='base_nav_llama', node=BaseNavLLaMAPolicyConfig)

cs.store(group='dataset', name='base', node=BaseDatasetConfig)
cs.store(group='dataset', name='offline_episode', node=OfflineEpisodeDatasetConfig)

cs.store(group='lr', name='base', node=BaseLRConfig)
cs.store(group='lr', name='const', node=ConstantLRConfig)
cs.store(group='lr', name='exponential', node=ExponentialLRConfig)

cs.store(group='runner', name='base', node=BaseRunnerConfig)
cs.store(group='runner', name='base_train', node=TrainRunnerConfig)
cs.store(group='runner', name='bc', node=BCTrainRunnerConfig)
cs.store(group='runner', name='eval', node=EvalRunnerConfig)


