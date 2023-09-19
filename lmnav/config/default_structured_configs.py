from dataclasses import dataclass 
from hydra.core.config_store import ConfigStore
    
from typing import Optional
from omegaconf import MISSING, OmegaConf
import torch

@dataclass
class LoggerConfig:
    _target_: str = MISSING
    
@dataclass
class WBLoggerConfig(LoggerConfig):
    _target_: str = 'lmnav.common.writer.WandBLogger'

@dataclass
class ConsoleLoggerConfig(LoggerConfig):
    _target_: str = 'lmnav.common.writer.ConsoleLogger'

@dataclass
class ExperimentConfig:
    name: Optional[str] = None
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    root_dir: str = 'experiments/'
    project: str = 'lmnav'
    group: str = MISSING
    job_type: str = MISSING
    tags: Optional[list] = None
    notes: Optional[str] = None
    resume_id: Optional[str] = None

    logger: LoggerConfig = MISSING

### ARTIFACT CONFIG ###
@dataclass
class ArtifactConfig:
    name: str = MISSING
    version: str = MISSING
    dirpath: Optional[str] = MISSING
    

### MODEL CONFIGS ###
@dataclass
class BaseModelConfig:
    _target_: str = MISSING
    load_artifact: Optional[ArtifactConfig] = None
    use_artifact_policy_config: bool = False


@dataclass
class BaseVisualEncoderConfig(BaseModelConfig):
    _target_: str = MISSING
    vis_processor: Optional[dict] = None
    image_size: int = 224

@dataclass
class QformerVisualEncoderConfig(BaseVisualEncoderConfig):
    _target_: str = "lmnav.models.vis_encoders.QformerVisualEncoder"
    vit_precision: str = "fp16"
    vit_model: str = "eva_clip_g"
    drop_path_rate: int = 0
    use_grad_checkpoint: bool = False
    num_query_token: int = 32
    freeze_vit: bool = True
    freeze_qformer: bool = True
    qformer_compressor_cfg: Optional[dict] = None
    qformer_model: str = "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth"

@dataclass
class CLIPVisualEncoderConfig(BaseVisualEncoderConfig):
    _target_:str = "lmnav.models.vis_encoders.CLIPVisualEncoder"
    vit_precision: str = "fp16"
    vit_model: str = "openai/clip-vit-large-patch14"
    freeze_vit: bool = True
    
    
@dataclass
class BasePolicyConfig(BaseModelConfig):
    pass


@dataclass
class LinearHeadPolicyConfig(BasePolicyConfig):
    _target_: str = "lmnav.models.linear_head.LinearHead"
    in_dim: Optional[int] = None
    p_dropout: float = 0.2    


@dataclass
class OldEAIPolicyConfig(BasePolicyConfig):
    _target_: str = "old_eai_policy"
    ckpt: str = MISSING

    
@dataclass
class BaseNavLLaMAPolicyConfig(BasePolicyConfig):
    _target_: str = "lmnav.models.nav_llama.NavLLAMA"

    vis_encoder: BaseVisualEncoderConfig = MISSING

    freeze_llama_proj: bool = False
    
    low_resource: bool = False
    lora_config: Optional[dict] = None

    llama_model: str = "meta-llama/Llama-2-7b-chat-hf"
    max_trajectory_length: int = MISSING

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
class BaseLRConfig:
    lr: float = MISSING

@dataclass
class ConstantLRConfig(BaseLRConfig):
    _target_: str = 'constant'
    
@dataclass
class ExponentialLRConfig(BaseLRConfig):
    _target_: str = 'exponential'
    gamma: float = MISSING

@dataclass
class ActorCriticLRConfig(BaseLRConfig):
    _target_: str = 'group'
    actor: BaseLRConfig = MISSING
    critic: BaseLRConfig = MISSING
    
@dataclass
class BaseRunnerConfig:
    policy: BasePolicyConfig = MISSING
    dataset: BaseDatasetConfig = MISSING
    store_artifact: Optional[ArtifactConfig] = None

@dataclass
class TrainRunnerConfig(BaseRunnerConfig):
    steps: int = MISSING
    batch_size: int = MISSING
    minibatch_size: int = MISSING
    num_grad_accums: int = MISSING
    max_grad_norm: Optional[float] = 1.2
    ckpt_freq: int = 50

@dataclass
class BCTrainRunnerConfig(TrainRunnerConfig):
    episodes_per_batch: int = MISSING
    lr_schedule: BaseLRConfig = MISSING

@dataclass
class PPOTrainRunnerConfig(TrainRunnerConfig):
    actor: BasePolicyConfig = MISSING
    critic: BaseModelConfig = MISSING
    
    lr_schedule: ActorCriticLRConfig = MISSING
    num_rollout_steps: int = MISSING
    ppo_epochs: int = MISSING
    num_envs: int = MISSING
    cliprange_value: float = 0.2
    cliprange: float = 0.2
    vf_coef: float = 0.1
    gamma: float = 1
    lam: float = 0.95
    ratio_threshold: float = 10.0
    deterministic: bool = False

@dataclass
class EvalRunnerConfig(BaseRunnerConfig):
    ckpt: str = MISSING
    num_episodes: int = MISSING
    save_videos: bool = MISSING
    deterministic: bool = MISSING
    dtg_threshold: float = 1.0
    num_envs: int = MISSING
    

cs = ConfigStore.instance()
cs.store(group='exp', name='base', node=ExperimentConfig)

cs.store(group='logger', name='base', node=LoggerConfig)
cs.store(group='logger', name='wb', node=WBLoggerConfig)
cs.store(group='logger', name='console', node=ConsoleLoggerConfig)

cs.store(group='generator', name='base', node=EpisodeGeneratorConfig)
cs.store(group='generator/filter_method', name='base', node=BaseFilterMethodConfig)
cs.store(group='generator/filter_method', name='dtg', node=DTGFitlerMethodConfig)

cs.store(group='models/policy', name='base', node=BasePolicyConfig)
cs.store(group='models/policy/old_eai_policy', name='old_eai_policy', node=OldEAIPolicyConfig)
cs.store(group='models/policy/nav_llama', name='base_nav_llama', node=BaseNavLLaMAPolicyConfig)
cs.store(group='models', name='linear', node=LinearHeadPolicyConfig)

cs.store(group='models/vis_encoder', name='base', node=BaseVisualEncoderConfig)
cs.store(group='models/vis_encoder', name='qformer', node=QformerVisualEncoderConfig)
cs.store(group='models/vis_encoder', name='clip', node=CLIPVisualEncoderConfig)

cs.store(group='dataset', name='base', node=BaseDatasetConfig)
cs.store(group='dataset', name='offline_episode', node=OfflineEpisodeDatasetConfig)

cs.store(group='lr', name='base', node=BaseLRConfig)
cs.store(group='lr', name='constant', node=ConstantLRConfig)
cs.store(group='lr', name='exponential', node=ExponentialLRConfig)
cs.store(group='lr', name='actor_critic', node=ActorCriticLRConfig)

cs.store(group='runner', name='base', node=BaseRunnerConfig)
cs.store(group='runner', name='base_train', node=TrainRunnerConfig)
cs.store(group='runner', name='bc', node=BCTrainRunnerConfig)
cs.store(group='runner', name='eval', node=EvalRunnerConfig)
cs.store(group='runner', name='ppo', node=PPOTrainRunnerConfig)

OmegaConf.register_new_resolver('quote', lambda x: x.replace('+', '_').replace('=', '_'))
