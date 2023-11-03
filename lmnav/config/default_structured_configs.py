from dataclasses import dataclass
from hydra.core.config_store import ConfigStore

from typing import List, Optional, Tuple
from omegaconf import MISSING, OmegaConf
import torch


@dataclass
class LoggerConfig:
    _target_: str = MISSING


@dataclass
class WBLoggerConfig(LoggerConfig):
    _target_: str = "lmnav.common.writer.WandBLogger"


@dataclass
class ConsoleLoggerConfig(LoggerConfig):
    _target_: str = "lmnav.common.writer.ConsoleLogger"


@dataclass
class ExperimentConfig:
    name: Optional[str] = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    root_dir: str = "experiments/"
    project: str = "lmnav"
    group: str = MISSING
    job_type: str = MISSING
    tags: Optional[list] = None
    notes: Optional[str] = None
    resume_id: Optional[str] = None

    logger: LoggerConfig = MISSING


### ARTIFACT CONFIG ###
@dataclass
class ArtifactConfig:
    name: str = "${quote:${...exp.group}-${...exp.job_type}-${...exp.name}}"
    dirpath: Optional[str] = None
    version: str = MISSING


### MODEL CONFIGS ###
@dataclass
class BaseModelConfig:
    is_model: bool = True
    use_artifact_config: bool = False
    load_artifact: Optional[ArtifactConfig] = None


@dataclass
class BaseObservationEncoderConfig(BaseModelConfig):
    _target_: str = MISSING
    vis_processor: Optional[dict] = None
    image_size: int = 224


@dataclass
class QformerObservationEncoderConfig(BaseObservationEncoderConfig):
    _target_: str = "lmnav.models.vis_encoders.QformerObservationEncoder"
    vit_precision: str = "fp16"
    vit_model: str = "eva_clip_g"
    drop_path_rate: int = 0
    use_grad_checkpoint: bool = False
    num_query_token: int = 32
    freeze_backbone: bool = True
    freeze_qformer: bool = True
    qformer_compressor_cfg: Optional[dict] = None
    qformer_model: str = "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth"
    max_batch_size: int = 1024


@dataclass
class CLIPObservationEncoderConfig(BaseObservationEncoderConfig):
    _target_: str = "lmnav.models.vis_encoders.CLIPObservationEncoder"
    vit_precision: str = "fp16"
    vit_model: str = "openai/clip-vit-large-patch14"
    freeze_backbone: bool = True
    max_batch_size: int = 1024
    fuse_rgb_goal: bool = False
    precomputed_embeddings: bool = False


@dataclass
class VC1ObservationEncoderConfig(BaseObservationEncoderConfig):
    _target_: str = "lmnav.models.vis_encoders.VC1ObservationEncoder"
    vit_precision: str = "fp16"
    vit_model: str = "vc1_vitl"
    freeze_backbone: bool = True
    max_batch_size: int = 3096
    precomputed_embeddings: bool = False


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
class PPOAgentModelConfig(BaseModelConfig):
    _target_: str = "lmnav.models.ppo_agent.PPOAgent"
    actor: BasePolicyConfig = MISSING
    critic: BaseModelConfig = MISSING
    max_trajectory_length: int = MISSING


@dataclass
class BaseNavLLaMAPolicyConfig(BasePolicyConfig):
    _target_: str = "lmnav.models.nav_llama.NavLLAMA"

    vis_encoder: BaseObservationEncoderConfig = MISSING

    freeze_llama_proj: bool = False

    low_resource: bool = False
    lora_config: Optional[dict] = None

    llama_model: str = "meta-llama/Llama-2-7b-chat-hf"
    max_trajectory_length: int = MISSING
    action_head_mode: str = "lm"


@dataclass
class BaseNavVanillaTransformerPolicyConfig(BasePolicyConfig):
    _target_: str = "lmnav.models.nav_vanilla.NavVanillaTransformer"

    vis_encoder: BaseObservationEncoderConfig = MISSING

    d_hidden: int = 512
    d_head: int = 64
    n_heads: int = 8
    n_blocks: int = 2
    drop_p: float = 0.2
    ln_mode: str = "post"
    max_trajectory_length: int = 200

@dataclass
class BaseNavTransformerXLPolicyConfig(BasePolicyConfig):
    _target_: str = "lmnav.models.nav_txl.TransformerXL"

    vis_encoder: BaseObservationEncoderConfig = MISSING

    d_hidden: int = 512
    n_heads: int = 8
    n_blocks: int = 2
    drop_p: float = 0.2
    ln_mode: str = "post"
    max_trajectory_length: int = 200
    positional_encoding: str = "learned"
    use_gtrxl: bool = True
    gtrxl_bias: float = 0.0


@dataclass
class BaseNavGRUPolicyConfig(BasePolicyConfig):
    _target_: str = "lmnav.models.nav_gru.NavGRU"

    vis_encoder: BaseObservationEncoderConfig = MISSING

    d_hidden: int = 512
    n_layer: int = 2
    drop_p: float = 0.2
    weight_decay: float = 0.0
    max_trajectory_length: int = 200


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
    max_episode_id_repeats: int = -1


### DATASET CONFIGS ###
@dataclass
class BaseDatasetConfig:
    _target_: str = MISSING
    artifact: ArtifactConfig = MISSING


@dataclass
class OfflineEpisodeDatasetConfig(BaseDatasetConfig):
    _target_: str = "datasets.offline_episode"


### DATA TRANSFORM CONFIGS ###
@dataclass
class BaseDataTransformConfig:
    _target_: str = "lmnav.dataset.transforms.BaseDataTransform"


@dataclass
class SequentialDataTransformConfig:
    _target_: str = "lmnav.dataset.transforms.SequentialDataTransform"
    list_of_transforms: List[BaseDataTransformConfig] = MISSING


@dataclass
class ReverseTurnsTransformConfig(BaseDataTransformConfig):
    _target_: str = "lmnav.dataset.transforms.ReverseTurnsTransform"


### TRAINER CONFIGS ###
@dataclass
class BaseLRConfig:
    lr: float = MISSING


@dataclass
class ConstantLRConfig(BaseLRConfig):
    _target_: str = "constant"


@dataclass
class ExponentialLRConfig(BaseLRConfig):
    _target_: str = "exponential"
    gamma: float = MISSING


@dataclass
class WarmupThenLRConfig(BaseLRConfig):
    _target_: str = "warmup_then"
    warmup_start: float = MISSING
    warmup_end: float = MISSING
    warmup_steps: int = MISSING
    after_warmup: BaseLRConfig = MISSING


@dataclass
class ActorCriticLRConfig(BaseLRConfig):
    _target_: str = "group"
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
    max_grad_norm: Optional[float] = 1.0
    weight_decay: float = 0.0
    betas: Tuple[float, float] = (0.9, 0.999)
    ckpt_freq: int = 50
    lr_schedule: BaseLRConfig = MISSING


@dataclass
class BCTrainRunnerConfig(TrainRunnerConfig):
    episodes_per_batch: int = MISSING
    transforms: BaseDataTransformConfig = MISSING


@dataclass
class PPOTrainRunnerConfig(TrainRunnerConfig):
    policy: PPOAgentModelConfig = MISSING

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
cs.store(group="exp", name="base", node=ExperimentConfig)

cs.store(group="logger", name="base", node=LoggerConfig)
cs.store(group="logger", name="wb", node=WBLoggerConfig)
cs.store(group="logger", name="console", node=ConsoleLoggerConfig)

cs.store(group="generator", name="base", node=EpisodeGeneratorConfig)
cs.store(group="generator/filter_method", name="base", node=BaseFilterMethodConfig)
cs.store(group="generator/filter_method", name="dtg", node=DTGFitlerMethodConfig)

cs.store(group="models/policy", name="base", node=BasePolicyConfig)
cs.store(
    group="models/policy/old_eai_policy", name="old_eai_policy", node=OldEAIPolicyConfig
)
cs.store(
    group="models/policy/nav_llama",
    name="base_nav_llama",
    node=BaseNavLLaMAPolicyConfig,
)
cs.store(
    group="models/policy/nav_vanilla",
    name="base_nav_vanilla",
    node=BaseNavVanillaTransformerPolicyConfig,
)
cs.store(
    group="models/policy/nav_txl",
    name="base_nav_txl",
    node=BaseNavTransformerXLPolicyConfig,
)

cs.store(
    group="models/policy/nav_gru", name="base_nav_gru", node=BaseNavGRUPolicyConfig
)
cs.store(group="models", name="linear", node=LinearHeadPolicyConfig)

cs.store(group="models/vis_encoder", name="base", node=BaseObservationEncoderConfig)
cs.store(
    group="models/vis_encoder", name="qformer", node=QformerObservationEncoderConfig
)
cs.store(group="models/vis_encoder", name="clip", node=CLIPObservationEncoderConfig)
cs.store(group="models/vis_encoder", name="vc1", node=VC1ObservationEncoderConfig)

cs.store(group="dataset", name="base", node=BaseDatasetConfig)
cs.store(group="dataset", name="offline_episode", node=OfflineEpisodeDatasetConfig)

cs.store(group="transforms", name="base", node=BaseDataTransformConfig)
cs.store(group="transforms", name="reverse_turns", node=ReverseTurnsTransformConfig)

cs.store(group="lr", name="base", node=BaseLRConfig)
cs.store(group="lr", name="constant", node=ConstantLRConfig)
cs.store(group="lr", name="exponential", node=ExponentialLRConfig)
cs.store(group="lr", name="warmup_then", node=WarmupThenLRConfig)
cs.store(group="lr", name="actor_critic", node=ActorCriticLRConfig)

cs.store(group="runner", name="base", node=BaseRunnerConfig)
cs.store(group="runner", name="base_train", node=TrainRunnerConfig)
cs.store(group="runner", name="base_bc", node=BCTrainRunnerConfig)
cs.store(group="runner", name="eval", node=EvalRunnerConfig)
cs.store(group="runner", name="ppo", node=PPOTrainRunnerConfig)

OmegaConf.register_new_resolver(
    "quote", lambda x: x.replace("+", "_").replace("=", "_")
)
