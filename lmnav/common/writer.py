from abc import ABC, abstractmethod
from omegaconf import OmegaConf
import wandb
import os
from urllib.parse import unquote, urlparse
from lmnav.common.registry import registry

from lmnav.config.default_structured_configs import WBLoggerConfig
from tqdm import tqdm

    

class BaseLogger(ABC):
    
    def __init__(self, eval_mode=False):
        self.eval_mode = eval_mode

    @abstractmethod
    def open(self, cfg, override_run_name=None):
        pass
        
    @abstractmethod
    def write(self, log_dict):
        pass

    @abstractmethod
    def save_artifact(self, name, atype, filepath):
        pass

    @abstractmethod
    def load_dataset(self, artifact):
        pass

    @abstractmethod
    def load_model(self, artifact):
        pass

    @abstractmethod
    def load_model_versions(self, artifact):
        pass

    @abstractmethod
    def load_runs(self, filters=None):
        pass

class ConsoleLogger(BaseLogger):

    def open(self, cfg, override_run_name=None):
        pass
        
    def write(self, log_dict):
        print(log_dict)
        
    def save_artifact(self, name, atype, filepath):
        pass 

    def load_dataset(self, dataset_cfg):
        name = f"{dataset_cfg.artifact.name}:{dataset_cfg.artifact.version}" 
        name = name.split(':')[0]
        dirpath = f'data/datasets/lmnav/{name}'
        files = [os.path.join(dirpath, path) for path in os.listdir(dirpath)]
        return files

    def load_model(self, artifact):
        return "/srv/flash1/pputta7/projects/lm-nav/experiments/lmnav-1env/bc/lora/ckpts/ckpt.0.pth"

    def load_model_versions(self, artifact):
        raise NotImplementedError

    def load_runs(self, filters=None):
        raise NotImplementedError

class WandBLogger(BaseLogger):

    def open(self, cfg, override_run_name=None):
        config = cfg.exp
        wb_kwargs = {}
        if config.project != "":
            wb_kwargs["project"] = config.project
        if config.name != "":
            wb_kwargs["name"] = config.name
        if config.group != "":
            wb_kwargs["group"] = config.group
        if config.tags is not None:
            wb_kwargs["tags"] = config.tags
        if config.notes is not None:
            wb_kwargs["notes"] = config.notes
        if config.job_type is not None:
            wb_kwargs['job_type'] = config.job_type

        # special tings for eval mode
        if self.eval_mode:
            wb_kwargs['job_type'] = 'eval'
        if override_run_name is not None:
            wb_kwargs["name"] = override_run_name
            
        slurm_info_dict = {
            k[len("SLURM_") :]: v
            for k, v in os.environ.items()
            if k.startswith("SLURM_")
        }
        if wandb is None:
            raise ValueError(
                "Requested to log with wandb, but wandb is not installed."
            )

        if config.resume_id is not None:
            wb_kwargs["id"] = config.resume_id
            wb_kwargs["resume"] = "must"
            print(f"Attempting to resume {config.resume_id}")
        
        self.run = wandb.init(  # type: ignore[attr-defined]
            config={
                "slurm": slurm_info_dict,
                **OmegaConf.to_container(cfg),  # type: ignore[arg-type]
            },
            **wb_kwargs,
        ) 
    
    def write(self, log_dict):
        wandb.log(log_dict)

    def save_artifact(self, name, atype, filepath):
        artifact = wandb.Artifact(name, type=atype)
        if 'file://' not in filepath:
            filepath = f'file://{filepath}'
        artifact.add_reference(filepath)
        wandb.log_artifact(artifact)

    def load_dataset(self, dataset_cfg):
        name = f"{dataset_cfg.artifact.name}:{dataset_cfg.artifact.version}" 
        artifact = wandb.use_artifact(name)
        files = [unquote(urlparse(v.ref).path) for k, v in artifact.manifest.entries.items()]
        return files

    def load_model(self, artifact):
        version = 'latest' if artifact.version == '*' else artifact.version
        name = f"{artifact.name}:{version}"
        artifact = wandb.use_artifact(name)
        files = [unquote(urlparse(v.ref).path) for k, v in artifact.manifest.entries.items()]
        assert len(files) == 1, 'there was more than 1 file for checkpoint'
        return files[0]
    
    def load_model_versions(self, artifact):
        latest = f"{artifact.name}:latest"
        latest_artifact = wandb.use_artifact(latest)
        latest_version = int(latest_artifact.source_version[1:])
        
        return [f'v{i}' for i in range(latest_version)]            
            
    def load_runs(self, filters=None):
        api = wandb.Api()
        return api.runs("cvmlp-embodiment-transfer/lmnav", filters)
       
