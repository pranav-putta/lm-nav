from abc import ABC, abstractmethod
from omegaconf import OmegaConf
import wandb
import os
from urllib.parse import unquote, urlparse
from lmnav.common.registry import registry

from lmnav.config.default_structured_configs import WBLoggerConfig

    

class BaseLogger(ABC):
    
    def __init__(self, config):
        pass

    @abstractmethod
    def write(self, log_dict):
        pass

    @abstractmethod
    def save_artifact(self, name, atype, filepath):
        pass

    @abstractmethod
    def load_dataset(self, name):
        pass



@registry.register_logger('console')
class ConsoleLogger(BaseLogger):

    def write(self, log_dict):
        print(log_dict)
        
    def save_artifact(self, name, atype, filepath):
        pass

    def load_dataset(self, name):
        name = name.split(':')[0]
        dirpath = f'data/datasets/lmnav/{name}'
        files = [os.path.join(dirpath, path) for path in os.listdir(dirpath)]
        return files

@registry.register_logger('wb')
class WandBLogger(BaseLogger):

    def __init__(self, config: WBLoggerConfig):
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
        
        self.run = wandb.init(  # type: ignore[attr-defined]
            config={
                "slurm": slurm_info_dict,
                **OmegaConf.to_container(config),  # type: ignore[arg-type]
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

    def load_dataset(self, name):
        artifact = wandb.use_artifact(name)
        files = [unquote(urlparse(v.ref).path) for k, v in artifact.manifest.entries.items()]
        return files
        
