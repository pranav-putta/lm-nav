from abc import ABC, abstractmethod
from omegaconf import OmegaConf
import wandb
import os
from urllib.parse import unquote, urlparse

def get_writer(cfg, resume_run_id):
    writer_type = cfg.bc.writer

    if writer_type == 'console':
        return ConsoleWriter(cfg, resume_run_id)
    elif writer_type == 'wb':
        return WandBWriter(cfg, resume_run_id)
    else:
        raise NotImplementedError()
    

class BaseWriter(ABC):
    
    def __init__(self, config, resume_run_id):
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


class ConsoleWriter(BaseWriter):

    def write(self, log_dict):
        print(log_dict)
        
    def save_artifact(self, name, atype, filepath):
        pass

    def load_dataset(self, name):
        dirpath = f'data/datasets/{name}'
        files = [path for path in os.listdir(dirpath)]
        return files

class WandBWriter(BaseWriter):

    def __init__(self, config, resume_run_id):
        wb_kwargs = {}
        if config.habitat_baselines.wb.project_name != "":
            wb_kwargs["project"] = config.habitat_baselines.wb.project_name
        if config.habitat_baselines.wb.run_name != "":
            wb_kwargs["name"] = config.habitat_baselines.wb.run_name
        if config.habitat_baselines.wb.entity != "":
            wb_kwargs["entity"] = config.habitat_baselines.wb.entity
        if config.habitat_baselines.wb.group != "":
            wb_kwargs["group"] = config.habitat_baselines.wb.group

        slurm_info_dict = {
            k[len("SLURM_") :]: v
            for k, v in os.environ.items()
            if k.startswith("SLURM_")
        }
        if wandb is None:
            raise ValueError(
                "Requested to log with wandb, but wandb is not installed."
            )

        # TODO: add resume behavior
        if resume_run_id is not None:
            wb_kwargs["id"] = resume_run_id
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
        
