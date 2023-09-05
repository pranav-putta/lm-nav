from abc import ABC, abstractmethod
from omegaconf import OmegaConf
import wandb
import os

def get_writer(cfg):
    writer_type = cfg.bc.writer

    if writer_type == 'console':
        return ConsoleWriter(cfg)
    elif writer_type == 'wb':
        return WandBWriter(cfg)
    else:
        raise NotImplementedError()
    

class BaseWriter(ABC):
    
    def __init__(self, config):
        pass

    @abstractmethod
    def write(self, log_dict):
        pass


class ConsoleWriter(BaseWriter):

    def write(self, log_dict):
        print(log_dict)

class WandBWriter(BaseWriter):

    def __init__(self, config):
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
        # if resume_run_id is not None:
        #     wb_kwargs["id"] = resume_run_id
        #     wb_kwargs["resume"] = "must"
        
        self.run = wandb.init(  # type: ignore[attr-defined]
            config={
                "slurm": slurm_info_dict,
                **OmegaConf.to_container(config),  # type: ignore[arg-type]
            },
            **wb_kwargs,
        ) 
    
    def write(self, log_dict):
        wandb.log(log_dict)
        
