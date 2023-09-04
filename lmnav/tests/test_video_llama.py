from lmnav.common.config import Config
from lmnav.common.registry import registry

#%%
from lmnav.models import *
from lmnav.processors import *

import os
import torch

from collections import namedtuple

os.chdir('/srv/flash1/pputta7/projects/lm-nav')

Args = namedtuple("Args", "cfg_path, model_type, gpu_id, options")
args = Args("/srv/flash1/pputta7/projects/lm-nav/exp_configs/video_llama_eval_only_vl.yaml", "llama_v2", 0, [])

cfg = Config(args)

model_config = cfg.model_cfg
# model_cls = registry.get_model_class(model_config.arch)
# model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))
# model.eval()

vis_processor_cfg = cfg.config.preprocess.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

x = torch.rand(3, 70, 480, 640)
vis_processor.transform(x)

