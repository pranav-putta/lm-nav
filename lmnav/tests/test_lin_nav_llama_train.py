from collections import namedtuple

from lmnav.common.config import Config
from lmnav.common.registry import registry

from lmnav.models import *
from lmnav.processors import *
from lmnav.common.episode_processor import apply_transforms_inputs

import lmnav
import torch
import einops


def _init_components(cfg_path, device):
    Args = namedtuple("Args", "cfg_path, model_type, gpu_id, options")
    args = Args(cfg_path, "llama_v2", 0, [])

    cfg = Config(args)

    model_config = cfg.model_cfg
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to(device)
    model.train()

    vis_processor_cfg = cfg.config.preprocess.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

    return model, vis_processor


def embed_visual(imgs, model):
    img_embds, img_attns = model.encode_Qformer_visual(imgs.half().to(model.device))
    return img_embds, img_attns

def tokenize_actions(actions, model):
    action_tkns = [model.llama_tokenizer(' '.join(act), return_tensors='pt', add_special_tokens=False) for act in actions]
    action_tkns_t = torch.stack([tkn.input_ids.to(model.device) for tkn in action_tkns])
    return action_tkns_t


def embed_actions(action_tkns_t, model):
    action_embds = [model.llama_model.model.embed_tokens(tkns_t) for tkns_t in action_tkns_t]
    action_embds = torch.cat(action_embds, dim=0)
    return action_embds


def prompt1_wrap(prompt, rgbs, goals, actions, model):
    rgbs_embd, rgbs_attn = embed_visual(rgbs, model)
    goals_embd, goals_attn = embed_visual(goals, model)
    action_tkns_t = tokenize_actions(actions, model)
    actions_embd = embed_actions(action_tkns_t, model)

    
    B, T, Q, H = rgbs_embd.shape
 
    prompt_segs = prompt.split("{}")
    prompt_tkns = [model.llama_tokenizer(seg, return_tensors='pt', add_special_tokens=(i == 0)) for i, seg in enumerate(prompt_segs) if len(seg)]
    prompt_embd = [model.llama_model.model.embed_tokens(tkns.input_ids.to(model.device)) for tkns in prompt_tkns]
    prompt_embd = [embd.repeat(B, 1, 1) for embd in prompt_embd]
    
    actions_embd = einops.rearrange(actions_embd, 'b t h -> b t 1 h')
    print(rgbs_embd.shape, actions_embd.shape)
    s_a_embds = torch.cat([rgbs_embd, actions_embd], dim=2)
    s_a_embds = s_a_embds.view(B, T * (Q + 1), H)

    goals_embd = goals_embd.squeeze(1)

    embds = [prompt_embd[0], goals_embd, prompt_embd[1], s_a_embds]
    embds = torch.cat(embds, dim=1)

    # construct targets
    prompt_tgts = torch.ones(B, prompt_embd[0].shape[1] + goals_embd.shape[1] + prompt_embd[1].shape[1]).fill_(-100).to('cuda')
    rgb_tgts = torch.ones(B, T, Q).fill_(-100).to(model.device)
    act_tgts = action_tkns_t.permute(0, 2, 1)
    s_a_tgts = torch.cat([rgb_tgts, act_tgts], dim=2).view(B, T * (Q + 1))

    tgts = torch.cat([prompt_tgts, s_a_tgts], dim=1).long().to(model.device) 
    
    return embds, tgts


def test_construct_inputs(B, T):
    goals = torch.rand(B, 1, 3, 480, 640)
    rgbs = torch.rand(B, T, 3, 480, 640)
    actions = torch.randint(0, 4, (B, T)).tolist()

    return goals, rgbs, actions
    
    

def main():
    cfg_path = "/srv/flash1/pputta7/projects/lm-nav/exp_configs/lin_nav_llama_train.yaml"
    device = 'cuda'
    B, T = 2, 20
    prompt = "You are a navigational agent tasked with exploring an indoor environment to find a goal image. \
           You can choose to move { left, right, forward, stop } at every step. The goal image is {}. \
           After every image, choose the best action. {}"
    
    model, vis_processor = _init_components(cfg_path, device)
    goals, rgbs, actions = test_construct_inputs(B, T)
    
    rgbs, goals, actions = apply_transforms_inputs(vis_processor, rgbs, goals, actions)

    print("Shapes after transform")
    print(rgbs.shape, goals.shape)
    
    embd, tgts = prompt1_wrap(prompt, rgbs, goals, actions, model)
    embd = embd.to(model.device)
    tgts = tgts.to(model.device)

    print("Prompt embedding shape")
    print(embd.shape, tgts.shape)

    outputs = model.llama_model(inputs_embeds=embd,
                                labels=tgts,
                                return_dict=True)
    print("Model loss") 
    print(outputs.loss)

if __name__ == "__main__":
    main()
