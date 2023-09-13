from habitat_baselines.common.obs_transformers import apply_obs_transforms_obs_space, get_active_obs_transforms
import torch
from lmnav.emb_transfer.old_eai_policy import OldEAIPolicy
from lmnav.common.registry import registry

@registry.register_fn('policies.old_eai_policy')
def setup_eai_teacher(config, env_spec):
    print("setting up eai teacher...")
    device = torch.device(config.exp.device)
    obs_transforms = get_active_obs_transforms(config)
    env_spec.observation_space = apply_obs_transforms_obs_space(
            env_spec.observation_space, obs_transforms
        )
    obs_space, action_space = env_spec.observation_space, env_spec.action_space

    teacher = OldEAIPolicy.hardcoded(OldEAIPolicy, obs_space, action_space)
    teacher.obs_transforms = obs_transforms
    teacher.device = device
    
    ckpt_dict = torch.load(config.generator.policy.ckpt, map_location='cpu')
    state_dict = ckpt_dict['state_dict']
    state_dict = {k[len('actor_critic.'):]: v for k, v in state_dict.items()}

    teacher.load_state_dict(state_dict, strict=False)
    teacher = teacher.to(device)
    teacher = teacher.eval()

    for param in teacher.parameters():
        param.requires_grad = False
    
    print("done setting up eai teacher...")
    return teacher


