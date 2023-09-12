from functools import partial

from habitat_baselines.common.obs_transformers import apply_obs_transforms_obs_space, get_active_obs_transforms
from lmnav.dataset.data_gen import start_data_gen_process
from lmnav.common.episode_processor import extract_inputs_from_dataset 

import habitat


import unittest
    
from lmnav.emb_transfer.old_eai_policy import OldEAIPolicy
import torch

   
def dtg_filter_fn(threshold, episode):
    return len(episode) < 500 and episode[-1]['info']['distance_to_goal'] <= threshold
     
    
def setup_eai_teacher(teacher_ckpt, device, config, env_spec):
    print("setting up eai teacher...")
    obs_transforms = get_active_obs_transforms(config)
    env_spec.observation_space = apply_obs_transforms_obs_space(
            env_spec.observation_space, obs_transforms
        )
    obs_space, action_space = env_spec.observation_space, env_spec.action_space

    teacher = OldEAIPolicy.hardcoded(OldEAIPolicy, obs_space, action_space)
    teacher.obs_transforms = obs_transforms
    teacher.device = torch.device(device)

    ckpt_dict = torch.load(teacher_ckpt, map_location='cpu')
    state_dict = ckpt_dict['state_dict']
    state_dict = {k[len('actor_critic.'):]: v for k, v in state_dict.items()}

    teacher.load_state_dict(state_dict, strict=False)
    teacher = teacher.to(device)
    teacher = teacher.eval()

    for param in teacher.parameters():
        param.requires_grad = False
    
    print("done setting up eai teacher...")
    return teacher



class TestEpisodeProcessor(unittest.TestCase):
    
    def setUp(self):
        self.device = 'cuda:0'
        self.config = habitat.get_config("./lmnav/configs/datagen/test_imagenav_data_gen_env744.yaml")

    def test_data_gen_process(self):
        setup_teacher = partial(setup_eai_teacher, self.config.bc.teacher_ckpt, self.device)         
        filter_fn = partial(dtg_filter_fn, self.config.bc.dtg_threshold)
        process, conn, queue = start_data_gen_process(self.config, setup_teacher, filter_fn, deterministic=False)
        
        dataset = [queue.get() for _ in range(1)]
       
        conn.send("EXIT")
        conn.close()

        process.join()
        process.close()
 
        print("Collected episodes!")
        
        print(dataset) 
 
if __name__ == '__main__':
    unittest.main()
