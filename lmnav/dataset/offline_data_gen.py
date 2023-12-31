import argparse
from collections import defaultdict
from functools import partial
from pympler import tracker
import gc
import tarfile
from habitat_sim.utils.datasets_download import gzip
import numpy as np
from habitat.config.read_write import read_write
import torch
import torch.distributed
import pickle

import copy
import time
import os
from lmnav.common.registry import registry 
from lmnav.common.writer import BaseLogger

from lmnav.dataset.data_gen import start_data_gen_process
from lmnav.config.default import get_config
from lmnav.dataset.filter_methods import *
from lmnav.common.actor_setups import *
from hydra.utils import instantiate

class OfflineDataGenerator:

    def __init__(self, run):
        self.config = get_config(run)
        self.num_gpus = torch.cuda.device_count()
        self.buffer = []
        self.episode_repeats = defaultdict(lambda: 0)
        self.N = 0
        self.latest_generator_stats = {}
        
        self.writer = instantiate(self.config.exp.logger)
        self.writer.open(self.config)

        
    def initialize_data_generator(self, exp_folder, max_buffer_len, gpu_id):
        os.makedirs(exp_folder, exist_ok=True)
        
        # update N if there are already generated files in bfufer
        files = os.listdir(exp_folder)
        self.N = len(files) * max_buffer_len

        cfg = copy.deepcopy(self.config)        

        with read_write(cfg):
            cfg.exp.device = f'cuda:{gpu_id}'
            cfg.habitat_baselines.torch_gpu_id = gpu_id
            cfg.habitat.simulator.habitat_sim_v0.gpu_device_id = gpu_id

        filter_fn = partial(registry.get_fn(cfg.generator.filter_method._target_), cfg.generator.filter_method)
        setup_policy = registry.get_fn(self.config.generator.policy._target_)

        process, conn, queue = start_data_gen_process(cfg, setup_policy, filter_fn, cfg.generator.deterministic) 
        return cfg, process, conn, queue
         
    
    def reformat_episode(self, episode_stats, episode):
        info_keys_to_keep = set(['distance_to_goal', 'success', 'spl'])
        
        new_episode = {**episode_stats}
        new_episode['rgb'] = torch.stack([step['observation']['rgb'].clone() for step in episode])
        new_episode['depth'] = torch.stack([step['observation']['depth'].clone() for step in episode]).to(torch.float16)
        new_episode['imagegoal'] = episode[0]['observation']['imagegoal'][None, :].clone()
        new_episode['reward'] = torch.tensor([step['reward'] for step in episode]).to(torch.float16)
        new_episode['action'] = torch.tensor([step['action'] for step in episode]).to(torch.uint8)
        new_episode['action_probs'] = torch.stack([step['probs'] for step in episode]).clone()
        new_episode['info'] = [{ k:v for k, v in step['info'].items() if k in info_keys_to_keep } for step in episode]

        return new_episode


    def generate(self):
        generator_cfg = self.config.generator
        max_buffer_len = generator_cfg.ckpt_freq
        exp_folder = os.path.join(generator_cfg.store_artifact.dirpath, generator_cfg.store_artifact.name)
        
        print(f"Starting generation with {self.num_gpus} gpus")
        print(f"Max episode id repeats: {generator_cfg.max_episode_id_repeats}")
        print(f"Saving data to {exp_folder}")
        
        self.cfgs, self.processes, self.conns, self.queues = zip(*[self.initialize_data_generator(exp_folder, max_buffer_len, i) for i in range(self.num_gpus)]) 
        
        step = 0
        episode = None
        
        while self.N < self.config.generator.num_episodes:
            # TODO; update this to organize data by scene 
            for queue in self.queues:
                while not queue.empty():
                    byte_data = queue.get()
                    generator_stats, episode_stats, episode = pickle.loads(byte_data) 
                    self.latest_generator_stats = generator_stats

                    # reformat episode
                    formatted_episode = self.reformat_episode(episode_stats, episode)              

                    # check if max episode reached and update
                    self.episode_repeats[(formatted_episode['scene_id'], formatted_episode['episode_id'])] += 1
                    if self.episode_repeats[(formatted_episode['scene_id'], formatted_episode['episode_id'])] < generator_cfg.max_episode_id_repeats:
                        self.buffer.append(formatted_episode) 

                    del episode_stats
                    del generator_stats
                    del episode
                    
            while len(self.buffer) >= max_buffer_len:
                data = self.buffer[:max_buffer_len]
                self.buffer = self.buffer[max_buffer_len:]
                
                # dump data into torch store
                datanum = self.N // max_buffer_len
                filepath = os.path.join(exp_folder, f'data.{datanum}.pkl')
                   
                with open(filepath, 'wb+') as f:
                # in the special case where each file is only 1 episode, get rid of the list
                    pickle.dump(data[0] if len(data) == 1 else data, f)

                # del data
                self.N += max_buffer_len
           

            step += 1
            time.sleep(1)
            if step % 10 == 0:
                self.writer.write({'step': step, 
                               'num_episodes': self.N,
                               'buffer_len': len(self.buffer),
                               **self.latest_generator_stats })

                
       
        
def main():
    parser = argparse.ArgumentParser(description="Example argparse for cfg_path")
    parser.add_argument('cfg_path', type=str, help="Path to the configuration file")
    args = parser.parse_args()

    generator = OfflineDataGenerator(args.cfg_path)
    generator.generate()

if __name__ == '__main__':
    main()
