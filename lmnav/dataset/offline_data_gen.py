import argparse
from functools import partial
import gc
import tarfile
from habitat_sim.utils.datasets_download import gzip
import numpy as np
import multiprocessing as mp
from multiprocessing.connection import Connection
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

import torchvision.transforms as transforms

class OfflineDataGenerator:

    def __init__(self, run):
        self.config = get_config(run)
        self.num_gpus = torch.cuda.device_count()
        self.buffer = []
        self.N = 0
        self.latest_generator_stats = {}
        
        self.writer: BaseLogger = registry.get_logger_class(self.config.exp.logger._target_)(self.config)
        self.img_transform = transforms.Compose([transforms.Resize((224, 224), antialias=True)])

        
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
         
    
    def reformat_episode(self, episode_stats, raw_episode):
        episode = copy.deepcopy(raw_episode)
        info_keys_to_keep = set(['distance_to_goal', 'success', 'spl'])
        
        new_episode = {**episode_stats}
        new_episode['rgb'] = torch.from_numpy(np.stack([step['observation']['rgb'] for step in episode]))
        new_episode['depth'] = torch.from_numpy(np.stack([step['observation']['depth'] for step in episode]).astype(np.float16))
        new_episode['imagegoal'] = torch.from_numpy(episode[0]['observation']['imagegoal'][None, :])
        new_episode['reward'] = torch.from_numpy(np.array([step['reward'] for step in episode]).astype(np.float16))
        new_episode['action'] = torch.from_numpy(np.array([step['action'] for step in episode]).astype(np.uint8))
        new_episode['action_probs'] = torch.stack([step['probs'] for step in episode])
        new_episode['info'] = [{ k:v for k, v in step['info'].items() if k in info_keys_to_keep } for step in episode]

        del raw_episode
        return new_episode


        
    def generate(self):
        generator_cfg = self.config.generator
        max_buffer_len = generator_cfg.ckpt_freq
        exp_folder = os.path.join(generator_cfg.store_artifact.dirpath, generator_cfg.store_artifact.name)
        
        print(f"Starting generation with {self.num_gpus} gpus")
        print(f"Saving data to {exp_folder}")
        
        self.cfgs, self.processes, self.conns, self.queues = zip(*[self.initialize_data_generator(exp_folder, max_buffer_len, i) for i in range(self.num_gpus)]) 
        
        step = 0
        while self.N < self.config.generator.num_episodes:
            # TODO; update this to organize data by scene 
            for queue in self.queues:
                while not queue.empty():
                    generator_stats, episode_stats, episode = queue.get()
                    self.latest_generator_stats = generator_stats

                    # reformat episode
                    formatted_episode = self.reformat_episode(episode_stats, episode)              
                    self.buffer.append(formatted_episode) 

                    del episode

                    
            while len(self.buffer) >= max_buffer_len:
                data = self.buffer[:max_buffer_len]
                self.buffer = self.buffer[max_buffer_len:]
                
                # dump data into torch store
                datanum = self.N // max_buffer_len
                filepath = os.path.join(exp_folder, f'data.{datanum}.pth')

                # in the special case where each file is only 1 episode, get rid of the list
                if max_buffer_len == 1:
                    data = data[0]
                    
                # gzip file as well
                with gzip.open(filepath, 'wb+') as f:
                    pickle.dump(data, f)
               
                self.N += max_buffer_len
                gc.collect()
                

            time.sleep(10)
            step += 1
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
