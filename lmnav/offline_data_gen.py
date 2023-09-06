import argparse
import gc
import tarfile
import numpy as np
import multiprocessing as mp
from multiprocessing.connection import Connection
from habitat.config.read_write import read_write
import torch
import torch.distributed
import habitat

import copy
import time
import os
from lmnav.common.writer import get_writer
from lmnav.common.registry import registry as llama_registry

from lmnav.data_gen import start_data_gen_process

import torchvision.transforms as transforms

class OfflineDataGenerator:

    def __init__(self, cfg_path):
        self.cfg_path = cfg_path
        self.config = habitat.get_config(self.cfg_path)
        self.num_gpus = torch.cuda.device_count()
        self.buffer = []
        self.N = 0
        self.latest_episode_stats = {}
        
        self.writer = get_writer(self.config)
        self.img_transform = transforms.Compose([transforms.Resize((224, 224), antialias=True)])

        
    def initialize_data_generator(self, gpu_id):
        cfg = copy.deepcopy(self.config)        

        with read_write(cfg):
            cfg.habitat_baselines.torch_gpu_id = gpu_id
            cfg.habitat.simulator.habitat_sim_v0.gpu_device_id = gpu_id

        device = torch.device(f'cuda:{gpu_id}')

        process, conn = start_data_gen_process(device, cfg, deterministic=False) 
        return cfg, process, conn
         
    
    def reformat_episode(self, raw_episode):
        episode = copy.deepcopy(raw_episode)
        info_keys_to_keep = set(['distance_to_goal', 'success', 'spl'])
        
        new_episode = {}
        new_episode['rgb'] = torch.from_numpy(np.stack([step['observation']['rgb'] for step in episode]))
        new_episode['depth'] = torch.from_numpy(np.stack([step['observation']['depth'] for step in episode]).astype(np.float16))
        new_episode['imagegoal'] = torch.from_numpy(episode[0]['observation']['imagegoal'][None, :])
        new_episode['reward'] = torch.from_numpy(np.array([step['reward'] for step in episode]).astype(np.float16))
        new_episode['action'] = torch.from_numpy(np.array([step['action'] for step in episode]).astype(np.uint8))
        new_episode['info'] = [{ k:v for k, v in step['info'].items() if k in info_keys_to_keep } for step in episode]

        del raw_episode
        return new_episode


    @staticmethod
    def save_and_compress_data(data, datanum, exp_folder):
        filepath = os.path.join(exp_folder, f'data.{datanum}.pth')
        tarfilepath = os.path.join(exp_folder, f'data.{datanum}.tar.gz')
        
        torch.save(data, filepath)
        with tarfile.open(tarfilepath, 'w:gz') as tar:
            tar.add(filepath, arcname=os.path.basename(filepath))

        os.remove(filepath)
        
        for i in range(len(data)):
            del data[i]
           
        
    def generate(self):
        print(f"Starting generation with {self.num_gpus} gpus")
        self.cfgs, self.processes, self.conns = zip(*[self.initialize_data_generator(i) for i in range(self.num_gpus)]) 
        max_buffer_len = self.config.data_gen.ckpt_freq
        exp_folder = self.config.data_gen.exp_folder

        os.makedirs(exp_folder, exist_ok=True)
        
        step = 0
        while self.N < self.config.data_gen.num_episodes:
            # TODO; update this to organize data by scene 
            for conn in self.conns:
                while conn.poll(timeout=1):
                    episode_stats, episode = conn.recv()
                    self.latest_episode_stats = episode_stats

                    # reformat episode
                    formatted_episode = self.reformat_episode(episode)              
                    self.buffer.append(formatted_episode) 

                    del episode

                    
            if len(self.buffer) >= max_buffer_len:
                data = self.buffer[:max_buffer_len]
                self.buffer = self.buffer[max_buffer_len:]
                
                # dump data into torch store
                datanum = self.N // max_buffer_len
                filepath = os.path.join(exp_folder, f'data.{datanum}.pth')
                torch.save(data, filepath)
               
                self.N += max_buffer_len
                gc.collect()
                

            time.sleep(10)
            step += 1
            self.writer.write({'step': step, 
                               'num_episodes': self.N,
                               'buffer_len': len(self.buffer),
                               **self.latest_episode_stats })

                
       
        
def main():
    parser = argparse.ArgumentParser(description="Example argparse for cfg_path")
    parser.add_argument('cfg_path', type=str, help="Path to the configuration file")
    args = parser.parse_args()

    generator = OfflineDataGenerator(args.cfg_path)
    generator.generate()

if __name__ == '__main__':
    main()