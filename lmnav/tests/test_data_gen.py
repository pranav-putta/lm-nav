from lmnav.data_gen import start_data_gen_process
from lmnav.data_gen import _init_envs, _create_obs_transforms, _setup_teacher

import os
import habitat

import time

def test_data_gen_process():
    device = 'cuda:0'

    process, queue = start_data_gen_process(device, deterministic=False)

    print("Starting queue iterations")
    for i in range(10):
        print(f"Iteration {i}: {queue.get()}")

        for j in range(10):
            print('.', end='')
            time.sleep(1)
        print()
    

if __name__ == '__main__':
    test_data_gen_process()
