from functools import partial

from lmnav.dataset.data_gen import start_data_gen_process

import unittest
from lmnav.config.default import get_config
from lmnav.common.registry import registry
from lmnav.common.actor_setups import *
from lmnav.dataset.filter_methods import *
    

class TestEpisodeProcessor(unittest.TestCase):
    
    def setUp(self):
        self.device = 'cuda:0'
        self.config = get_config("datagen/imagenav_data_gen_env744")
        
    def test_data_gen_process(self):
        filter_fn = partial(registry.get_fn(self.config.generator.filter_method._target_), self.config.generator.filter_method)
        setup_policy = registry.get_fn(self.config.generator.policy._target_)
        
        process, conn, queue = start_data_gen_process(self.config, setup_policy, filter_fn, deterministic=False)
        
        dataset = [queue.get() for _ in range(1)]
       
        conn.send("EXIT")
        conn.close()

        process.join()
        process.close()
 
        print("Collected episodes!")
        
        print(dataset) 
 
if __name__ == '__main__':
    unittest.main()
