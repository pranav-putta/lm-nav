from functools import partial

from lmnav.dataset.data_gen import start_data_gen_process
from hydra.utils import instantiate

import habitat
import unittest
from lmnav.config.default import get_config
from lmnav.common.registry import registry
from lmnav.common.actor_setups import *
   
    

class TestEpisodeProcessor(unittest.TestCase):
    
    def setUp(self):
        self.device = 'cuda:0'
        self.config = get_config("datagen/test_imagenav_data_gen_env744")

    def test_data_gen_process(self):
        filter_fn = partial(dtg_filter_fn, self.config.generator.dtg_threshold)
        setup_actor = registry.get_fn(f"actors.{self.config.generator.actor.name}")
        
        process, conn, queue = start_data_gen_process(self.config, setup_actor, filter_fn, deterministic=False)
        
        dataset = [queue.get() for _ in range(1)]
       
        conn.send("EXIT")
        conn.close()

        process.join()
        process.close()
 
        print("Collected episodes!")
        
        print(dataset) 
 
if __name__ == '__main__':
    unittest.main()
