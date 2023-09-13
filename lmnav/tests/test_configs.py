import unittest
from lmnav.config.default import get_config

   
class TestConfig(unittest.TestCase):
    
    # def test_datagen_config(self):
        # config = get_config("datagen/test_imagenav_data_gen_env744")        
        # print(config)

    def test_llama_config(self):
        config = get_config("train/nav_llama/q_compression/4_tokens_1env.yaml")
        # config = get_config("train/nav_llama/lora/1env.yaml")
        import pdb
        pdb.set_trace()
        print(config)
        
 
if __name__ == '__main__':
    unittest.main()
