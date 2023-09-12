import unittest
from lmnav.config.default import get_config

   
class TestConfig(unittest.TestCase):
    
    def test_config(self):
        config = get_config("datagen/test_imagenav_data_gen_env744")        
        print(config)
        
 
if __name__ == '__main__':
    unittest.main()
