from transformers import CLIPModel, CLIPProcessor, CLIPVisionModel
import torch
import unittest 
from lmnav.common.utils import catchtime
from lmnav.models.vis_encoders import CLIPVisualProcessor, CustomCLIPVisualProcessor
from matplotlib import pyplot as plt
import gzip
import pickle
from PIL import Image

class TestCLIPProcessor(unittest.TestCase):

    def setUp(self):
        self.clip_processor = CLIPVisualProcessor(CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32"))
        self.custom_processor = CustomCLIPVisualProcessor()
        self.B = 13
        self.num_trials = 1
        
        with open('data/datasets/lmnav/offline_10envs/data.0.pth', 'rb') as f:
            self.images = torch.load(f)['rgb'].permute(3, 0, 1, 2).float()
        print("loaded images with shape", self.images.shape)

    def test_clip_processor_time(self):
        with catchtime("CLIP Processor"):
            for _ in range(self.num_trials):
                images = self.clip_processor.transform(self.images)
                assert images.shape == (3, self.B, 224, 224)
                
    def test_custom_clip_processor_time(self):
        with catchtime("Custom CLIP Processor"):
            for _ in range(self.num_trials):
                images = self.custom_processor.transform(self.images)
                assert images.shape == (3, self.B, 224, 224) 

    def test_diffs(self):
        img1 = self.clip_processor.transform(self.images.clone())
        img2 = self.custom_processor.transform(self.images.clone())

        def save_img(img, name):
            img = Image.fromarray(img[:, 0].permute(1, 2, 0).to(torch.uint8).numpy() * 255)
            img.save(name)

        print("L1 norm", (img1 - img2).sum() / self.B)

        save_img(img1, "clip.png")
        save_img(img2, "custom.png")

if __name__ == "__main__":
    unittest.main()

