import numpy as np
import cv2
import torch
import sys

x = sys.argv[1]

data = torch.load(f"data/datasets/lmnav/offline_10envs/data.{x}.pth")
images_tensor = np.array(data['rgb'])
tensor_shape = images_tensor.shape

# Mock data for the tensor of images

# Output video path
video_path = "t2.mp4"

# Create a video writer object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # codec for .mp4 format
fps = 3  # you can adjust this value based on your preference
video_writer = cv2.VideoWriter(video_path, fourcc, fps, (tensor_shape[2], tensor_shape[1]))

# Write each image in the tensor to the video
for img in images_tensor:
    bgr_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    video_writer.write(bgr_img)

video_writer.release()


