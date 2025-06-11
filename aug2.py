import cv2
import torch
from torchvision import transforms
from torchvision.io import read_video, write_video
import torchvision.transforms.functional as F
import os

# Define augmentation (e.g., random horizontal flip and color jitter)
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=1.0),
    transforms.ColorJitter(brightness=0.5, contrast=0.5)
])

# Load video using torchvision (returns video tensor [T, H, W, C])
video_path = '0.mp4'
video, _, info = read_video(video_path, pts_unit='sec')  # video: [T, H, W, C], dtype=torch.uint8

video = video.permute(0,3,1, 2) #yz
# Apply transform to each frame
augmented_frames = torch.stack([transform(F.to_pil_image(frame)).convert('RGB') for frame in video])
#augmented_frames = augmented_frames.permute(0, 3, 1, 2)  # [T, C, H, W] #yz
augmented_frames = augmented_frames.to(torch.uint8)

# Save the augmented video
output_path = 'augmented_00.mp4'
fps = info['video_fps']
write_video(output_path, augmented_frames, fps)

print(f"Augmented video saved to {output_path}")
