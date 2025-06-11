import torch
import torchvision.io as io
import torchvision.transforms as transforms

# 1. Load the video
video_path = "0.mp4" # Replace with your video file
video, _, _ = io.read_video(video_path) # video is a tensor of shape (T, H, W, C)
print(f"video {video.shape}")

# 2. Define augmentations
transform = transforms.Compose([
    transforms.ToPILImage(), # Convert tensor to PIL Image for augmentation
    transforms.RandomHorizontalFlip(p=0.5), # Example augmentation: horizontal flip
    transforms.RandomRotation(degrees=10), # Example augmentation: random rotation
    transforms.ToTensor(), # Convert PIL Image back to tensor
])

# Apply augmentation to each frame individually
augmented_frames = []
for frame in video:
    augmented_frame = transform(frame)
    augmented_frames.append(augmented_frame)
augmented_video = torch.stack(augmented_frames) # Stack augmented frames back into video tensor

# 3. Write the augmented video
output_path = "aug_0.mp4"
fps = 30 # Set your desired FPS
io.write_video(output_path, augmented_video.permute(0, 2, 3, 1).to(torch.uint8), fps=fps) # Ensure correct format (T, H, W, C) and type (uint8) for write_video
print(f"Augmented video saved to {output_path}")
