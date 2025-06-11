import cv2
import albumentations as A
import numpy as np

# --- Define your augmentation pipeline ---
transform = A.Compose([
#    A.GaussNoise(std_range=(0.2,0.2))
#    A.Sharpen()
#    A.HorizontalFlip(p=1),
#    A.ChannelShuffle(p=1)
    A.ChannelDropout()
#    A.RandomBrightnessContrast(p=0.2),
#   A.MotionBlur(blur_limit=5, p=0.3),
#    A.Rotate(limit=(5,5), p=1)
])

# --- Input/output video paths ---
input_path = 'input_video.mp4'
output_path = 'output_augmented.mp4'

# --- Open video reader ---
cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    raise IOError("Error opening video file")

# --- Get video properties ---
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# --- Set up video writer ---
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# --- Process frame-by-frame ---
frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert BGR (OpenCV) to RGB (Albumentations)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Apply augmentations
    augmented = transform(image=image)
    aug_img = augmented['image']

    # Convert back to BGR for OpenCV
    aug_bgr = cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR)

    # Write to output video
    out.write(aug_bgr)
    frame_count += 1

print(f"✅ Augmented video saved. Total frames: {frame_count}")

# --- Cleanup ---
cap.release()
out.release()
cv2.destroyAllWindows()
