import cv2
import albumentations as A
import numpy as np
import os

def transform_video(transform, input_file, out_file):
    cap = cv2.VideoCapture(input_file)
    if not cap.isOpened():
        raise IOError("Error opening video file")

    # --- Get video properties ---
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    print(f"file {input_file}, fps {fps}, width {frame_width}, height {frame_height}")

    # --- Set up video writer ---
    out = cv2.VideoWriter(out_file, fourcc, fps, (frame_width, frame_height))

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


def augment(name, transform, src_dir, target_dir):
    folder_path = os.path.join(target_dir, name)
    os.makedirs(folder_path, exist_ok=True)

    for file in sorted(os.listdir(src_dir)):
        src_file = os.path.join(src_dir, file)
        if not os.path.isfile(src_file):
            continue
        target_file = os.path.join(target_dir, name, file)
        transform_video(transform, src_file, target_file)

transform = A.Compose([
#    A.GaussNoise(std_range=(0.1,0.1))
#    A.Sharpen()
#    A.HorizontalFlip(p=1),
#    A.ChannelShuffle(p=1)
#    A.RandomBrightnessContrast(p=0.2),
#   A.MotionBlur(blur_limit=5, p=0.3),
#    A.Rotate(limit=(5,5), p=1)
])

augmentations = {
    "horizonal_flip": A.Compose([A.HorizontalFlip(p=1)]),
    "channel_shuffle": A.Compose([A.ChannelShuffle(p=1)]),
    "rotate_5": A.Rotate(limit=(5,5), p=1),
    "noise_0.2": A.GaussNoise(std_range=(0.2,0.2))
}

src_dir = './data/videos_160'
target_dir = './data/augmented_data'
for name, transform in augmentations.items():
    augment(name, transform, src_dir=src_dir, target_dir=target_dir)