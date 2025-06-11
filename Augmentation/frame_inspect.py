import cv2
import albumentations as A
import numpy as np
import os

def check_video(input_file):
    cap = cv2.VideoCapture(input_file)
    if not cap.isOpened():
        raise IOError("Error opening video file")

    # --- Get video properties ---
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"file {input_file}, frames {length}, fps {fps}, width {frame_width}, height {frame_height}")

    # --- Cleanup ---
    cap.release()
    return length


def inspect(src_dir):
    frame_count = []
    for file in sorted(os.listdir(src_dir)):
        src_file = os.path.join(src_dir, file)
        if not os.path.isfile(src_file):
            continue
        frame_count.append(check_video(src_file))
    return frame_count


src_dir = './data/videos_160'
fc = np.array(inspect(src_dir=src_dir))
print(f"avg {fc.mean()}, min {fc.min()}, max {fc.max()}")
print(f"total:{len(fc)} >96:{len(np.where(fc>96)[0])} >128:{len(np.where(fc>128)[0])} >148:{len(np.where(fc>148)[0])} >192:{len(np.where(fc>192)[0])} >256:{len(np.where(fc>256)[0])} >288:{len(np.where(fc>288)[0])} >304:{len(np.where(fc>304)[0])} >320:{len(np.where(fc>320)[0])} >512:{len(np.where(fc>512)[0])}")

