
import os.path as osp
import os
import cv2
import pandas as pd
import numpy as np

data_dir = "golfdb/data"
video_dir = osp.join(data_dir, 'videos_160')
data_file = osp.join(data_dir, "golfDB.pkl")
output_dir = "image_dump/images"

def dump_video_frames(data_file, idx, nb=2):
    df = pd.read_pickle(data_file)
    a = df.loc[idx, :]  # annotation info
    events = a['events']
    events -= events[0]  # now frame #s correspond to frames in preprocessed video clips

    video_file = osp.join(video_dir, '{}.mp4'.format(a['id']))
    print(f"video_file {video_file}")
    cap = cv2.VideoCapture(video_file)

    events = events[1:-1]
    print(f"events {events}")

    for event_frame in events:
        for frame_number in range(event_frame-nb, event_frame+nb+1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            output_file = osp.join(output_dir, '{}.mp4_{}.jpg'.format(a['id'], frame_number))
            ret, frame = cap.read()
            if ret:
                x = 1
                cv2.imwrite(output_file, frame)
            else:
                print(f"Failed to read frame {frame_number}")

    cap.release()    

dump_video_frames(data_file,  257)