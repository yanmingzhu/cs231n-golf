
import os.path as osp
import os
import cv2
import pandas as pd
import numpy as np

data_dir = "golfdb/data"
data_file = osp.join(data_dir, "golfDB.pkl")

count = np.zeros(9)
for idx in range(1400):
    df = pd.read_pickle(data_file)
    a = df.loc[idx, :]  # annotation info
    events = a['events']
    events -= events[0]  # now frame #s correspond to frames in preprocessed video clips
    count  = count + 1
    count[-1] += events[-1] # total frames = events[-1]+1


print(count)
print(count/count.sum())

