import os.path as osp
import os
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import random

def sample_transformation(root_vid_dir):
    original = "videos_160"
    augment_dir = os.path.join(root_vid_dir, "augmented_data")
    all_vid_dirs = []
    all_vid_dirs.append(os.path.join(root_vid_dir, original))
    '''
    for dirpath, dirnames, _ in os.walk(augment_dir):
        for dirname in dirnames:
            full_path = os.path.join(dirpath, dirname)
            all_vid_dirs.append(full_path)
    '''

    '''
    all_vid_dirs.append(os.path.join(augment_dir, 'channel_shuffle'))
    all_vid_dirs.append(os.path.join(augment_dir, 'horizonal_flip'))
    all_vid_dirs.append(os.path.join(augment_dir, 'noise_0.2'))
    all_vid_dirs.append(os.path.join(augment_dir, 'rotate_5'))
    '''

    return random.choice(all_vid_dirs)

class GolfDB2(Dataset):
    def __init__(self, data_file, root_vid_dir, seq_length, transform=None, train=True, wrap=False):
        self.df = pd.read_pickle(data_file)
        self.root_vid_dir = root_vid_dir
        self.seq_length = seq_length
        self.transform = transform
        self.train = train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        a = self.df.loc[idx, :]  # annotation info
        events = a['events']
        events -= events[0]  # now frame #s correspond to frames in preprocessed video clips

        images, labels = [], []
        transformation_dir = sample_transformation(self.root_vid_dir)
        video_dir = osp.join(transformation_dir, '{}.mp4'.format(a['id']))
        cap = cv2.VideoCapture(video_dir)
        #print(f"reading {video_dir}")

        if self.train:
            # random starting position, sample 'seq_length' frames
            start_frame = np.random.randint(events[-1] + 1)

            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            end_sample_idx = max(0, frame_count - self.seq_length)
            half = end_sample_idx // 2
            weight1 = np.linspace(2, 1, half)
            weight2 = np.linspace(1, 2, end_sample_idx+1-half)
            weights = np.concatenate((weight1, weight2))
            #print(weights)
            weights /= weights.sum()
            sample_idx = np.arange(end_sample_idx+1)
            #print(f"sample_idx {sample_idx.shape} weights {weights.shape}")

            start_frame = np.random.choice(sample_idx, p=weights)
            #print(f"start_frame {start_frame} {type(start_frame)}")
            #start_frame = random.randint(0, max(0, frame_count - self.seq_length))
            sample_len = min(self.seq_length, frame_count - start_frame)
            #print(f"frame count {events[-1]+1}, seqlen {self.seq_length}, start_frame {start_frame} margin {events[-1] - start_frame - self.seq_length}")
            #start_frame = 0
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            pos = start_frame
            while len(images) < self.seq_length:
                ret, img = cap.read()
                if ret:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    images.append(img)
                    if pos in events[1:-1]:
                        labels.append(np.where(events[1:-1] == pos)[0][0])
                    else:
                        labels.append(8)
                    pos += 1
                else:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    pos = 0
            cap.release()
        else:
            # full clip
            for pos in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
                _, img = cap.read()
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(img)
                if pos in events[1:-1]:
                    labels.append(np.where(events[1:-1] == pos)[0][0])
                else:
                    labels.append(8)
            cap.release()

        sample = {'images':np.asarray(images), 'labels':np.asarray(labels)}
        if self.transform:
            sample = self.transform(sample)
        sample['length'] = sample_len
        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        images, labels = sample['images'], sample['labels']
        images = images.transpose((0, 3, 1, 2))
        return {'images': torch.from_numpy(images).float().div(255.),
                'labels': torch.from_numpy(labels).long()}


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)

    def __call__(self, sample):
        images, labels = sample['images'], sample['labels']
        images.sub_(self.mean[None, :, None, None]).div_(self.std[None, :, None, None])
        return {'images': images, 'labels': labels}


if __name__ == '__main__':

    norm = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet mean and std (RGB)

    dataset = GolfDB2(data_file='data/train_split_1.pkl',
                     root_vid_dir='data',
                     seq_length=200,
                     transform=transforms.Compose([ToTensor(), norm]),
                     train=True)

    data_loader = DataLoader(dataset, batch_size=3, shuffle=False, num_workers=1, drop_last=False)

    for i, sample in enumerate(data_loader):
        images, labels, length = sample['images'], sample['labels'], sample['length']
        print(f"images, {images.shape} {labels.shape}")
        events = np.where(labels.squeeze() < 8)[0]
        print('sample_len {}, {} events: {}'.format(length.numpy(), len(events), events))




    





       

