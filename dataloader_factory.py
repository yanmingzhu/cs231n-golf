from dataloader2 import GolfDB2, Normalize, ToTensor
from torch.utils.data import DataLoader
from torchvision import transforms
import cv2
import numpy as np
from golfdb.dataloader import GolfDB
from detectors.YoloPoseDetector import YoloPoseDetector

root = '/content'

def get_transformations(model):
    transformations = [ToTensor()]
    if not isinstance(model, (YoloPoseDetector)):
        transformations.insert(0, ResizeImage())
        transformations.append(Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    print(f"transformation {len(transformations)}")
    return transformations

def get_training_data_loader(model, batch_size=22, seq_length=64, data_remote=False):
    split=1

    if data_remote:
        dataset = GolfDB(data_file='data/train_split_1.pkl',
                        vid_dir='data/video_160',
                        seq_length=seq_length,
                        transform=transforms.Compose(get_transformations(model)),
                        train=True)
    else:
        dataset = GolfDB2(data_file='{}/data/train_split_{}.pkl'.format(root, split),
                        root_vid_dir='/content/data',
                        seq_length=seq_length,
                        transform=transforms.Compose(get_transformations(model)),
                        train=True)

    n_cpu=6
    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=n_cpu,
                             drop_last=True)
    print("dataloader completed")

    return data_loader

def get_test_data_loader(model, batch_size=22, seq_length=64):
    split=1

    dataset = GolfDB(data_file='golfdb/data/val_split_{}.pkl'.format(split),
                     vid_dir='golfdb/data/videos_160',
                     seq_length=seq_length,
                     transform=transforms.Compose(get_transformations(model)),
                     train=False)

    print(f"dataset len {len(dataset)}")
    n_cpu=6
    data_loader = DataLoader(dataset,
                             batch_size=1,
                             shuffle=False,
                             num_workers=n_cpu,
                             drop_last=False)
    
    return data_loader

def get_val_data_loader(model, batch_size=22, seq_length=64):
    split=1

    dataset = GolfDB(data_file='golfdb/data/val_split_{}.pkl'.format(split),
                     vid_dir='data/augmented_data/rotate_5',
                     seq_length=seq_length,
                     transform=transforms.Compose(get_transformations(model)),
                     train=False)

    print(f"dataset len {len(dataset)}")
    n_cpu=6
    data_loader = DataLoader(dataset,
                             batch_size=1,
                             shuffle=False,
                             num_workers=n_cpu,
                             drop_last=False)
    
    return data_loader

class ResizeImage():
    def  __init__(self, size=224):
        self.size = size

    def __call__(self, sample):
        if sample != None:
            output = [cv2.resize(img, (self.size, self.size)) for img in sample['images']]
            output = np.array(output)
            sample["images"] = output
            #print(f"output {output.shape}")
        return sample