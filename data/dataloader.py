"""Dataloader for UrbanSound8K dataset"""
# Standard dist imports
import os

# Third party imports
import librosa
import cv2
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from torch.utils.data import Dataset
import torchvision.transforms as transforms

# Project level imports
from d_utils import extract_features

# Module level constants
root_dir = '/Users/ktl014/Google Drive/ECE Classes/ECE 228 Machine Learning w: Physical Applications/urban_sound_recognition'
META_CSV = os.path.join(root_dir, 'dataset/UrbanSound8K/metadata/UrbanSound8K.csv')
IMAGE_DIR = os.path.join(root_dir, 'dataset/UrbanSound8K/audio')
assert os.path.exists(IMAGE_DIR), 'Invalid image directory'

DEBUG = True # Flag for quick development

def get_dataloader(batch_size, fold=[1], shuffle=True,
                   num_workers=0):
    dataset = UrbanSoundDataset(fold, parent_dir=IMAGE_DIR)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                shuffle=shuffle, num_workers=num_workers,
                                pin_memory=True)
    return loader

class UrbanSoundDataset(Dataset):
    """Custom dataset class for UrbanSound8K dataset"""
    def __init__(self, fold=1, parent_dir=IMAGE_DIR, input_size=224):
        assert isinstance(fold, list) and list
        fold = ['fold{}'.format(i) for i in fold]

        # Initialize
        self.input_size = input_size

        if DEBUG and 'fold1' in fold:
            self.features = np.load(os.path.join(root_dir, 'data/fold1_features.npy'))
            self.labels = np.load(os.path.join(root_dir, 'data/fold1_labels.npy'))
        else:
            self.features, self.labels = extract_features(parent_dir=parent_dir,
                                                          sub_dirs=fold,
                                                          bands=60,
                                                          frames=41)
        self.total_samples = len(self.features)

        # self.transform = transforms.Compose([transforms.Resize(input_size)])

    def __len__(self):
        return self.total_samples

    def __getitem__(self, index):
        # < TEMP: cv2 > couldn't resize using pytorch's resize bc not image
        img = cv2.resize(self.features[index], dsize=(self.input_size,
                                                      self.input_size))
        return img, self.labels[index]

if __name__ == '__main__':
    """Example for implementing dataloader in train and evaluation
    
    Initializing train (1) and val (3) loader with shuffled batch size of 1
    
    >> train_loader = get_dataloader(fold=[1], batch_size=1, shuffle=True)
    >> val_loader = get_dataloader(fold=[3], batch_size=1, shuffle=True)
    """
    # Initialize parameters
    batch_size = 1
    fold = [1]

    # Initialize dataloader
    loader = get_dataloader(fold=fold, batch_size=batch_size)

    # Grab data at a single instance
    img, label = next(iter(loader))

    # Getting data from looped dataloader
    for i, (img, label) in enumerate(loader):
        img = img.numpy()
        lbl = label.numpy()
        print(i, img.shape, img.min(), img.max(), img.dtype)
        print(i, lbl.shape, lbl.min(), lbl.max(), lbl.dtype)
        break