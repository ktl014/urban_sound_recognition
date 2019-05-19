"""Dataloader for UrbanSound8K dataset"""
# Standard dist imports
import os

# Third party imports
import librosa
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

def get_dataloader(batch_size, fold=[1], shuffle=True,
                   num_workers=0):
    dataset = UrbanSoundDataset(fold, parent_dir=IMAGE_DIR)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                shuffle=shuffle, num_workers=num_workers,
                                pin_memory=True)
    return loader

class UrbanSoundDataset(Dataset):
    """Custom dataset class for UrbanSound8K dataset"""
    def __init__(self, fold=1, parent_dir=IMAGE_DIR, image_info=META_CSV):
        assert isinstance(fold, list) and list
        fold = ['fold{}'.format(i) for i in fold]
        self.features, self.labels = extract_features(parent_dir=parent_dir,
                                                      sub_dirs=fold,
                                                      bands=60,
                                                      frames=41)
        self.total_samples = len(self.features)

    def __len__(self):
        return self.total_samples

    def __getitem__(self, index):
        return self.features[index], self.labels[index]

if __name__ == '__main__':
    batch_size = 1
    fold = list(range(1,3))
    loader = get_dataloader(fold=fold, batch_size=batch_size)
    for i, (img, label) in enumerate(loader):
        img = img.numpy()
        lbl = label.numpy()
        print(i, img.shape, img.min(), img.max(), img.dtype)
        print(i, lbl.shape, lbl.min(), lbl.max(), lbl.dtype)
        break