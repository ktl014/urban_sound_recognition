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
from data.d_utils import extract_features

# Module level constants
root_dir = '/Users/ktl014/Google Drive/ECE Classes/ECE 228 Machine Learning w: Physical Applications/urban_sound'
META_CSV = os.path.join(root_dir, 'dataset/UrbanSound8K/metadata/UrbanSound8K.csv')
IMAGE_DIR = os.path.join(root_dir, 'dataset/UrbanSound8K/audio')
assert os.path.exists(IMAGE_DIR), 'Invalid image directory'

DEBUG = True # Flag for quick development

def get_dataloader(batch_size, fold=[1], window_size=, input_size=224,
                   save=False, shuffle=True, num_workers=0):
    dataset = UrbanSoundDataset(fold,
                                parent_dir=IMAGE_DIR,
                                input_size=input_size,
                                save=save)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                shuffle=shuffle, num_workers=num_workers,
                                pin_memory=True)
    return loader

class UrbanSoundDataset(Dataset):
    """Custom dataset class for UrbanSound8K dataset"""
    def __init__(self, fold=1, parent_dir=IMAGE_DIR, input_size=224,
                 save=False):
        assert isinstance(fold, list) and list

        # Initialize
        self.fold = fold
        self.input_size = input_size

        fold_id = ''.join([str(i) for i in self.fold])
        if DEBUG:
            self.features = np.load(os.path.join(
                root_dir, 'data/fold{}_features.npy'.format(fold_id)))
            self.labels = np.load(os.path.join(
                root_dir, 'data/fold{}_labels.npy'.format(fold_id)))
        else:
            # Extracts features for all folds
            fold_str = ['fold{}'.format(i) for i in self.fold]
            self.features, self.labels = extract_features(parent_dir=parent_dir,
                                                          folds=fold_str,
                                                          bands=60,
                                                          frames=41)
            if save:
                outfile = os.path.join(root_dir, 'data/fold{}_{}.npy')
                np.save(outfile.format(fold_id, 'features'), self.features)
                np.save(outfile.format(fold_id, 'labels'), self.labels)

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
    fold = [3]

    # Initialize dataloader
    loader = get_dataloader(fold=fold, batch_size=batch_size)

    # Grab data at a single instance
    window, label = next(iter(loader))

    # Getting data from looped dataloader
    for i, (window, label) in enumerate(loader):

        for ii in range()
        window = window.numpy()
        lbl = label.numpy()
        print(i, window.shape, window.min(), window.max(), window.dtype)
        print(i, lbl.shape, lbl.min(), lbl.max(), lbl.dtype)
        break