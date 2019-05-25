"""Dataloader for UrbanSound8K dataset"""
# Standard dist imports
import os

# Third party imports
import librosa
import cv2
import numpy as np
import pandas as pd
import scipy.io.wavfile as wav
import torch
import torch.utils.data as data
from torch.utils.data import Dataset
import torchvision.transforms as transforms

# Project level imports
from data.d_utils import extract_features

# Module level constants
DEBUG = True # Flag for quick development
WINDOW_SIZE = 2**10
INPUT_SIZE = 224

def get_dataloader(batch_size, fold=1, db_prepped=False,
                   window_size=WINDOW_SIZE, root_dir=None,
                   input_size=INPUT_SIZE, shuffle=True, num_workers=0, save=False):
    dataset = UrbanSoundDataset(fold,
                                root_dir=root_dir,
                                input_size=input_size,
                                window_size=window_size,
                                db_prepped=db_prepped,
                                save=save)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                shuffle=shuffle, num_workers=num_workers,
                                pin_memory=True)
    return loader

class UrbanSoundDataset(Dataset):
    """Custom dataset class for UrbanSound8K dataset"""
    def __init__(self, fold=1, db_prepped=False,
                 window_size=WINDOW_SIZE, input_size=INPUT_SIZE,
                 root_dir=None, save=False):
        assert isinstance(fold, list) and list

        # Initialize attributes
        self.fold = fold
        self.input_size = input_size
        self.window_size = window_size

        fold_id = ''.join([str(i) for i in self.fold])
        if db_prepped:
            #TODO make smarter flag for checking if preprocessed features exist
            print('Loading {}'.format(self.fold))
            self.features = np.load(os.path.join(
                root_dir, 'data/folds/fold{}_features.npy'.format(fold_id)),
            allow_pickle=True)
            self.labels = np.load(os.path.join(
                root_dir, 'data/folds/fold{}_labels.npy'.format(fold_id)),
                allow_pickle=True)
        else:
            # Extracts features for all folds
            fold_str = ['fold{}'.format(i) for i in self.fold]
            image_dir = os.path.join(root_dir, 'dataset/UrbanSound8K/audio')
            self.features, self.labels = extract_features(image_dir=root_dir,
                                                          folds=fold_str,
                                                          bands=60,
                                                          frames=41)
            if save:
                print('Saving files')
                outfile = os.path.join(root_dir, 'data/folds/fold{}_{}.npy')
                np.save(outfile.format(fold_id, 'features'), self.features)
                np.save(outfile.format(fold_id, 'labels'), self.labels)

        self.total_samples = len(self.features)

        # self.transform = transforms.Compose([transforms.Resize(input_size)])

    def __len__(self):
        return self.total_samples

    def __getitem__(self, index):
        return np.array(self.features[index]), self.labels[index]

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
    loader = get_dataloader(fold=fold, batch_size=batch_size,
                            db_prepped=True, root_dir=os.path.dirname(os.getcwd()))

    """Uncomment when debugging the dataloader"""
    # Grab data at a single instance
    window, label = next(iter(loader))

    # Getting data from looped dataloader
    for i, (window, label) in enumerate(loader):

        for ii in range(window):
            img = window.numpy()[:, i, :, :, :]
            lbl = label.numpy()
            print(ii, window.shape, window.min(), window.max(), window.dtype)
            print(ii, img.shape, img.min(), img.max(), img.dtype)
            print(ii, lbl.shape, lbl.min(), lbl.max(), lbl.dtype)
            break
