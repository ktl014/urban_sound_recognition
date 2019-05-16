"""Dataloader for UrbanSound8K dataset"""
# Standard dist imports
import os

# Third party imports
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from torch.utils.data import Dataset
import torchvision.transforms as transforms

# Project level imports
from urban_sound_recognition.data.d_utils import extract_feature

# Module level constants
root_dir = '/Users/ktl014/Google Drive/ECE Classes/ECE 228 Machine Learning w: Physical Applications/urban_sound_recognition'
META_CSV = os.path.join(root_dir, 'dataset/UrbanSound8K/metadata/UrbanSound8K.csv')
IMAGE_DIR = os.path.join(root_dir, 'dataset/UrbanSound8K/audio/fold{}')

def get_dataloader(batch_size, fold=1, shuffle=True,
                   num_workers=4):
    assert fold in range(1, 11)
    dataset = UrbanSoundDataset(fold, image_dir=IMAGE_DIR, image_info=META_CSV)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                shuffle=shuffle, num_workers=num_workers,
                                pin_memory=True)
    return loader

class UrbanSoundDataset(Dataset):
    """Custom dataset class for UrbanSound8K dataset"""
    def __init__(self, fold=1, image_dir=IMAGE_DIR, image_info=META_CSV):
        self.fold = fold
        self.img_dir = image_dir.format(self.fold)

        meta = pd.read_csv(image_info)
        self.meta = meta[meta['fold'] == self.fold].reset_index(drop=True)

        fname = 'slice_file_name' # File name column
        self.meta[fname] = self.meta[fname].apply(lambda x: os.path.join(self.img_dir, x))

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, index):
        fname = self.meta.iloc[index]['slice_file_name']

        try:
            mfccs, chroma, mel, contrast, tonnetz = extract_feature(fname)
        except Exception as e:
            print("Error encountered while parsing file: ", fname)

        img = np.hstack([mfccs, chroma, mel, contrast, tonnetz])

        label = self.meta.iloc[index]['classID']
        return img, label

if __name__ == '__main__':
    batch_size = 16
    n_folds = 10
    for fold in range(1,n_folds+1):
        loader = get_dataloader(fold=fold, batch_size=batch_size)
        for i, (img, label) in enumerate(loader):
            img = img.numpy()
            lbl = label.numpy()
            print(i, img.shape, img.min(), img.max(), img.dtype)
            print(i, lbl.shape, lbl.min(), lbl.max(), lbl.dtype)
            break