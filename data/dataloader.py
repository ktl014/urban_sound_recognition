"""Dataloader for UrbanSound8K dataset"""
# Standard dist imports
import os

# Third party imports
import librosa
import cv2
import numpy as np
import pandas as pd
import pickle
import scipy.io.wavfile as wav
import torch
import torch.utils.data as data
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from sklearn.preprocessing import LabelBinarizer

# Project level imports
from data.d_utils import extract_features

# Module level constants
DEBUG = False # Flag for quick development
WINDOW_SIZE = 2**10
INPUT_SIZE = 224
DATA_FNAME = 'data/193_features.p'

def get_dataloader_v2(db_prepped=False, traintest_split=6984):
    """Get dataloader

    It will return the training and test set, the label encoder, and the data columns

    Args:
        db_prepped:
        traintest_split:

    Returns:

    """
    if db_prepped:
        data = pickle.load(open(DATA_FNAME, "rb"))
        print(f'Data loaded: {type(data)}, {data.shape}, {data.columns}' if DEBUG else "") 
        
        # Get extracted features and labels (i.e. filter out irrelevant data points)
        samples = pd.DataFrame(list(data['sample']))
        data_cols = samples.columns
        samples['label'] = data['label']
        
        # Train test split
        train = samples[:traintest_split]
        test = samples[traintest_split:]
        print(f'Train: {train.shape} | Test: {test.shape}')
        
        # Initialize one hot encoder
        LB = LabelBinarizer().fit(train['label'])

        return train, test, LB, data_cols

def get_dataloader(batch_size, fold=1, db_prepped=False,
                   window_size=WINDOW_SIZE, input_size=INPUT_SIZE,
                   shuffle=True, num_workers=0, save=False,
                   quick_dev=False):
    dataset = UrbanSoundDataset(fold,
                                input_size=input_size,
                                window_size=window_size,
                                db_prepped=db_prepped,
                                save=save, quick_dev=quick_dev)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                shuffle=shuffle, num_workers=num_workers,
                                pin_memory=True)
    return loader

class UrbanSoundDataset(Dataset):
    """Custom dataset class for UrbanSound8K dataset"""
    def __init__(self, fold=1, db_prepped=False,
                 window_size=WINDOW_SIZE, input_size=INPUT_SIZE,
                 save=False, quick_dev=False):
        assert isinstance(fold, list) and list

        # Initialize attributes
        self.fold = fold
        self.input_size = input_size
        self.window_size = window_size

        fold_id = ''.join([str(i) for i in self.fold])
        if db_prepped:
            #TODO make smarter flag for checking if preprocessed features exist
            print('Loading {}'.format(self.fold))
            
            self.features = pickle.load(open(os.path.join('data/folds/fold{}_features.p'.format(fold_id)), 'rb'))
            self.labels = pickle.load(open(os.path.join('data/folds/fold{}_labels.p'.format(fold_id)), 'rb'))
        else:
            # Extracts features for all folds
            fold_str = ['fold{}'.format(i) for i in self.fold]
            image_dir = os.path.join('dataset/UrbanSound8K/audio')
            self.features, self.labels = extract_features(image_dir=image_dir,
                                                          folds=fold_str,
                                                          bands=60,
                                                          frames=41, quick_dev=quick_dev)
            if save:
                print('Saving files')
                outfile = os.path.join('data/folds/fold{}_{}.p')
                pickle.dump(self.features, open(outfile.format(fold_id, 'features'), 'wb'))
                pickle.dump(self.labels, open(outfile.format(fold_id, 'labels'), 'wb'))

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
