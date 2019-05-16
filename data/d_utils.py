"""Dataset utility functions"""
# Standard dist imports

# Third party imports
import librosa
import numpy as np
from torch.autograd import Variable

# Project level imports

# Module level constants

def load_sound_files(file_paths):
    raw_sounds = []
    for fp in file_paths:
        X,sr = librosa.load(fp)
        raw_sounds.append(X)
    return raw_sounds

def extract_feature(file_name):
    X, sample_rate = librosa.load(file_name)
    stft = np.abs(librosa.stft(X))
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),
    sr=sample_rate).T,axis=0)
    return mfccs,chroma,mel,contrast,tonnetz



def to_cuda(item, computing_device, label=False):
    """ Typecast item to cuda()
    Wrapper function for typecasting variables to cuda() to allow for
    flexibility between different types of variables (i.e. long, float)
    Loss function usually expects LongTensor type for labels, which is why
    label is defined as a bool.
    Computing device is usually defined in the Trainer()
    Args:
        item: Desired item. No specific type
        computing_device (str): Desired computing device.
        label (bool): Flag to convert item to long() or float()
    Returns:
        item
    """
    if label:
        item = Variable(item.to(computing_device)).long()
    else:
        item = Variable(item.to(computing_device)).float()
    return item