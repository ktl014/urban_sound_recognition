"""Dataset utility functions"""
# Standard dist imports

# Third party imports
import librosa

import scipy.io.wavfile as wav
from numpy.lib import stride_tricks

import numpy as np
import torch
from torch.autograd import Variable

# Project level imports

# Module level constants

def load_sound_files(file_paths):
    raw_sounds = []
    for fp in file_paths:
        X,sr = librosa.load(fp)
        raw_sounds.append(X)
    return raw_sounds

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


def windows(data, window_size):
    start = 0
    while start < len(data):
        yield start, start + window_size
        start += (window_size / 2)

def extract_features(parent_dir, sub_dirs, file_ext="*.wav", bands=60,
                     frames=41, print_freq=10):
    import glob
    import os
    window_size = 512 * (frames - 1)
    log_specgrams = []
    labels = []
    for l, sub_dir in enumerate(sub_dirs):
        files = glob.glob(os.path.join(parent_dir, sub_dir, file_ext))
        for i, fn in enumerate(files):
            # Read in file
            # samplerate, sound_clip = wav.read(fn)
            if i % print_freq == 0:
                print('{}: {}/{}'.format(sub_dir, i, len(files)))

            sound_clip, samplerate = librosa.load(fn)

            # Get label
            label = os.path.basename(fn).split('-')[1]

            # Window sound clip
            for (start, end) in windows(sound_clip, window_size):
                start, end = int(start), int(end)
                if (len(sound_clip[start:end]) == window_size):
                    signal = sound_clip[start:end]

                    # Spectrogram processing
                    s = stft(signal, 2**10)
                    sshow, freq = logscale_spec(s, factor=1.0, sr=samplerate)
                    ims = 20. * np.log10(np.abs(sshow) / 10e-6)

                    log_specgrams.append(ims)
                    labels.append(label)

    return np.array(log_specgrams), np.array(labels, dtype=np.int)


""" short time fourier transform of audio signal """
def stft(sig, frameSize, overlapFac=0.5, window=np.hanning):
    win = window(frameSize)
    hopSize = int(frameSize - np.floor(overlapFac * frameSize))

    samples = np.append(np.zeros(int(np.floor(frameSize / 2.0))), sig)
    cols = int(np.ceil((len(samples) - frameSize) / float(hopSize))) + 1
    samples = np.append(samples, np.zeros(frameSize))

    frames = stride_tricks.as_strided(samples, shape=(cols, frameSize),
                                      strides=(samples.strides[0] * hopSize,
                                               samples.strides[0])).copy()
    frames *= win

    return np.fft.rfft(frames)


""" scale frequency axis logarithmically """
def logscale_spec(spec, sr=44100, factor=20.):
    timebins, freqbins = np.shape(spec)

    scale = np.linspace(0, 1, freqbins) ** factor
    scale *= (freqbins - 1) / max(scale)
    scale = np.unique(np.round(scale))

    newspec = np.complex128(np.zeros([timebins, len(scale)]))
    for i in range(0, len(scale)):
        if i == len(scale) - 1:
            newspec[:, i] = np.sum(spec[:, int(scale[i]):], axis=1)
        else:
            newspec[:, i] = np.sum(spec[:, int(scale[i]):int(scale[i + 1])],
                                   axis=1)

    allfreqs = np.abs(np.fft.fftfreq(freqbins * 2, 1. / sr)[:freqbins + 1])
    freqs = []
    for i in range(0, len(scale)):
        if i == len(scale) - 1:
            freqs += [np.mean(allfreqs[int(scale[i]):])]
        else:
            freqs += [np.mean(allfreqs[int(scale[i]):int(scale[i + 1])])]

    return newspec, freqs
