"""Dataset utility functions"""
# Standard dist imports

# Third party imports
import cv2
import librosa
import random

import scipy.io.wavfile as wav
from numpy.lib import stride_tricks

import numpy as np
import torch
from torch.autograd import Variable

# Project level imports

# Module level constants

def load_sound_file(file_path):
    samplerate, sound_clip = wav.read(file_path)
    return samplerate, sound_clip


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

def extract_features(image_dir, folds, file_ext="*.wav", bands=60,
                     frames=41, print_freq=10, quick_dev=False, ):
    import glob
    import os
    window_size = 512 * (frames - 1)
    window_features = []
    labels = []
    bad_count = 0

    n_fft = 1024
    hop_length = 256
    n_mels = 40
    f_min = 0
    f_max = 8000
    s_r = 16000

    for l, fold in enumerate(folds):
        files = glob.glob(os.path.join(image_dir, fold, file_ext))
        if quick_dev:
            files = random.sample(files, 200)

        for i, fn in enumerate(files):
            # Read in file
            try:
                # samplerate, sound_clip = wav.read(fn)
                sound_clip, samplerate = librosa.load(fn, sr=s_r)

            except:
                bad_count += 1
                print("BAD AUDIO INPUT!! fn:{} {}".format(os.path.basename(
                    fn), bad_count))
                continue

            if i % print_freq == 0:
                print('{}: {}/{}'.format(fold, i, len(files)))


            # Stretch sound
            # length = librosa.get_duration(sound_clip)
            # if (length < 4.0):
            #     sound_clip = stretch_sound(sound_clip, length)

            # Get label
            label = os.path.basename(fn).split('-')[1]

            # Window sound clip
            log_specgrams = []
            for (start, end) in windows(sound_clip, window_size):
                start, end = int(start), int(end)
                if (len(sound_clip[start:end]) == window_size):
                    signal = sound_clip[start:end]

                    # Spectrogram processing
                    ims = librosa.feature.melspectrogram(y=signal, sr=s_r, n_fft=n_fft, hop_length=hop_length,
                                                         power=1.0, n_mels=n_mels,fmin=f_min,fmax=f_max)

                    # Resize
                    ims = cv2.resize(ims, dsize=(224, 224))

                    # Stack grayscale into three channels for CNN
                    ims = np.broadcast_to(ims, (3,) + ims.shape)

                    log_specgrams.append(ims)

            labels.append(label)
            window_features.append(log_specgrams)

    return np.array(window_features), np.array(labels, dtype=np.int)

def stretch_sound(sound, length):
    import math
    repetition = math.ceil(4.0 / length)
    sound_new = (sound * (int)(repetition))  # extend the sound clip to atleast 4 secs
    sound_new = sound_new[:4000]  # crop sound to exactly 4 secs
    return sound_new

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
