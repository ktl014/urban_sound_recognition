# Import libraries
import sys
import numpy as np
import pandas as pd
import sklearn as sk
from os import listdir
from os.path import isfile, join
from timeit import default_timer as timer

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from six.moves import cPickle as pickle
from six.moves import range

import librosa
import soundfile as sf
from python_speech_features import mfcc
from python_speech_features import logfbank

def extract_feature(file_name: str) -> tuple:
    """
    Extracts 193 chromatographic features from sound file. 
    including: MFCC's, Chroma_StFt, Melspectrogram, Spectral Contrast, and Tonnetz
    NOTE: this extraction technique changes the time series nature of the data
    """
    X, sample_rate = librosa.load(file_name)
    stft = np.abs(librosa.stft(X))
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
    return mfccs,chroma,mel,contrast,tonnetz


raw_sound = pd.read_csv('../dataset/UrbanSound8K/metadata/UrbanSound8K.csv')
fold_list = ['fold1', 'fold2', 'fold3', 'fold4', 'fold5', 'fold6', 'fold7', 'fold8', 'fold9', 'fold10']
mfcc_data = []
exception_count = 0

start_time = timer()
for i in range(10):
    # get file names
    mypath = 'dataset/UrbanSound8K/audio/'+ fold_list[i] + '/'
    files = [mypath + f for f in listdir(mypath) if isfile(join(mypath, f))]
    
    for fn in files:
        try: # extract features
            mfccs,chroma,mel,contrast,tonnetz = extract_feature(fn)
            features = np.empty((0,193))
            ext_features = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
            features = np.vstack([features,ext_features])
            
        except: # else exception (.ds_store files are part of mac file systems)
            print(fn)
            exception_count += 1
            continue
            
        l_row = raw_sound.loc[raw_sound['slice_file_name']==fn.split('/')[-1]].values.tolist()
        label = l_row[0][-1]
        fold = i+1
    
        mfcc_data.append([features, features.shape, label, fold])
        
        
print("Exceptions: ", exception_count)
end_time = timer()
print(print("time taken: {0} minutes {1:.1f} seconds".format((end_time - start_time)//60, (end_time - start_time)%60)))

cols=["features", "shape","label", "fold"]
mfcc_pd = pd.DataFrame(data = mfcc_data, columns=cols)

# Convert label to class number
le = LabelEncoder()
label_num = le.fit_transform(mfcc_pd["label"])

# one hot encode
ohe = OneHotEncoder()
onehot = ohe.fit_transform(label_num.reshape(-1, 1))

for i in range(10):
    mfcc_pd[le.classes_[i]] = onehot[:,i].toarray()
    
ll = [mfcc_pd['features'][i].ravel() for i in range(mfcc_pd.shape[0])]
mfcc_pd['sample'] = pd.Series(ll, index=mfcc_pd.index)
del mfcc_pd['features']

# save the network
pickle.dump(mfcc_pd, open('193_features.p','wb'))
