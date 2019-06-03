""" """
import os
import random

from multiprocessing import Process, Pipe

import numpy as np
import pandas as pd

import librosa

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix

def mean_mfccs(x):
    """Take average of MFCCs"""
    return [np.mean(feature) for feature in librosa.feature.mfcc(x)]

def parse_audio(x):
    """Parse each audio file for transformation"""
    return x.flatten('F')[:x.shape[0]]

def get_samples(block_size=2):
    """Get training inputs: spectrogram and labels"""
    # Static for quick development
    train_path = "dataset/UrbanSound8K/audio/fold1"
    train_file_names = os.listdir(train_path)
    # train_file_names = random.sample(train_file_names, 500)

    samples = []
    labels = []

    bad_count = 0
    for i,file_name in enumerate(train_file_names):
        if i % 50 == 0:
            print('Loaded {}/{}'.format(i, len(train_file_names)))

        try:
            x, sr = librosa.load(os.path.join(train_path, file_name))
        except:
            bad_count += 1
            print("BAD AUDIO INPUT!! fn:{} {}".format(os.path.basename(
                file_name), bad_count))
            continue

        x = parse_audio(x)
        samples.append(mean_mfccs(x))
        labels.append(os.path.basename(file_name).split('-')[1])

    return np.array(samples), np.array(labels)

X, Y = get_samples()
x_train, x_test, y_train, y_test = train_test_split(X, Y)

print(f'Shape: {x_train.shape}')
print(f'Observation: \n{x_train[0]}')
print(f'Labels: {y_train[:5]}')

scaler = StandardScaler()
scaler.fit(x_train)
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)

pca = PCA().fit(x_train_scaled)

plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)')
plt.show()

grid_params = {
    'n_neighbors': [3, 5, 7, 9, 11, 15],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

model = GridSearchCV(KNeighborsClassifier(), grid_params, cv=5, n_jobs=-1)
model.fit(x_train_scaled, y_train)

print(f'Model Score: {model.score(x_test_scaled, y_test)}')

y_predict = model.predict(x_test_scaled)
cm =confusion_matrix(y_predict, y_test)
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print(f'Confusion Matrix: \n{cm}')
print(f'Class Accuracies')
for acc in np.argsort(cm.diagonal()):
    print('Class {}: {}'.format(acc, cm.diagonal()[acc]))
