# Urban Sound Classification (Group 36)

## Team Members
- Kevin Le (@ktl014)


## Problem
Conducting sound recognition in urban settings

## Summary
Sound is being utilized more frequently to convey information, since vision can be obstructed in 
some cases, preventing the use of most computer vision tools. Thus, environmental sound 
classification is a motivating interest to pursue within the field of deep learning. In this 
research, we showcase sound recognition performance from a range of simple to complex models.

## Methodology

- Extract sound features from audio wave files: mel-frequency cepstral coefficients (MFCC), 
chromagram, mel-scaled spectrogram, etc.
- Model selections consisted of KNN, DNN, and CNN. Network topology ranges from 3 to 4 layers.
- Evaluation metrics used: AUC and Accuracy. Loss and confusion matrices provided in addition


## Dataset
UrbanSound8K dataset. To download the dataset, please direct yourself to 
https://urbansounddataset.weebly.com/urbansound8k.html and query for the 'DOWNLOAD' section.

## Applications
1. Context aware computing
2. Surveillance
3. Noise mitigation enabled by smart acoustic sensor networks

## File Structure

```
Root
|
+----dataset
|
+----processed_data
|
+----data
|       |   dataloader.py
|       |   d_utils.py
|       |   extract_features193.py
|
+----model
|       |   model_tf.py
|       |   model_vggcnn.py
|       |   model_vgglstm.py
|
+----utils
|       |   eval_utils.py
|
|    training_193.py
|    train_knn.py
|    training_lstm.py
|    prepare_db.py
|    plotting.ipynb
```

## Instructions on running the code

* Python version: Python 3.6.6 64-bit
### Required packages

1. librosa
1. numpy
2. matplotlib
3. torch
3. tensorflow
4. soundfile
5. pickle
6. scipy
7. cv2
8. python_speech_features
9. scikit-learn
10. pandas

For installing these packages, you can use either ```pip3``` to install packages. For example, 

```pip3 install numpy```

### Run the code
1. Run the ```extract_features193.py``` to generate the ```data/193_features.p``` 
2. Run the ```training_193.py``` and ```train_knn.py``` to train and evaluate the model.
    - After each training session for each type of model, it will output several pickle files, 
    such as ```dataset_acc.p```, ```test_preds.p```, ```test_labels.p``` under ```../data```. 
    This will be used to plot.
3. Plot the results by opening ```plotting.ipynb``` and run each code block to plot the loss, 
accuracy, etc.