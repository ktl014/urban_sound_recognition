#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""

@author: kevin

Overall code organization:
main()
- gets data from `get_dataloader_v2()` in `data/dataloader.py`
>> Data is loaded from the pickle file generated from the `extract 193 features`

- initializes tensorflow graph
>> within this graph comes the initialization of the following:
    placeholder variables (e.g. tf_data, tf_labels)
    the model (e.g. CNN, RNN, DNN)
    loss computation
    optimizer
    prediction

You can modify the model in `model/model_tf.py`. Review that script for more detail

"""
import os
import sys
import pickle
import random
import argparse
import logging
from timeit import default_timer as timer

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_curve, auc
import tensorflow as tf

from model.model_tf import CNN, DNN, RNN
from data.dataloader import get_dataloader_v2
from utils.eval_utils import test_accuracy, accuracy, init_feed_dict, dump_data

parser = argparse.ArgumentParser(description='PyTorch Training')

parser.add_argument('-e', '--epochs', action='store', default=20, type=int, help='epochs (default: 20)')
parser.add_argument('-i', '--iterations', action='store', default=20000, type=int, help='iterations (default: 20000)')
parser.add_argument('--batchSize', action='store', default=97, type=int, help='batch size (default: 1)')
parser.add_argument('--windowSize', action='store', default=25, type=int, help='number of frames (default: 25)')
parser.add_argument('--h_dim', action='store', default=256, type=int, help='LSTM hidden layer dimension (default: 256)')
parser.add_argument('--lr','--learning-rate',action='store',default=1e-4, type=float,help='learning rate (default: '
                                                                                          '0.01)')
parser.add_argument('--train_fold',action='store',default=1, type=int,help='Training Fold (default: 0)')
parser.add_argument('--test_fold',action='store',default=1, type=int,help='Testing Fold (default: 0)')

parser.add_argument('--train_f', action='store_false', default=True, help='Flag to train (STORE_FALSE)(default: True)')
parser.add_argument('--useGPU_f', action='store_false', default=True, help='Flag to use GPU (STORE_FALSE)(default: True)')
parser.add_argument('--gpu_num', action='store', default=0, type=int, help='gpu_num (default: 0)')
parser.add_argument("--net", default='AlexNet', const='AlexNet',nargs='?', choices=['VGG', 'AlexNet'], help="net model(default:VGG)")
parser.add_argument("--model", default='CNN', choices=['DNN', 'CNN', 'RNN'], help="model type(default:CNN)")

arg = parser.parse_args()

def main():
    if not os.path.exists('model'):
        os.makedirs('model')
    if not os.path.exists('log'):
        os.makedirs('log')
    model_path = 'model/model_{}'.format(arg.model)+str(arg.lr)+'_'+arg.net+'.pt'
    
    logger = logging.getLogger('netlog')
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(console)
    ch = logging.FileHandler('log/logfile_LSTM'+str(arg.lr)+'_'+arg.net+'.log')
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.info("================================================")
    logger.info('TensorFlow Version: {}'.format(tf.__version__))
    logger.info("Learning Rate: {}".format(arg.lr))
    logger.info("Nbr of Epochs: {}".format(arg.epochs))
    logger.info("Nbr of Iterations: {}".format(arg.iterations))
    logger.info("Batch Size: {}".format(arg.batchSize))
    logger.info("Window Size: {}".format(arg.windowSize))
    logger.info("Hidden Layer Dimension: {}".format(arg.h_dim))
    logger.info("GPU num: {}".format(arg.gpu_num))
    logger.info('Model Type: {}'.format(arg.model))

    # Constants
    num_of_classes=10
    feature_size = 193

    # Grab data
    trainLoader, testLoader, lblEncoder, data_cols = get_dataloader_v2(db_prepped=True)

    # Start graph
    graph = tf.Graph()
    with graph.as_default():
        tf_data = tf.placeholder(tf.float32, shape=[None, feature_size], name='data')
        tf_labels = tf.placeholder(tf.float32, shape=[None, num_of_classes], name='label')
        keep_prob = tf.placeholder(tf.float32, name='prob')

        # Select Model
        beta = 0.01
        if arg.model == 'CNN':
            model = CNN(feature_size=feature_size, num_of_classes=num_of_classes,
                        patch_size=5, num_channels=1,
                        depth1=32, num_hidden=1050)

            logits = model.forward(data=tf.expand_dims(tf.expand_dims(tf_data, [-1]), 1), proba=keep_prob)
        elif arg.model == 'DNN':
            model = DNN(feature_size=feature_size, num_of_classes=num_of_classes,
                        n_hidden1=400, n_hidden2=500, n_hidden3=400)

            logits = model.forward(tf_data, keep_prob)

        else:
            beta = 0.04
            model = RNN(feature_size=feature_size, batch_size=arg.batchSize, num_of_classes=num_of_classes,
                        n_hidden=200)
            logits = model.forward(tf.expand_dims(tf_data, -1), keep_prob)

        # Training computation
        model_params = [beta*tf.nn.l2_loss(i) for i in model.params]
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=tf_labels) +
                              sum(model_params))

        # Optimizer.
        optimizer = tf.train.AdamOptimizer(learning_rate=arg.lr).minimize(loss)

        # Predictions for the training, validation, and test data.
        prediction = tf.nn.softmax(logits)

        session_state = {
            'loss': loss,
            'optimizer': optimizer,
            'prediction': prediction,
        }
        print('Basic {} made'.format(arg.model))

    # Train and test model
    run_session(trainLoader, testLoader, lblEncoder, data_cols,
                graph=graph, session_state=session_state,
                num_itrs=arg.iterations, name=arg.model, k_prob=.2,
                batch_size=arg.batchSize)
    
def run_session(train, test, LB, data_cols, graph, session_state,
                num_itrs, name, batch_size,
                k_prob=1.0, mute=False, record=False):
    """Run tensorflow session

    The session is essentially the training loop, that runs the graph that was initialized in the main.

    Args:
        train (pd.DataFrame): Training set
        test (pd.DataFrame): Test set
        LB (sklearn.LabelBinarizer): One hot label encoded
        data_cols (pd.DataFrame.columns): Feature columns
        graph: Tensorflow Graph
        session_state (dict): Session state dictionary containing loss, optimizer, and prediction
        num_itrs (int): Number of epochs
        name (str): Model type
        batch_size (int): Batch size
        k_prob (float):
        mute:
        record:

    Returns:
        None

    """
    # Stats container
    """Use this to access the accuracy and predictions, given the model_name
    >> test_preds['RNN']
    0.983
    """
    acc_over_time = {}
    test_preds = {}
    
    start = timer()
    test_labels = LB.transform(test['label'])

    with tf.Session(graph=graph) as session:
        if record:
            merged = tf.merge_all_summaries()  
            writer = tf.train.SummaryWriter("/tmp/tensorflowlogs", session.graph)
        #tf.initialize_all_variables().run()
        tf.global_variables_initializer().run()

        print("Initialized")
        accu = []
        
        for iteration in range(num_itrs):
            
            # get batch
            train_batch = train.sample(batch_size)
            t_d = train_batch[data_cols]
            t_l = LB.transform(train_batch['label'])
            
            # make feed dict
            feed_dict = init_feed_dict(mode='train', data=t_d, label=t_l, prob=k_prob)
            
            # run model on batch
            _, l, predictions = session.run([session_state['optimizer'],
                                             session_state['loss'],
                                             session_state['prediction']], feed_dict=feed_dict)
            
            # mid model accuracy checks
            if (iteration % 1000 == 0) and not mute:
                print("\tMinibatch loss at iteration {}: {}".format(iteration, l))
                print("\tMinibatch accuracy: {:.1f}".format(accuracy(predictions, t_l)))
            if (iteration % 5000 == 0) and not mute:
                print("Test accuracy: {:.1f}".format(test_accuracy(session, test_data=test, test_labels=test_labels, data_cols=data_cols,
                                         prediction=session_state['prediction'], during=True)))
            if (iteration % 1000 == 0) and not mute:
                accu.append(tuple([iteration, test_accuracy(session, test_data=test, test_labels=test_labels, data_cols=data_cols,
                                         prediction=session_state['prediction'], during=True)]))
                
        # record accuracy and predictions
        test_preds[name] = test_accuracy(session, test_data=test, test_labels=test_labels, data_cols=data_cols,
                                         prediction=session_state['prediction'], during=False)
        print("Final Test accuracy: {:.1f}".format(accuracy(test_preds[name], test_labels)))
        end = timer()
        test_preds[name] = test_preds[name].ravel()
        acc_over_time[name] = accu
        print("time taken: {0} minutes {1:.1f} seconds".format((end - start)//60, (end - start)%60))

        # Save data
        dump_data(data=acc_over_time)
        dump_data(data=test_preds, fname='data/test_preds.p')
        dump_data(data=test_labels, fname='data/test_labels.p')




if __name__ == "__main__":
    main()