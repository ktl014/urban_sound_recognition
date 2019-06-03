import numpy as np
import pickle
import tensorflow as tf

def accuracy(predictions, labels):
    """Compute training accuracy"""
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])

def test_accuracy(session, test_data, test_labels, data_cols, prediction, during=True, check_size=97):
    """Compute the test accuracy given a test set"""
    test_data.reset_index(inplace=True, drop=True)
    feed_dict = init_feed_dict(mode='test', data=test_data.loc[0:check_size-1,data_cols], prob=1.0)
    epoch_pred = session.run(prediction, feed_dict=feed_dict)

    for i in range(check_size, test_data.shape[0], check_size):
        # Init test feed dict
        feed_dict = init_feed_dict(mode='test', data=test_data.loc[i:i+check_size-1,data_cols], prob=1.0)
        # Run session
        epoch_pred = np.concatenate([epoch_pred, session.run(prediction, feed_dict=feed_dict)], axis=0)
    if during:
        return accuracy(epoch_pred, test_labels)
    else:
        return epoch_pred

def init_feed_dict(mode, data=None, label=None, prob=None):
    """Initialize feed dictionary"""
    if mode == 'train':
        feed_dict = {get_placeholder('data'):data,
                     get_placeholder('label'):label,
                     get_placeholder('prob'):prob}
    else:
        feed_dict = {get_placeholder('data'):data,
                     get_placeholder('prob'):prob}
    return feed_dict

def get_placeholder(name):
    """Get placeholder variables given the name"""
    return tf.get_default_graph().get_tensor_by_name(name+':0')

def dump_data(data, fname='data/dataset_acc.p'):
    import os
    if os.path.exists(fname):
        assert isinstance(data, dict), 'Data needs to be dictionary for merging'
        print(f'Detected existing filename. Now merging.')
        existing_data = load_data(fname)
        data = {**data, **existing_data}

    pickle.dump(data, open(fname, 'wb'))
    print(f'Data saved as {fname}')

def load_data(fname='data/dataset_acc.p'):
    return pickle.load(open(fname, 'rb'))

