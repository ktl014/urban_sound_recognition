import tensorflow as tf

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

class CNN(object):
    def __init__(self, batch_size, feature_size, num_labels=10, **kwargs):
        # Variables.
        self.feature_size = feature_size
        self.batch_size = batch_size
        self.num_labels = num_labels
        patch_size = kwargs['patch_size']
        depth1 = kwargs['depth1']
        num_channels = kwargs['num_channels']
        num_hidden = kwargs['num_hidden']

        self.layer1_weights = weight_variable([1, patch_size, 1, depth1])
        self.layer1_biases = bias_variable([depth1])
        self.layer2_weights = weight_variable([(self.feature_size // 2 + 1) * depth1, num_hidden])
        self.layer2_biases = bias_variable([num_hidden])
        self.layer3_weights = weight_variable([num_hidden, self.num_labels])
        self.layer3_biases = bias_variable([self.num_labels])

        self.params = [
            self.layer1_weights,
            self.layer1_biases,
            self.layer2_weights,
            self.layer2_biases,
            self.layer3_weights,
            self.layer3_biases
        ]

    def forward(self, data, proba):
        # Convolution
        conv1 = tf.nn.conv2d(data, self.layer1_weights, [1, 1, 2, 1], padding='SAME') + self.layer1_biases
        pooled1 = tf.nn.max_pool(tf.nn.relu(conv1), ksize=[1, 1, 2, 1], strides=[1, 1, 1, 1], padding='SAME')

        # Fully Connected Layer
        shape = pooled1.get_shape().as_list()
        reshape = tf.reshape(pooled1, [-1, shape[1] * shape[2] * shape[3]])
        full2 = tf.nn.relu(tf.matmul(reshape, self.layer2_weights) + self.layer2_biases)

        # Dropout
        full2 = tf.nn.dropout(full2, proba)

        return tf.matmul(full2, self.layer3_weights) + self.layer3_biases

