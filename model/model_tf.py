import tensorflow as tf

def weight_variable(shape):
    """Initialize weight variables"""
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)

def bias_variable(shape):
    """Initialize bias variables"""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

class CNN(object):
    def __init__(self, feature_size=193, num_of_classes=10, **kwargs):
        # Variables.
        self.feature_size = feature_size
        self.num_of_classes = num_of_classes
        patch_size = kwargs['patch_size'] #Default: 5
        depth1 = kwargs['depth1'] #Default: 32
        num_channels = kwargs['num_channels'] #Default: 1
        num_hidden = kwargs['num_hidden'] #Default: 1050

        self.layer1_weights = weight_variable([1, patch_size, 1, depth1]) # (1, 5, 1, 32)
        self.layer1_biases = bias_variable([depth1])
        self.layer2_weights = weight_variable([(self.feature_size // 2 + 1) * depth1, num_hidden]) # (3140, 1050)
        self.layer2_biases = bias_variable([num_hidden])
        self.layer3_weights = weight_variable([num_hidden, self.num_of_classes]) # (1050, 10)
        self.layer3_biases = bias_variable([self.num_of_classes])

        self.params = [
            self.layer1_weights,
            self.layer1_biases,
            self.layer2_weights,
            self.layer2_biases,
            self.layer3_weights,
            self.layer3_biases
        ]

    def forward(self, data, proba):
        """ Forward propagate input

        Args:
            data: dimension ~ (?, 1, 193, 1)
            proba: 0 to 1.0

        Returns:
            matrix of dimension (?, 10)

        """
        # Convolution
        conv1 = tf.nn.conv2d(data, filter=self.layer1_weights,
                             strides=[1, 1, 2, 1], padding='SAME') + self.layer1_biases
        pooled1 = tf.nn.max_pool(tf.nn.relu(conv1), ksize=[1, 1, 2, 1], strides=[1, 1, 1, 1], padding='SAME')

        # Fully Connected Layer
        shape = pooled1.get_shape().as_list() # [None, num_channels, feature_size, depth]
        reshape = tf.reshape(pooled1, [-1, shape[1] * shape[2] * shape[3]]) #(?, 3140)
        full2 = tf.nn.relu(tf.matmul(reshape, self.layer2_weights) + self.layer2_biases)

        # Dropout
        full2 = tf.nn.dropout(full2, proba) # (?, 1050)

        return tf.matmul(full2, self.layer3_weights) + self.layer3_biases # (?, 10)

class DNN(object):
    def __init__(self, feature_size=193, num_of_classes=10, **kwargs):
        n_hidden1 = kwargs['n_hidden1'] # 400
        n_hidden2 = kwargs['n_hidden2'] # 500
        n_hidden3 = kwargs['n_hidden3'] # 400

        # weights and biases
        self.layer1_weights = weight_variable([feature_size, n_hidden1])
        self.layer1_biases = bias_variable([n_hidden1])
        self.layer2_weights = weight_variable([n_hidden1, n_hidden2])
        self.layer2_biases = bias_variable([n_hidden2])
        self.layer3_weights = weight_variable([n_hidden2, n_hidden3])
        self.layer3_biases = bias_variable([n_hidden3])
        self.layer4_weights = weight_variable([n_hidden3, num_of_classes])
        self.layer4_biases = bias_variable([num_of_classes])

        self.params = [
            self.layer1_weights,
            self.layer1_biases,
            self.layer2_weights,
            self.layer2_biases,
            self.layer3_weights,
            self.layer3_biases,
            self.layer4_weights,
            self.layer4_biases,
        ]

    def forward(self, data, proba=1.0):
        layer1 = tf.nn.relu(tf.matmul(data, self.layer1_weights) + self.layer1_biases)
        layer1 = tf.nn.dropout(layer1, proba)

        layer2 = tf.nn.relu(tf.matmul(layer1, self.layer2_weights) + self.layer2_biases)
        layer2 = tf.nn.dropout(layer2, proba)

        layer3 = tf.nn.relu(tf.matmul(layer2, self.layer3_weights) + self.layer3_biases)
        layer3 = tf.nn.dropout(layer3, proba)

        return tf.matmul(layer3, self.layer4_weights) + self.layer4_biases

class RNN(object):
    def __init__(self, feature_size, batch_size, num_of_classes, **kwargs):
        self.feature_size = feature_size
        self.batch_size = batch_size
        self.n_hidden = kwargs['n_hidden'] # 200

        # weights and biases
        self.layer1_weights = weight_variable([feature_size * self.n_hidden, num_of_classes])
        self.layer1_biases = bias_variable([num_of_classes])

        self.params = [
            self.layer1_weights,
            self.layer1_biases
        ]

    def forward(self, data, proba=1.0):
        # Init RNN cell # tensorflow seems to take care of not reinitializing each time model is called
        cell = tf.nn.rnn_cell.BasicLSTMCell(self.n_hidden, state_is_tuple=True)
        # run rnn cell
        layer1, _istate = tf.nn.dynamic_rnn(cell, data, dtype=tf.float32)
        # reshape for output layer.
        reshape = tf.reshape(layer1, shape=[self.batch_size, self.feature_size*self.n_hidden], name='test')
        layer2 = tf.nn.dropout(reshape, proba)
        return tf.matmul(layer2, self.layer1_weights) + self.layer1_biases
