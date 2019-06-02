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
