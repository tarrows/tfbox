import tensorflow as tf


def conv2d_maxpool(x_tensor, conv_num_outputs, conv_ksize, conv_strides, pool_ksize, pool_strides):
    """
    Apply convolution then max pooling to x_tensor
    :param x_tensor: TensorFlow Tensor
    :param conv_num_outputs: Number of outputs for the convolutional layer
    :param conv_ksize: kernal size 2-D Tuple for the convolutional layer
    :param conv_strides: Stride 2-D Tuple for convolution
    :param pool_ksize: kernal size 2-D Tuple for pool
    :param pool_strides: Stride 2-D Tuple for pool
    : return: A tensor that represents convolution and max pooling of x_tensor
    """

    weights_shape = list(conv_ksize) + [x_tensor.get_shape().as_list()[-1], conv_num_outputs]
    weights = tf.Variable(tf.truncated_normal(shape=weights_shape, mean=0.0, stddev=0.1))
    bias = tf.Variable(tf.zeros(conv_num_outputs))
    strides_attr = [1] + list(conv_strides) + [1]

    conv = tf.nn.conv2d(x_tensor, weights, strides=strides_attr, padding='SAME')
    conv = tf.nn.bias_add(conv, bias)
    conv = tf.nn.relu(conv)

    pool_ksize_attr = [1] + list(pool_ksize) + [1]
    pool_strides_attr = [1] + list(pool_strides) + [1]

    conv = tf.nn.max_pool(conv, ksize=pool_ksize_attr, strides=pool_strides_attr, padding='SAME')
    return conv


def conv_net(x, keep_prob):
    """
    Create a convolutional neural network model
    : x: Placeholder tensor that holds image data.
    : keep_prob: Placeholder tensor that hold dropout keep probability.
    : return: Tensor that represents logits
    """

    conv_ksize = (5, 5)
    conv_strides = (1, 1)
    pool_ksize = (2, 2)
    pool_strides = (2, 2)

    cnn = conv2d_maxpool(x, 10, conv_ksize, conv_strides, pool_ksize, pool_strides)
    cnn = conv2d_maxpool(cnn, 25, conv_ksize, conv_strides, pool_ksize, pool_strides)
    cnn = conv2d_maxpool(cnn, 40, conv_ksize, conv_strides, pool_ksize, pool_strides)
    cnn = tf.contrib.layers.flatten(cnn)

    cnn = tf.contrib.layers.fully_connected(inputs=cnn, num_outputs=64, activation_fn=tf.nn.relu)
    cnn = tf.nn.dropout(cnn, keep_prob)
    cnn = tf.contrib.layers.fully_connected(inputs=cnn, num_outputs=32, activation_fn=tf.nn.relu)
    cnn = tf.nn.dropout(cnn, keep_prob)

    output = tf.contrib.layers.fully_connected(inputs=cnn, num_outputs=10)
    return output


# Remove previous weights, bias, inputs, etc..
tf.reset_default_graph()

# Inputs image: (32, 32, 3) label: n_classes = 10
input_image_shape = [None, 32, 32, 3]
input_label_shape = [None, 10]

x = tf.placeholder(tf.float32, shape=input_image_shape, name="x")
y = tf.placeholder(tf.float32, shape=input_label_shape, name="y")
keep_prob = tf.placeholder(tf.float32, shape=None, name="keep_prob")

# Model
logits = conv_net(x, keep_prob)

# Name logits Tensor, so that is can be loaded from disk after training
logits = tf.identity(logits, name='logits')

# Loss and Optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
optimizer = tf.train.AdamOptimizer().minimize(cost)


def train_neural_network(session, optimizer, keep_probability, feature_batch, label_batch):
    """
    Optimize the session on a batch of images and labels
    : session: Current TensorFlow session
    : optimizer: TensorFlow optimizer function
    : keep_probability: keep probability
    : feature_batch: Batch of Numpy image data
    : label_batch: Batch of Numpy label data
    """
    feeds = {x: feature_batch,  y: label_batch, keep_prob: keep_probability}
    session.run(optimizer, feed_dict=feeds)
