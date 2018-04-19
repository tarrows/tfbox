import tensorflow as tf
import pickle


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


def conv_net(x_tensor, keep_probability):
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

    cnn = conv2d_maxpool(x_tensor, 10, conv_ksize, conv_strides, pool_ksize, pool_strides)
    cnn = conv2d_maxpool(cnn, 25, conv_ksize, conv_strides, pool_ksize, pool_strides)
    cnn = conv2d_maxpool(cnn, 40, conv_ksize, conv_strides, pool_ksize, pool_strides)
    cnn = tf.contrib.layers.flatten(cnn)

    cnn = tf.contrib.layers.fully_connected(inputs=cnn, num_outputs=64, activation_fn=tf.nn.relu)
    cnn = tf.nn.dropout(cnn, keep_probability)
    cnn = tf.contrib.layers.fully_connected(inputs=cnn, num_outputs=32, activation_fn=tf.nn.relu)
    cnn = tf.nn.dropout(cnn, keep_probability)

    output = tf.contrib.layers.fully_connected(inputs=cnn, num_outputs=10)
    return output


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


def print_stats(session, feature_batch, label_batch, cost, accuracy):
    """
    Print information about loss and validation accuracy
    : session: Current TensorFlow session
    : feature_batch: Batch of Numpy image data
    : label_batch: Batch of Numpy label data
    : cost: TensorFlow cost function
    : accuracy: TensorFlow accuracy function
    """
    feeds = {x: feature_batch, y: label_batch, keep_prob: 1.0}
    accuracy = session.run(accuracy, feed_dict=feeds)
    loss = session.run(cost, feed_dict=feeds)
    print('accuracy: {:>2}, loss: {} '.format(accuracy, loss))


def batch_features_labels(features, labels, batch_size):
    """
    Split features and labels into batches
    """
    for start in range(0, len(features), batch_size):
        end = min(start + batch_size, len(features))
        yield features[start:end], labels[start:end]


def load_preprocess_training_batch(batch_id, batch_size):
    """
    Load the Preprocessed Training data and return them in batches of <batch_size> or less
    """
    filename = 'preprocess_batch_' + str(batch_id) + '.p'
    features, labels = pickle.load(open(filename, mode='rb'))

    # Return the training data in batches of size <batch_size> or less
    return batch_features_labels(features, labels, batch_size)


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
cost_ = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
opt = tf.train.AdamOptimizer().minimize(cost_)

# Accuracy
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accr = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

# Hyperparameters
epochs = 20
batchsize = 128
keep_prob = 0.5


print('Checking the Training on a Single Batch...')
with tf.Session() as sess:
    # Initializing the variables
    sess.run(tf.global_variables_initializer())

    # Training cycle
    for epoch in range(epochs):
        batch_i = 1
        for batch_features, batch_labels in load_preprocess_training_batch(batch_i, batchsize):
            train_neural_network(sess, opt, keep_prob, batch_features, batch_labels)
            print('Epoch {:>2}, Batch {}:  '.format(epoch + 1, batch_i), end='')
            print_stats(sess, batch_features, batch_labels, cost_, accr)
