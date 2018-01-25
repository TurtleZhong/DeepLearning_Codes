"""
This project is a simple demo for LeNet
(http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)
used for Traffic Sign (http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).
* Author: ZhongXinliang
* Email: xinliangzhong@deepmotion.ai
* Date: 2018.01.24
"""
import pickle
import os
import random
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.contrib.layers import flatten
from sklearn.utils import shuffle

data_path = '../datasets/traffic-signs-data'


def load_data(data_path):
    training_file = os.path.join(data_path, 'train.p')
    testing_file = os.path.join(data_path, 'test.p')

    with open(training_file, mode='rb') as f:
        train = pickle.load(f)
    with open(testing_file, mode='rb') as f:
        test = pickle.load(f)

    x_train, y_train = train['features'], train['labels']
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, shuffle=True)
    x_test, y_test = test['features'], test['labels']
    return x_train, y_train, x_valid, y_valid, x_test, y_test


def random_show_image(x_train):
    image_index = random.randint(0, len(x_train))
    image = x_train[image_index]
    cv2.imshow('image', image)
    cv2.waitKey(0)


def plot_image(image, nr, nc, i, label=""):
    """
    Plot a single image.
    If 'i' is greater than 0, then plot this image as
    a subplot of a larger plot.
    """

    if i > 0:
        plt.subplot(nr, nc, i)
    else:
        plt.figure(figsize=(nr, nc))

    plt.xticks(())
    plt.yticks(())
    plt.xlabel(label)
    plt.tight_layout()
    plt.imshow(image, cmap="gray")


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


def get_image_per_class(X, y):
    """
    Plot a representatative of each image class in a 5x10 image grid

    The training dataset is traversed until a sample of each class
    is encountered and cached.

    Another loop then travereses all of the cached images and displays them.
    The two loops are required because we want to display the image samples
    in class order, not in the order they are encountered.
    """
    signs_left = len(np.bincount(y_train))
    class_images = [np.zeros(shape=(32,32)) for x in range(signs_left)]
    class_images = np.array(class_images)

    i = 0
    while signs_left > 0:
        if not class_images.all():
            image = X[i].squeeze()
            class_images[y[i]] = rgb2gray(image)
            signs_left -= 1
        i += 1
    return class_images


def summarize_stats(class_images, y_train, y_valid):
    """
    'class_images' is a list of images, one per class.
    This function plots this images list, and print underneath each one its class,
    the number of training samples, the percent of training samples,
    and the percent of validation samples
    """
    # Create a histogram of the classes
    y_train_hist = np.bincount(y_train)
    y_valid_hist = np.bincount(y_valid)

    nr = 5
    nc = 9
    plt.figure(figsize=(nr, nc))
    for image, i in zip(class_images, range(len(class_images))):
        label = (str(i) + "\n"                                            # class
              + str(y_train_hist[i]) + "\n"                               # no. of training samples
              + "{:.1f}%".format(100 * y_train_hist[i]/sum(y_train_hist)) + "\n"   # representation in training samples
              + "{:.1f}%".format(100 * y_valid_hist[i]/sum(y_valid_hist)))     # representation in validation samples
        plot_image(image, nr, nc, i+1, label)
    plt.show()
# Load the data.
x_train, y_train, x_valid, y_valid, x_test, y_test = load_data(data_path)

# Number of training examples
n_train = len(x_train)

# Number of validation examples.
n_validation = len(x_valid)

# Number of testing examples.
n_test = len(x_test)

# The shape of an traffic sign image
image_shape = x_train[0].shape

# How many unique classes/labels there are in the dataset.
n_classes = len(np.bincount(y_train))

print("Number of training examples =", n_train)
print("Number of validation examples =", n_validation)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)


"""
Display a list of training images, one from each class
"""
class_images = get_image_per_class(x_train, y_train)
summarize_stats(class_images, y_train, y_valid)

"""
Pre-process the Data Set (normalization, gray_scale)
"""


# Min-Max scaling for grayscale image data
# http://sebastianraschka.com/Articles/2014_about_feature_scaling.html#about-min-max-scaling
def normalize_scale(X):
    a = 0
    b = 1.0
    return a + X * (b - a) / 255


# http://lamda.nju.edu.cn/weixs/project/CNNTricks/CNNTricks.html
def standardize(X):
    X -= np.mean(X)  # zero-center
    X /= np.std(X)  # normalize
    return (X)


n_channels = 3  # assume default netowrk input is RGB


def rgb2gray(X):
    # gray = np.dot(X, [:,0.299, 0.587, 0.114])
    gray = np.dot(X, [0.299, 0.587, 0.114])
    gray = gray.reshape(len(X), 32, 32, 1)
    return gray


# preprocessing pipeline
def preprocess_dataset(X):
    X = rgb2gray(X)
    X = normalize_scale(X)
    X = standardize(X)
    return X

x_train = preprocess_dataset(x_train)
x_valid = preprocess_dataset(x_valid)
x_test = preprocess_dataset(x_test)

"""
The main architecture of the LeNet.
"""
# Shuffle the training data.
x_train, y_train = shuffle(x_train, y_train)

EPOCHS = 100
BATCH_SIZE = 128


def le_net(x):
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1

    # Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    conv1_w = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 6), mean=mu, stddev=sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1 = tf.nn.conv2d(x, conv1_w, strides=[1, 1, 1, 1], padding='VALID')
    conv1 = tf.nn.bias_add(conv1, conv1_b)
    # Activation.
    conv1 = tf.nn.relu(conv1)
    # Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    # Layer 2: Convolutional. Output = 10x10x16.
    conv2_w = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean=mu, stddev=sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2 = tf.nn.conv2d(conv1, conv2_w, strides=[1, 1, 1, 1], padding='VALID')
    conv2 = tf.nn.bias_add(conv2, conv2_b)
    # Activation.
    conv2 = tf.nn.relu(conv2)
    # Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    # Flatten. Input = 5x5x16. Output = 400.
    fc0 = flatten(conv2)
    # Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1_w = tf.Variable(tf.truncated_normal(shape=(400, 120), mean=mu, stddev=sigma))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1 = tf.add(tf.matmul(fc0, fc1_w), fc1_b)
    # Activation.
    fc1 = tf.nn.relu(fc1)
    # Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_w = tf.Variable(tf.truncated_normal(shape=(120, 84), mean=mu, stddev=sigma))
    fc2_b = tf.Variable(tf.zeros(84))
    fc2 = tf.add(tf.matmul(fc1, fc2_w), fc2_b)
    # Activation.
    fc2 = tf.nn.relu(fc2)
    # Layer 5: Fully Connected. Input = 84. Output = 10.
    fc3_w = tf.Variable(tf.truncated_normal(shape=(84, 43), mean=mu, stddev=sigma))
    fc3_b = tf.Variable(tf.zeros(43))
    logits = tf.add(tf.matmul(fc2, fc3_w), fc3_b)
    return logits

x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 43)

# Training pipeline
rate = 0.0005

logits = le_net(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=rate)
training_operation = optimizer.minimize(loss_operation)

# Model evaluation
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()


def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

# Train the model
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     num_examples = len(x_train)
#
#     print("Training...")
#     print()
#     for i in range(EPOCHS):
#         x_train, y_train = shuffle(x_train, y_train)
#         for offset in range(0, num_examples, BATCH_SIZE):
#             end = offset + BATCH_SIZE
#             batch_x, batch_y = x_train[offset:end], y_train[offset:end]
#             sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
#
#         validation_accuracy = evaluate(x_valid, y_valid)
#         print("EPOCH {} ...".format(i + 1))
#         print("Validation Accuracy = {:.3f}".format(validation_accuracy))
#         print()
#
#     saver.save(sess, '../TrafficSignClassfier/traffic_sign_classfier.ckpt')
#     print("Model saved")

with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))

    test_accuracy = evaluate(x_test, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))