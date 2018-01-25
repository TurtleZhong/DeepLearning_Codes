"""
This file is used for prediction.
* Author: ZhongXinliang
* Email: xinliangzhong@deepmotion.ai
* Date: 2018.01.24
"""
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import random
from LeNet.lenet import LeNet

saver = tf.train.Saver()

mnist = input_data.read_data_sets("../MNIST_data/", reshape=False)
X_test = mnist.test.images


X_test = np.pad(X_test, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')


def predict_one_image(input):
    assert (input.shape() == (32, 32, 1))


with tf.Session() as sess:
    saver.restore(sess, 'lenet.ckpt')
    print(saver)
    image_index = random.randint(0, len(X_test))
    image = X_test[image_index]

    print('Image.shape:', image.shape)
    cv2.imshow('test_image', image)
    cv2.waitKey(0)
    image = np.reshape(image, newshape=(1, 32, 32, 1))
    print('logits = ', LeNet(image))
    num = tf.argmax(LeNet(image), 1)
    print(num)
