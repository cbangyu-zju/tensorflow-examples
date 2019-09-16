import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_Data/data/", one_hot=True)

import logging


class LogisticModel(object):

    def __init__(self):
        self._logger = logging.getLogger(__name__)

    def train(self):
        self._train()
        return

    def _train(self):
        num_sample = mnist.train.num_examples
        self._logger.warn("num_sample: %s", num_sample)
        num_feature = 784 # X_train.shape[1]
        num_class = 10
        self._logger.warn("num_feature: %s", num_feature)

        x = tf.placeholder(tf.float32, [None, num_feature])
        y = tf.placeholder(tf.float32, [None, num_class])

        W = tf.Variable(tf.zeros([num_feature, num_class]))
        b = tf.Variable(tf.zeros([num_class]))
        pred = tf.nn.softmax(tf.matmul(x, W) + b)
        cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), axis=1))

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

        batch_size = 10
        display_step = 2

        init = tf.initialize_all_variables()

        with tf.Session() as sess:
            sess.run(init)
            for epoch in range(100):
                avg_cost = 0
                total_batch = int(num_sample / batch_size)
                for i in range(total_batch):
                    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                    _, c = sess.run([optimizer, cost],
                                    feed_dict={x: batch_xs,
                                               y: batch_ys})

                    avg_cost += c / total_batch
                if (epoch + 1) % display_step == 0:
                    self._logger.warning("Epoch: %s, cost= %s", epoch + 1, avg_cost)
            w = W.eval(sess)
            b_0 = b.eval(sess)
        self._logger.warning("w: %s, B: %s", w, b_0)

    def _set_sample(self, sample):
        sample_0 = sample[0]

        num_feature = len(sample_0[0])
        num_sample = len(sample)
        X = np.zeros((num_sample, num_feature), dtype=float)
        Y = np.zeros((num_sample, 2), dtype=float)
        for i in range(num_sample):
            X[i, ] = np.asarray(sample[i][0])
            Y[i, ] = np.asarray(sample[i][1])
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

        return X_train, X_test, y_train, y_test


def main():
    model = LogisticModel()
    model.train()


if __name__ == '__main__':
    main()
