import numpy as np
import math
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf


import logging

class SampleGenerator(object):

    def __init__(self):
        self._logger = logging.getLogger(__name__)
        self._data = []

    def run(self):
        weight = self._init_logistic_weight()
        self._logger.warning("%s weight: %s", len(weight), weight)
        samples = self._generate_sample(weight)
        self._logger.warning("sample: %s", samples[0])
        return samples

    def _init_logistic_weight(self, weight=None, num_weight=10):
        if not weight:
            weight = []
            for i in range(num_weight):
                weight.append(random.uniform(-1.0, 1.0))
            return weight
        return weight

    # y = 1/(1+exp(-wx))
    def _generate_sample(self, weight, num_sample=200):
        num_feature = len(weight) - 1
        samples = []
        for i in range(num_sample):
            x = []
            for j in range(num_feature):
                x.append(random.randint(-10, 10))
            y = 1/(1+ math.exp(weight[num_feature] + sum([weight[i] * x[i] for i in range(num_feature)])))
            if y > 0.5:
                y = [1.0, 0.0]
            else:
                y = [0.0, 1.0]
            samples.append((x, y))
        return samples


class LogisticModel(object):

    def __init__(self):
        self._logger = logging.getLogger(__name__)

    def train(self, sample):
        X_train, X_test, y_train, y_test = self._set_sample(sample)
        self._train(X_train, y_train)
        return

    def _train(self, X_train, Y_train):
        num_sample = X_train.shape[0]
        self._logger.warn("num_sample: %s", num_sample)
        num_feature = X_train.shape[1]
        self._logger.warn("num_feature: %s", num_feature)

        num_class = Y_train.shape[1]
        self._logger.warn("num_class: %s", num_class)

        x = tf.placeholder(tf.float32, [None, num_feature])
        y = tf.placeholder(tf.float32, [None, num_class])

        W = tf.Variable(tf.zeros([num_feature, num_class]))
        b = tf.Variable(tf.zeros([num_class]))
        pred = tf.nn.softmax(tf.matmul(x, W) + b)
        # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(pred, y))
        cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), reduction_indices=1))

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
                    _, c = sess.run([optimizer, cost],
                                    feed_dict={x: X_train[i * batch_size: (i + 1) * batch_size],
                                               y: Y_train[i * batch_size: (i + 1) * batch_size, :]})

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
        num_class = len(sample_0[1])
        X = np.zeros((num_sample, num_feature), dtype=float)
        Y = np.zeros((num_sample, num_class), dtype=float)
        for i in range(num_sample):
            X[i, ] = np.asarray(sample[i][0])
            Y[i, ] = np.asarray(sample[i][1])
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

        return X_train, X_test, y_train, y_test


def main():
    sample_generator = SampleGenerator()
    sample = sample_generator.run()

    model = LogisticModel()
    model.train(sample)


if __name__ == '__main__':
    main()
