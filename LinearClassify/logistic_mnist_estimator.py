import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_Data/data/", one_hot=True)

import logging


class LogisticModelMnist(object):

    def __init__(self):
        self._logger = logging.getLogger(__name__)

    def init_paras(self):
        pass

    def train(self):
        pass


def main():
    pass


if __name__ == '__main__':
    main()
