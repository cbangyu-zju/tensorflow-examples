import os
import tensorflow as tf
import numpy as np
from PIL import Image


def _int64_feature(value):
    if not isinstance(value,list):
        value=[value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _byte_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def image_to_tfrecord(data_path):
    pass


def read_label(filename):
    pass