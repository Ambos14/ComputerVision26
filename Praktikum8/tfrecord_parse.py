import logging
import os

import numpy as np
import tensorflow as tf
from PIL import Image

logger = logging.getLogger(__name__)


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def save_tfrecord(iterator, tfrecord, max_num=None):
    with tf.io.TFRecordWriter(tfrecord) as writer:
        for i, (img, img_label) in enumerate(iterator):
            if max_num and i >= max_num:
                logger.info(f'max num {i + 1} of records reached')
                break
            img_raw = img.tobytes()
            img_label_raw = img_label.tobytes()
            height, width, channel = img.shape

            example = tf.train.Example(features=tf.train.Features(feature={
                'height': _int64_feature([height]),
                'width': _int64_feature([width]),
                'channel': _int64_feature([channel]),
                'image': _bytes_feature([img_raw]),
                'image_label': _bytes_feature([img_label_raw]),
            }))

            writer.write(example.SerializeToString())

            if i % 1000 == 0:
                logger.info(f'save_tfrecord: total records = {i + 1}')


def parser_tfrecord(record):
    features = tf.io.parse_single_example(
        record,
        features={
            'height': tf.io.FixedLenFeature([], tf.int64),
            'width': tf.io.FixedLenFeature([], tf.int64),
            'channel': tf.io.FixedLenFeature([], tf.int64),
            'image': tf.io.FixedLenFeature([], tf.string),
            'image_label': tf.io.FixedLenFeature([], tf.string),
        })

    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)
    channel = tf.cast(features['channel'], tf.int32)

    image = tf.io.decode_raw(features['image'], tf.uint8)
    image = tf.reshape(image, [height, width, channel])

    image_label = tf.io.decode_raw(features['image_label'], tf.uint8)
    image_label = tf.reshape(image_label, [height, width])
    return image, image_label


def visualize_tfrecord(tfrecord_path, target_dir, parser_fun_tfrecord, max_num_visualized=None):
    target_dir = os.path.join(target_dir, 'visualize')
    os.makedirs(target_dir, exist_ok=True)
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = dataset.map(parser_fun_tfrecord)
    for i, d in enumerate(dataset):
        if max_num_visualized and i >= max_num_visualized:
            break

        for j, e in enumerate(d):
            im = Image.fromarray(e.numpy())
            im.save(os.path.join(target_dir, f'd{i}_{j}.png'))

def split_tfrecord(tfrecord, output_dir, ratio=0.8, shuffle=True):
    # separate a tfrecord into train and eval tfrecords
    total_num = total_num_tfrecord(tfrecord)
    train_num = int(total_num * ratio)
    eval_num = total_num - train_num

    tfr = tf.data.TFRecordDataset(tfrecord)
    if shuffle:
        tfr = tfr.shuffle(max(50000, total_num))
    tfr_train = tfr.take(train_num)
    tfr_eval = tfr.skip(train_num)

    tfrecord_train = os.path.join(output_dir, 'data_train.tfrecord')
    tfrecord_eval = os.path.join(output_dir, 'data_eval.tfrecord')

    with tf.io.TFRecordWriter(tfrecord_train) as writer:
        itr = tfr_train.as_numpy_iterator()
        for i in itr:
            writer.write(i)

    with tf.io.TFRecordWriter(tfrecord_eval) as writer:
        itr = tfr_eval.as_numpy_iterator()
        for i in itr:
            writer.write(i)

    return tfrecord_train, tfrecord_eval


def total_num_tfrecord(tfrecord):
    total_num = sum(1 for _ in tf.data.TFRecordDataset(tfrecord))
    return total_num