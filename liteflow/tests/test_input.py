"""Test module for the `dket.input` module."""

import datetime
import os
import unittest

import tensorflow as tf

from liteflow import input as linput


def _timestamp():
    frmt = "%Y-%m-%d--%H-%M-%S.%f"
    stamp = datetime.datetime.now().strftime(frmt)
    print 'STAMP: ' + stamp
    return stamp


def _encode(key, vector):
    example = tf.train.Example(
        features=tf.train.Features(
            feature={
                'key': tf.train.Feature(
                    int64_list=tf.train.Int64List(
                        value=[key])),
                'vector': tf.train.Feature(
                    int64_list=tf.train.Int64List(
                        value=vector))}))
    return example


def _decode(message):
    features = {
        'key': tf.FixedLenFeature([], tf.int64),
        'vector': tf.VarLenFeature(tf.int64)
    }
    parsed = tf.parse_single_example(
        serialized=message,
        features=features)
    key = parsed['key']
    vector = tf.sparse_tensor_to_dense(parsed['vector'])
    return key, vector


def _save_records(fpath, *records):
    with tf.python_io.TFRecordWriter(fpath) as fout:
        for record in records:
            fout.write(record.SerializeToString())


def _read(fpath, num_epochs=None, shuffle=True):
    queue = tf.train.string_input_producer(
        string_tensor=[fpath],
        num_epochs=num_epochs,
        shuffle=shuffle)
    reader = tf.TFRecordReader()
    _, value = reader.read(queue)
    key, vector = _decode(value)
    return key, vector


class ShuffleBatchTest(tf.test.TestCase):
    """."""

    TMP_DIR = '/tmp'
    RANDOM_SEED = 23

    # TODO(petrux): since the shuffling is actually changing from time
    # to time, you should try to write some quality criterion on the
    # output (or otherwise try to make the shuffling deterministic).
    # In the meantime, the test is skipped.
    @unittest.skip('Fix.')
    def test_base(self):
        """."""

        # NOTA BENE: all the test depends on the value
        # used for the random seed, so if you change it
        # you HAVE TO re run the generation and check
        # manually in order to update the expected results.
        # Bottom line: DON'T CHANGE THE RANDOM SEED.
        tf.reset_default_graph()
        tf.set_random_seed(self.RANDOM_SEED)

        filename = os.path.join(self.TMP_DIR, _timestamp() + '.rio')
        data = [
            (1, [1]),
            (2, [2, 2]),
            (3, [3, 3, 3]),
            (4, [4, 4, 4, 4]),
            (5, [5, 5, 5, 5, 5]),
            (6, [6, 6, 6, 6, 6, 6])]
        examples = [_encode(k, v) for k, v in data]
        _save_records(filename, *examples)
        tensors = _read(filename, num_epochs=4, shuffle=False)

        batch_size = 3
        batch = linput.shuffle_batch(tensors, batch_size, seed=self.RANDOM_SEED)

        actual_keys = []
        expected_keys = [3, 5, 2, 5, 1, 2, 6, 2, 1, 1, 1, 6, 3, 4, 6, 4, 3, 4, 5, 5, 4, 6, 2, 3]

        with tf.Session() as sess:
            sess.run(tf.local_variables_initializer())
            sess.run(tf.global_variables_initializer())
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            try:
                while True:
                    bkey, bvector = sess.run(batch)
                    bkey = bkey.tolist()
                    length = max(bkey)
                    self.assertEqual((batch_size, length), bvector.shape)
                    actual_keys = actual_keys + bkey

            except tf.errors.OutOfRangeError as ex:
                coord.request_stop(ex=ex)
            finally:
                coord.request_stop()
                coord.join(threads)

        self.assertEquals(actual_keys, expected_keys)
        os.remove(filename)

if __name__ == '__main__':
    tf.test.main()
