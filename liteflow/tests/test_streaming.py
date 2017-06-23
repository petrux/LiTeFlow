"""Test module for the liteflow.streaming module."""

import numpy as np
import tensorflow as tf

from liteflow import streaming


class TestStreamingAverage(tf.test.TestCase):
    """Test case for the liteflow.streaming.StreamingAverage class."""

    def test_default(self):
        """Default test for the StreamingAvrage class."""

        # Set up the data.
        values_01 = np.asarray([[1, 2, 3], [4, 5, 1000]], dtype=np.float32)  # pylint: disable=I0011,E1101
        weights_01 = np.asarray([[1, 1, 1], [1, 1, 0]], dtype=np.float32)  # pylint: disable=I0011,E1101
        avg_01 = 3.0
        values_02 = np.asarray([[1000, 1000, 8], [9, 10, 1000]], dtype=np.float32)  # pylint: disable=I0011,E1101
        weights_02 = np.asarray([[0, 0, 1], [1, 1, 0]], dtype=np.float32)  # pylint: disable=I0011,E1101
        avg_02 = 9.0
        avg = 5.25

        # Build the graph.
        values = tf.placeholder(dtype=tf.float32, shape=[2, 3], name='values')
        weights = tf.placeholder(dtype=tf.float32, shape=[2, 3], name='weights')
        streaming_avg = streaming.StreamingAverage(name='StreamingAvg')
        streaming_avg.compute(values, weights)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            # First batch.
            feed_dict = {
                values: values_01,
                weights: weights_01
            }
            sess.run(streaming_avg.update_op, feed_dict)
            self.assertEqual(avg_01, sess.run(streaming_avg.batch_value, feed_dict))
            self.assertEqual(avg_01, sess.run(streaming_avg.value, feed_dict))

            # Second batch.
            feed_dict = {
                values: values_02,
                weights: weights_02,
            }
            sess.run(streaming_avg.update_op, feed_dict)
            self.assertEqual(avg_02, sess.run(streaming_avg.batch_value, feed_dict))
            self.assertEqual(avg, sess.run(streaming_avg.value, feed_dict))

            # Reset.
            sess.run(streaming_avg.reset_op)
            self.assertEqual(0.0, sess.run(streaming_avg.value))
            self.assertEqual(0, sess.run(streaming_avg.count))

            # Second batch as first.
            feed_dict = {
                values: values_02,
                weights: weights_02,
            }
            sess.run(streaming_avg.update_op, feed_dict)
            self.assertEqual(avg_02, sess.run(streaming_avg.batch_value, feed_dict))
            self.assertEqual(avg_02, sess.run(streaming_avg.value, feed_dict))


if __name__ == '__main__':
    tf.test.main()
