"""Test module for the liteflow.metrics module."""

import numpy as np
import tensorflow as tf

from liteflow import metrics


class TestAccuracy(tf.test.TestCase):
    """Test class for the liteflow.metrics.accuracy function."""

    def test_default(self):
        """Default test case."""
        targets = tf.constant([[2, 1, 0, 0]], dtype=tf.int32)
        weights = tf.placeholder(dtype=tf.float32, shape=targets.shape)
        predictions = tf.constant(
            [[[0.1, 0.8, 0.1],  # predicted: 1, WRONG.
              [0.1, 0.8, 0.1],  # predicted: 1, CORRECT.
              [0.8, 0.1, 0.1],  # predicted: 0, CORRECT
              [0.1, 0.1, 0.8]]],  # predicted: 2, WRONG.
            dtype=tf.float32)
        accuracy_t, weights_out_t = metrics.accuracy(targets, predictions, weights)

        act_weights = np.asarray([[1, 1, 0, 0]], dtype=np.float32)  # pylint: disable=I0011,E1101
        exp_accuracy = np.asarray([[0, 1, 0, 0]], dtype=np.float32)  # pylint: disable=I0011,E1101

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            act_accuracy, act_weights_out = sess.run(
                fetches=[accuracy_t, weights_out_t],
                feed_dict={weights: act_weights})

        self.assertEqual(weights, weights_out_t)
        self.assertAllEqual(act_weights, act_weights_out)
        self.assertAllEqual(exp_accuracy, act_accuracy)


if __name__ == '__main__':
    tf.test.main()
