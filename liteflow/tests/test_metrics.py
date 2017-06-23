"""Test module for the liteflow.metrics module."""

import mock

import numpy as np
import tensorflow as tf

from liteflow import metrics
from liteflow import streaming


class TestStreamingMetric(tf.test.TestCase):
    """Base test class for the liteflow.metrics.StreamingMetric class."""

    def test_default(self):
        """Default test case."""
        scope = 'MyScope'
        targets = tf.constant([[0, 1, 2], [0, 9, 23]], dtype=tf.int32)
        predictions = tf.constant([[0, 1, 2], [0, 9, 23]], dtype=tf.int32)
        weights = tf.constant([[1, 1, 1], [0, 0, 1]], dtype=tf.float32)
        values = tf.constant([5, 6, 7], dtype=tf.float32)
        weights_out = tf.constant([1, 0, 1], dtype=tf.float32)

        func = mock.Mock()
        func.side_effect = [(values, weights_out)]

        avg = streaming.StreamingAverage()
        avg.compute = mock.MagicMock()

        metric = metrics.StreamingMetric(func, avg)
        metric.compute(targets, predictions, weights, scope=scope)

        func.assert_called_once_with(targets, predictions, weights)
        avg.compute.assert_called_once()
        args, kwargs = avg.compute.call_args
        act_values, act_weights_out = args
        self.assertEqual(act_values, values)
        self.assertEqual(act_weights_out, weights_out)
        self.assertIn('scope', kwargs)
        self.assertEqual(kwargs.pop('scope').name, scope)


    def test_weights_in_none(self):
        """Test case with no weights passed to the wrapped function."""
        scope = 'MyScope'
        targets = tf.constant([[0, 1, 2], [0, 9, 23]], dtype=tf.int32)
        predictions = tf.constant([[0, 1, 2], [0, 9, 23]], dtype=tf.int32)
        values = tf.constant([5, 6, 7], dtype=tf.float32)
        weights_out = tf.constant([1, 0, 1], dtype=tf.float32)

        func = mock.Mock()
        func.side_effect = [(values, weights_out)]

        avg = streaming.StreamingAverage()
        avg.compute = mock.MagicMock()

        metric = metrics.StreamingMetric(func, avg)
        metric.compute(targets, predictions, scope=scope)

        func.assert_called_once_with(targets, predictions, None)
        avg.compute.assert_called_once()
        args, kwargs = avg.compute.call_args
        act_values, act_weights_out = args
        self.assertEqual(act_values, values)
        self.assertEqual(act_weights_out, weights_out)
        self.assertIn('scope', kwargs)
        self.assertEqual(kwargs.pop('scope').name, scope)


    def test_weights_out_none(self):
        """Test case with no weights returned by the wrapped function."""
        scope = 'MyScope'
        targets = tf.constant([[0, 1, 2], [0, 9, 23]], dtype=tf.int32)
        predictions = tf.constant([[0, 1, 2], [0, 9, 23]], dtype=tf.int32)
        weights = tf.constant([[1, 1, 1], [0, 0, 1]], dtype=tf.float32)
        values = tf.constant([5, 6, 7], dtype=tf.float32)

        func = mock.Mock()
        func.side_effect = [(values, None)]

        avg = streaming.StreamingAverage()
        avg.compute = mock.MagicMock()

        metric = metrics.StreamingMetric(func, avg)
        metric.compute(targets, predictions, weights, scope=scope)

        func.assert_called_once_with(targets, predictions, weights)
        avg.compute.assert_called_once()
        args, kwargs = avg.compute.call_args
        act_values, act_weights_out = args
        self.assertEqual(act_values, values)
        self.assertEqual(act_weights_out, None)
        self.assertIn('scope', kwargs)
        self.assertEqual(kwargs.pop('scope').name, scope)


    def test_weights_in_out_none(self):
        """Test case with no weights at all."""
        scope = 'MyScope'
        targets = tf.constant([[0, 1, 2], [0, 9, 23]], dtype=tf.int32)
        predictions = tf.constant([[0, 1, 2], [0, 9, 23]], dtype=tf.int32)
        values = tf.constant([5, 6, 7], dtype=tf.float32)

        func = mock.Mock()
        func.side_effect = [(values, None)]

        avg = streaming.StreamingAverage()
        avg.compute = mock.MagicMock()

        metric = metrics.StreamingMetric(func, avg)
        metric.compute(targets, predictions, scope=scope)

        func.assert_called_once_with(targets, predictions, None)
        avg.compute.assert_called_once()
        args, kwargs = avg.compute.call_args
        act_values, act_weights_out = args
        self.assertEqual(act_values, values)
        self.assertEqual(act_weights_out, None)
        self.assertIn('scope', kwargs)
        self.assertEqual(kwargs.pop('scope').name, scope)


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

    def test_no_weights(self):
        """Test case with no weights."""
        targets = tf.constant([[2, 1, 0, 0]], dtype=tf.int32)
        predictions = tf.constant(
            [[[0.1, 0.8, 0.1],  # predicted: 1, WRONG.
              [0.1, 0.8, 0.1],  # predicted: 1, CORRECT.
              [0.8, 0.1, 0.1],  # predicted: 0, CORRECT
              [0.1, 0.1, 0.8]]],  # predicted: 2, WRONG.
            dtype=tf.float32)
        accuracy_t, weights_t = metrics.accuracy(targets, predictions)

        exp_accuracy = np.asarray([[0, 1, 1, 0]], dtype=np.float32)  # pylint: disable=I0011,E1101

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            act_accuracy = sess.run(accuracy_t)

        self.assertIsNone(weights_t)
        self.assertAllEqual(exp_accuracy, act_accuracy)


class TestPerSentenceAccuracy(tf.test.TestCase):
    """Test class for the liteflow.metrics.per_sentence_accuracy function."""

    def test_default(self):
        """Default test case."""
        targets = tf.constant([[1, 2, 3, 4], [1, 2, 3, 4]], dtype=tf.int32)
        predictions = tf.constant([[1, 2, 3, 0], [1, 2, 3, 0]], dtype=tf.int32)
        weights = tf.constant([[1, 1, 1, 1], [1, 1, 1, 0]], dtype=tf.float32)

        accuracy_t, weights_t = metrics.per_sentence_accuracy(targets, predictions, weights)
        exp_accuracy = np.asarray([0, 1], dtype=np.float32)  # pylint: disable=I0011,E1101
        exp_weights = np.asarray([1, 1], dtype=np.float32)  # pylint: disable=I0011,E1101

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            act_accuracy, act_weights = sess.run([accuracy_t, weights_t])
        self.assertAllEqual(act_accuracy, exp_accuracy)
        self.assertAllEqual(act_weights, exp_weights)

    def test_no_weights(self):
        """Test case for weightless metric."""
        targets = tf.constant([[1, 2, 3, 4], [1, 2, 3, 0]], dtype=tf.int32)
        predictions = tf.constant([[1, 2, 3, 0], [1, 2, 3, 0]], dtype=tf.int32)

        accuracy_t, weights_t = metrics.per_sentence_accuracy(targets, predictions)
        exp_accuracy = np.asarray([0, 1], dtype=np.float32)  # pylint: disable=I0011,E1101
        exp_weights = np.asarray([1, 1], dtype=np.float32)  # pylint: disable=I0011,E1101

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            act_accuracy, act_weights = sess.run([accuracy_t, weights_t])
        self.assertAllEqual(act_accuracy, exp_accuracy)
        self.assertAllEqual(act_weights, exp_weights)


if __name__ == '__main__':
    tf.test.main()
