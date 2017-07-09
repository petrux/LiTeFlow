"""Test module for liteflow.losses module."""

import math
import mock

import tensorflow as tf

from liteflow import losses
from liteflow import streaming
from liteflow.losses import categorical_crossentropy as xentropy


class TestStreamingLoss(tf.test.TestCase):
    """Test case for the liteflow.metrics.StreamingMetric class."""

    def test_default(self):
        """Default test case."""

        scope = 'StreamingLossScope'
        targets = tf.constant([[0, 1, 2], [0, 9, 23]], dtype=tf.int32)
        predictions = tf.constant([[0, 1, 2], [0, 9, 23]], dtype=tf.int32)
        weights = tf.constant([[1, 1, 1], [0, 0, 1]], dtype=tf.float32)
        values = tf.constant([5, 6, 7], dtype=tf.float32)
        weights_out = tf.constant([1, 0, 1], dtype=tf.float32)

        func = mock.Mock()
        func.side_effect = [(values, weights_out)]

        avg = streaming.StreamingAverage()
        avg.compute = mock.MagicMock()

        loss = losses.StreamingLoss(func, avg)
        loss.compute(targets, predictions, weights, scope=scope)

        func.assert_called_once_with(targets, predictions, weights)
        avg.compute.assert_called_once()
        args, kwargs = avg.compute.call_args
        act_values, = args
        self.assertEqual(act_values, values)
        self.assertIn('weights', kwargs)
        self.assertEqual(kwargs.pop('weights'), weights_out)
        self.assertIn('scope', kwargs)
        self.assertEqual(kwargs.pop('scope').name, scope)

    def test_weights_in_none(self):
        """Test case with no weights passed to the wrapped function."""
        scope = 'StreamingLossScope'
        targets = tf.constant([[0, 1, 2], [0, 9, 23]], dtype=tf.int32)
        predictions = tf.constant([[0, 1, 2], [0, 9, 23]], dtype=tf.int32)
        values = tf.constant([5, 6, 7], dtype=tf.float32)
        weights_out = tf.constant([1, 0, 1], dtype=tf.float32)

        func = mock.Mock()
        func.side_effect = [(values, weights_out)]

        avg = streaming.StreamingAverage()
        avg.compute = mock.MagicMock()

        loss = losses.StreamingLoss(func, avg)
        loss.compute(targets, predictions, scope=scope)

        func.assert_called_once_with(targets, predictions, None)
        avg.compute.assert_called_once()
        args, kwargs = avg.compute.call_args
        act_values, = args
        self.assertEqual(act_values, values)
        self.assertIn('weights', kwargs)
        self.assertEqual(kwargs.pop('weights'), weights_out)
        self.assertIn('scope', kwargs)
        self.assertEqual(kwargs.pop('scope').name, scope)


    def test_weights_out_none(self):
        """Test case with no weights returned by the wrapped function."""
        scope = 'StreamingLossScope'
        targets = tf.constant([[0, 1, 2], [0, 9, 23]], dtype=tf.int32)
        predictions = tf.constant([[0, 1, 2], [0, 9, 23]], dtype=tf.int32)
        weights = tf.constant([[1, 1, 1], [0, 0, 1]], dtype=tf.float32)
        values = tf.constant([5, 6, 7], dtype=tf.float32)

        func = mock.Mock()
        func.side_effect = [(values, None)]

        avg = streaming.StreamingAverage()
        avg.compute = mock.MagicMock()

        loss = losses.StreamingLoss(func, avg)
        loss.compute(targets, predictions, weights, scope=scope)

        func.assert_called_once_with(targets, predictions, weights)
        avg.compute.assert_called_once()
        args, kwargs = avg.compute.call_args
        act_values, = args
        self.assertEqual(act_values, values)
        self.assertIn('weights', kwargs)
        self.assertEqual(kwargs.pop('weights'), None)
        self.assertIn('scope', kwargs)
        self.assertEqual(kwargs.pop('scope').name, scope)

    def test_weights_in_out_none(self):
        """Test case with no weights at all."""
        scope = 'StreamingLossScope'
        targets = tf.constant([[0, 1, 2], [0, 9, 23]], dtype=tf.int32)
        predictions = tf.constant([[0, 1, 2], [0, 9, 23]], dtype=tf.int32)
        values = tf.constant([5, 6, 7], dtype=tf.float32)

        func = mock.Mock()
        func.side_effect = [(values, None)]

        avg = streaming.StreamingAverage()
        avg.compute = mock.MagicMock()

        loss = losses.StreamingLoss(func, avg)
        loss.compute(targets, predictions, scope=scope)

        func.assert_called_once_with(targets, predictions, None)
        avg.compute.assert_called_once()
        args, kwargs = avg.compute.call_args
        act_values, = args
        self.assertEqual(act_values, values)
        self.assertIn('weights', kwargs)
        self.assertEqual(kwargs.pop('weights'), None)
        self.assertIn('scope', kwargs)
        self.assertEqual(kwargs.pop('scope').name, scope)


class TestCategoricalCrossentropy(tf.test.TestCase):
    """Test case for the liteflow.losses.categorical_crossentropy function."""

    def test_default(self):
        """Default test for liteflow.losses.categorical_crossentropy function."""
        targets = tf.constant([[0, 1, 2, 0]], dtype=tf.int32)
        predictions = tf.constant(
            [[[0.5, 0.3, 0.2],
              [0.5, 0.3, 0.2],
              [0.5, 0.3, 0.2],
              [0.9, 0.1, 0.0]]],
            dtype=tf.float32)
        weights = tf.constant([[1.0, 1.0, 1.0, 0.0]], dtype=tf.float32)
        loss_t, weights_out_t = xentropy(targets, predictions, weights)

        exp_loss = [[-math.log(0.5), -math.log(0.3), -math.log(0.2), 0.0]]

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            exp_weights_out = sess.run(weights)
            loss, weights_out = sess.run([loss_t, weights_out_t])

        self.assertAllClose(loss, exp_loss)
        self.assertAllEqual(weights_out, exp_weights_out)



    def test_no_weights(self):
        """Test for liteflow.losses.categorical_crossentropy function with no weights."""
        pass


if __name__ == '__main__':
    tf.test.main()