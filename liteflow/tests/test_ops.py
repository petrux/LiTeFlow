"""Test module for `liteflow.ops` module."""

import numpy as np
import tensorflow as tf

from liteflow import ops

class FitTrimPadTest(tf.test.TestCase):
    """Test case for the fit(), trim() and pad() functions."""

    _SEED = 23

    def setUp(self):
        tf.reset_default_graph()
        tf.set_random_seed(self._SEED)
        np.random.seed(seed=self._SEED)   # pylint: disable=I0011,E1101

    def test_fit_same_width(self):
        """Fit a tensor to a dimension which is the actual one.

        Given a 3D tensor of shape [batch, length, width], apply the
        `ops.fit()` operator to it with the same `width` as the target one
        and check that the output tensor is the same of the input one.
        """
        batch = 2
        length = 5
        width = 4

        shape = [None, None, None]
        input_ = tf.placeholder(dtype=tf.float32, shape=shape)
        output = ops.fit(input_, width)

        input_actual = np.random.rand(batch, length, width)  # pylint: disable=I0011,E1101
        output_expected = input_actual
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            output_actual = sess.run(output, {input_: input_actual})
        self.assertAllClose(output_expected, output_actual)

    def test_fit_to_less_width(self):
        """Fit a tensor to a smalles width (i.e. trimming).

        Given a 3D tensor of shape [batch, length, width], apply the
        `ops.fit()` operator to it with the a smaller `width` as the
        target one and check that the last axis of the tensor have been
        deleted.
        """
        batch = 2
        length = 5
        width = 4
        fit_width = 3
        delta = width - fit_width

        shape = [None, None, None]
        input_ = tf.placeholder(dtype=tf.float32, shape=shape)
        output = ops.fit(input_, fit_width)

        input_actual = np.random.rand(batch, length, width)  # pylint: disable=I0011,E1101
        delete_idx = [width - (i + 1) for i in range(delta)]
        output_expected = np.delete(input_actual, delete_idx, axis=2)  # pylint: disable=I0011,E1101
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            output_actual = sess.run(output, {input_: input_actual})
        self.assertAllClose(output_expected, output_actual)

    def test_fit_to_greater_width(self):
        """Fit a tensor to a larger width (i.e. padding).

        Given a 3D tensor of shape [batch, length, width], apply the
        `ops.fit()` operator to it with the a larger `width` as the
        target one and check that the tensor has now padded values
        (with 0 as the padding value).
        """
        batch = 2
        length = 5
        width = 4
        fit_width = 5

        shape = [None, None, None]
        input_ = tf.placeholder(dtype=tf.float32, shape=shape)
        output = ops.fit(input_, fit_width)

        input_actual = np.random.rand(batch, length, width)  # pylint: disable=I0011,E1101
        output_expected = np.pad(  # pylint: disable=I0011,E1101
            input_actual,
            ((0, 0), (0, 0), (0, fit_width - width)),
            mode='constant')
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            output_actual = sess.run(output, {input_: input_actual})
        self.assertAllClose(output_expected, output_actual)

    def test_trim_to_greatereq_width(self):
        """Trim a tensor to a greater or equal width.

        Given a 3D tensor of shape [batch, length, width], apply the
        `ops.trim()` operator to it with target `width` that is greater or
        equal than the original one, and check that the output tensor is the
        same of the input one.
        """
        batch = 2
        length = 5
        width = 4
        width_larger = 5

        shape = [None, None, None]
        input_ = tf.placeholder(dtype=tf.float32, shape=shape)
        output_same = ops.trim(input_, width)
        output_larger = ops.trim(input_, width_larger)

        input_actual = np.random.rand(batch, length, width)  # pylint: disable=I0011,E1101
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            output_same_actual, output_larger_actual = sess.run(
                [output_same, output_larger], {input_: input_actual})
        self.assertAllClose(input_actual, output_same_actual)
        self.assertAllClose(input_actual, output_larger_actual)

    def test_trim_to_less_width(self):
        """Trim a tensor to a smaller width.

        Given a 3D tensor of shape [batch, length, width], apply the
        `ops.trim()` operator to it with the a smaller `width` as the
        target one and check that the last axis of the tensor have been
        deleted.
        """
        batch = 2
        length = 5
        width = 4
        width_smaller = 2
        delta = width - width_smaller

        shape = [None, None, None]
        input_ = tf.placeholder(dtype=tf.float32, shape=shape)
        output = ops.trim(input_, width_smaller)

        input_actual = np.random.rand(batch, length, width)  # pylint: disable=I0011,E1101
        delete_idx = [width - (i + 1) for i in range(delta)]
        output_expected = np.delete(input_actual, delete_idx, axis=2)  # pylint: disable=I0011,E1101

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            output_actual = sess.run(output, {input_: input_actual})
        self.assertAllClose(output_expected, output_actual)

    def test_pad_to_lesseq_width(self):
        """Pad a tensor to a smaller (or equal) one.

        Given a 3D tensor of shape [batch, length, width], apply the
        `ops.pad()` operator to it with target `width` that is less or
        equal than the original one, and check that the output tensor is
        the same of the input one.
        """
        batch = 2
        length = 5
        width = 4
        width_smaller = 3

        shape = [None, None, None]
        input_ = tf.placeholder(dtype=tf.float32, shape=shape)
        output_same = ops.pad(input_, width)
        output_smaller = ops.pad(input_, width_smaller)

        input_actual = np.random.rand(batch, length, width)  # pylint: disable=I0011,E1101
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            output_same_actual, output_smaller_actual = sess.run(
                [output_same, output_smaller], {input_: input_actual})
        self.assertAllClose(input_actual, output_same_actual)
        self.assertAllClose(input_actual, output_smaller_actual)


    def test_pad_to_greater_width(self):
        """Pad a tensor to a larger one.

        Given a 3D tensor of shape [batch, length, width], apply the
        `ops.pad()` operator to it with the a larger `width` as the
        target one and check that the tensor has now padded values
        (with 0 as the padding value).
        """
        batch = 2
        length = 5
        width = 4
        fit_width = 5

        shape = [None, None, None]
        input_ = tf.placeholder(dtype=tf.float32, shape=shape)
        output = ops.pad(input_, fit_width)

        input_actual = np.random.rand(batch, length, width)  # pylint: disable=I0011,E1101
        output_expected = np.pad(  # pylint: disable=I0011,E1101
            input_actual,
            ((0, 0), (0, 0), (0, fit_width - width)),
            mode='constant')
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            output_actual = sess.run(output, {input_: input_actual})
        self.assertAllClose(output_expected, output_actual)


class SoftmaxTest(tf.test.TestCase):
    """Test case for the `ops.softmax()` function."""

    def test_mask(self):
        """Test a masked softmax.

        The test checks that in correspondence of the masked
        elements the softmax evaluates to `0.0` and that the sum
        along the -1 axis is `1.0`.
        """

        logits = tf.constant([[3.0, 1.0, 0.2, 23.0],
                              [1.0, 23.0, 0.2, 3.0],
                              [23.0, 0.2, 3.0, 1.0]],
                             dtype=tf.float32)
        mask = tf.constant([[1.0, 1.0, 1.0, 0.0],
                            [1.0, 0.0, 1.0, 1.0],
                            [0.0, 1.0, 1.0, 1.0]],
                           dtype=tf.float32)
        softmax = ops.softmax(logits, mask=mask)

        output_expected = [[0.8360188, 0.11314284, 0.05083836, 0.0],
                           [0.11314284, 0.0, 0.05083836, 0.8360188],
                           [0.0, 0.05083836, 0.8360188, 0.11314284]]

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            output_actual = sess.run(softmax)
        self.assertAllClose(output_expected, output_actual, rtol=1e-5, atol=1e-5)

    def test_no_mask(self):
        """Test a softmax without a mask."""

        logits = tf.constant([[[3.0, 1.0, 0.2],
                               [1.0, 0.2, 3.0],
                               [0.2, 3.0, 1.0]],
                              [[3.0, 1.0, 0.2],
                               [1.0, 0.2, 3.0],
                               [0.2, 3.0, 1.0]]], dtype=tf.float32)
        softmax = ops.softmax(logits)

        output_expected = [[[0.83601880, 0.11314284, 0.05083836],
                            [0.11314284, 0.05083836, 0.83601880],
                            [0.05083836, 0.83601880, 0.11314284]],
                           [[0.83601880, 0.11314284, 0.05083836],
                            [0.11314284, 0.05083836, 0.83601880],
                            [0.05083836, 0.83601880, 0.11314284]]]

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            output_actual = sess.run(softmax)
        self.assertAllClose(output_actual, output_expected)


class TimesliceTest(tf.test.TestCase):
    """Test for the `ops.timeslice()` function."""

    def test_timeslice(self):
        """Test.
        """
        tensor = tf.placeholder(dtype=tf.float32, shape=[None, None, None])
        indices = tf.placeholder(dtype=tf.int32, shape=[None])
        outputs = ops.timeslice(tensor, indices)

        tensor_actual = np.array([[[0.01, 0.01, 0.01],
                                   [0.02, 0.02, 0.02],
                                   [0.03, 0.03, 0.03],
                                   [0.04, 0.04, 0.04]],
                                  [[0.1, 0.1, 0.1],
                                   [0.2, 0.2, 0.2],
                                   [23, 23, 23],
                                   [23, 23, 23]]],
                                 dtype=np.float32)  # pylint: disable=I0011,E1101
        indices_actual = np.array([3, 1], dtype=np.int32)  # pylint: disable=I0011,E1101
        outputs_expected = np.array([[0.04, 0.04, 0.04], [0.2, 0.2, 0.2]],
                                    dtype=np.float32)  # pylint: disable=I0011,E1101

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            outputs_actual = sess.run(outputs, {tensor: tensor_actual, indices: indices_actual})
        self.assertAllClose(outputs_expected, outputs_actual)


if __name__ == '__main__':
    tf.test.main()
