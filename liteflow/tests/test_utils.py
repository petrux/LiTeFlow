"""Test module for the `liteflow.utils` module."""

import unittest

import numpy as np
import tensorflow as tf

from liteflow import utils


class TestGetDimension(unittest.TestCase):
    """Test case for the `liteflow.utils.get_dimension` function."""

    def test_default(self):
        """Basic test for the `liteflow.utils.get_dimension`."""
        exp_dim_0, exp_dim_1, exp_dim_2 = 9, 2, 3
        tensor = tf.placeholder(dtype=tf.float32, shape=[None, exp_dim_1, exp_dim_2])
        data = np.ones((exp_dim_0, exp_dim_1, exp_dim_2))

        dim_0 = utils.get_dimension(tensor, 0)
        dim_1 = utils.get_dimension(tensor, 1, ensure_tensor=True)
        dim_2 = utils.get_dimension(tensor, 2)

        # dim_0 and dim_1 are tensors and must be evaluated.
        with tf.Session() as sess:
            act_dim_0, act_dim_1 = sess.run([dim_0, dim_1], {tensor: data})
        self.assertEqual(exp_dim_0, act_dim_0)
        self.assertEqual(exp_dim_1, act_dim_1)

        # dim_2 is an integer and is equal to the expected value.
        self.assertEqual(int, type(dim_2))
        self.assertEqual(exp_dim_2, dim_2)

    def test_none_tensor(self):
        """A None tensor raises a TypeError."""
        self.assertRaises(TypeError, utils.get_dimension, None, 0)

    def test_none_dim(self):
        """A None dimension index raises a TypeError."""
        tensor = tf.placeholder(dtype=tf.float32, shape=[9, 23])
        self.assertRaises(TypeError, utils.get_dimension, tensor, None)

    def test_dimension_lt_zero(self):
        """A dimension index less than zero raises a ValueError."""
        tensor = tf.placeholder(dtype=tf.float32)
        self.assertRaises(ValueError, utils.get_dimension, tensor, -1)

    def test_dimension_ge_rank(self):
        """A dimension index greater or equal than the rank raises a ValueError."""
        tensor = tf.placeholder(dtype=tf.float32, shape=[9, 23])
        self.assertRaises(ValueError, utils.get_dimension, tensor, 2)

    def test_unspecified_shape(self):
        """Test with unspecified tensor shape."""

        tensor = tf.placeholder(dtype=tf.float32)

        shape = (9, 2, 3)
        data = np.ones(shape)
        rank = len(shape)
        dims = [utils.get_dimension(tensor, d) for d in range(rank)]

        with tf.Session() as sess:
            act_dims = sess.run(dims, {tensor: data})
        for act, exp in zip(act_dims, shape):
            self.assertEqual(act, exp)

        invalid_dim = utils.get_dimension(tensor, rank)
        with tf.Session() as sess:
            self.assertRaises(
                tf.errors.InvalidArgumentError,
                sess.run, invalid_dim, {tensor: data})
