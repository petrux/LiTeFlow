"""Test module for `liteflow.model` module."""

import unittest

import tensorflow as tf

from liteflow import model


class _TestModel(model.BaseModel):

    _HPARAMS = {
        'seed': 23,
        'label': 'CIAONE!',
        'ratio': 0.09523809523809523,
        'cool': True,
        'none': None,
    }

    def get_default_hparams(self):
        hparams = tf.contrib.training.HParams()
        for key, value in self._HPARAMS.iteritems():
            hparams.add_hparam(key, value)
        return hparams

    def _init(self): pass  # pylint: disable=I0011,C0321
    def input(self): pass  # pylint: disable=I0011,C0321
    def target(self): pass  # pylint: disable=I0011,C0321
    def output(self): pass  # pylint: disable=I0011,C0321
    def loss(self): pass  # pylint: disable=I0011,C0321
    def train(self): pass  # pylint: disable=I0011,C0321
    def summary(self): pass  # pylint: disable=I0011,C0321
    def eval(self): pass  # pylint: disable=I0011,C0321


class BaseModelTest(unittest.TestCase):
    """Basic tests for the `liteflow.model.BaseModel` class."""

    def test_hparams(self):
        """Test the HParams settings of the model."""
        hparams = tf.contrib.training.HParams()
        hparams.add_hparam('seed', 90)
        hparams.add_hparam('extra', 17)
        hparams.add_hparam('ratio', None)

        instance = _TestModel(hparams)
        actual = instance.hparams
        default = instance.get_default_hparams()

        # Overridden hparams are actually overridden.
        self.assertEquals(actual.seed, hparams.seed)
        self.assertEquals(actual.ratio, hparams.ratio)

        # Ignored hparams keep their default value.
        self.assertEquals(actual.label, default.label)
        self.assertEquals(actual.cool, default.cool)
        self.assertEquals(actual.none, default.none)

        # Extra hparams are ignored.
        self.assertNotIn('extra', actual.values())
