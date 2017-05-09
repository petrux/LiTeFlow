"""Base implementation for a ML Model."""

import abc

import tensorflow as tf


class BaseModel(object):
    """Base model implementation."""

    __metaclass__ = abc.ABCMeta

    def __init__(self, hparams):
        self._global_step = tf.train.create_global_step()
        self._hparams = self._set_hparams(hparams)
        self._init()

    @abc.abstractmethod
    def get_default_hparams(self):
        """Returns the default `tf.contrib.training.HParams`"""
        raise NotImplementedError('This method must be implemented in subclasses')

    def _set_hparams(self, hparams):
        actual = hparams.values()
        default = self.get_default_hparams().values()
        merged = tf.contrib.training.HParams()
        for key, value in default.iteritems():
            if key in actual:
                value = actual[key]
            merged.add_hparam(key, value)
        return merged

    @abc.abstractmethod
    def _init(self):
        """Initialize the model."""
        raise NotImplementedError('This method must be implemented in subclasses')

    @property
    def global_step(self):
        """The global step of the model."""
        return self._global_step

    @property
    def hparams(self):
        """The full initialization HParams of the model."""
        return self._hparams

    @abc.abstractproperty
    def input(self):
        """A tensor or a dictionary of tensors represenitng the model input(s)."""
        raise NotImplementedError('This property must be implemented in subclasses')

    @abc.abstractproperty
    def target(self):
        """A tensor representing the target output of the model."""
        raise NotImplementedError('This property must be implemented in subclasses')

    @abc.abstractproperty
    def output(self):
        """A tensor representing the actual output of the model."""
        raise NotImplementedError('This property must be implemented in subclasses')

    @abc.abstractproperty
    def loss(self):
        """The loss op of the model."""
        raise NotImplementedError('This property must be implemented in subclasses')

    @abc.abstractproperty
    def train(self):
        """The train op of the model."""
        raise NotImplementedError('This property must be implemented in subclasses')

    @abc.abstractproperty
    def summary(self):
        """The summary op of the model."""
        raise NotImplementedError('This property must be implemented in subclasses')

    @abc.abstractproperty
    def eval(self):
        """The evaluation op of the model."""
        raise NotImplementedError('This property must be implemented in subclasses')
