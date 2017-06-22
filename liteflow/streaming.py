"""Streaming computation base utilities."""

import abc

import tensorflow as tf

from liteflow import ops


def _local_variable(name, init_value=0.0, dtype=tf.float32, trainable=False):
    return tf.Variable(
        name=name,
        dtype=dtype,
        initial_value=init_value,
        trainable=trainable,
        collections=tf.GraphKeys.LOCAL_VARIABLES)


class StreamingComputation(object):
    """Base contract for a streaming computation unit."""

    __metaclass__ = abc.ABCMeta

    def __init__(self, name=None):
        self._name = name or self.__class__.__name__

    @property
    def name(self):
        """The name of the current computation unit."""
        return self._name

    @abc.abstractproperty
    def value(self):
        """A tensor representing the value of the computation."""
        raise NotImplementedError()

    @abc.abstractproperty
    def count(self):
        """The number of items considered so far  as a `tf.int32` tensor."""
        raise NotImplementedError()

    @abc.abstractproperty
    def batch_value(self):
        """A tensor representing the value of the computation for the actual batch."""
        raise NotImplementedError()

    @abc.abstractproperty
    def batch_count(self):
        """The number of items in the current batch as a `tf.int32` tensor."""
        raise NotImplementedError()

    @abc.abstractproperty
    def update_op(self):
        """An op that if computed updates the streaming computation.

        The execution of this op MUST increment `self.count` of the number
        of items considered for the current batch and consequently adjust
        the `self.value`.
        """
        raise NotImplementedError()

    @abc.abstractproperty
    def reset_op(self):
        """An op that if computed reset the unit at its initil state

        The execution of this op MUST reset the value of `self.count` to 0
        and the value of `self.value` to the proper value.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def __call__(self, *args, **kwargs):
        """Build the computational graph for the streaming computation unit."""
        raise NotImplementedError()


class StreamingAverage(StreamingComputation):
    """Streaming average computation unit."""

    def __init__(self, name='StreamingAverage'):
        super(StreamingAverage, self).__init__(name=name)
        self._value = None
        self._count = None
        self._total = None
        self._batch_value = None
        self._batch_count = None
        self._batch_total = None
        self._update_op = None
        self._reset_op = None

    @property
    def value(self):
        return self._value

    @property
    def count(self):
        return self._count

    @property
    def total(self):
        """The total value summed up so far."""
        return self._value

    @property
    def batch_value(self):
        return self._batch_value

    @property
    def batch_count(self):
        return self._batch_count

    @property
    def batch_total(self):
        """The total value summed up for the current batch."""
        return self._batch_total

    @property
    def update_op(self):
        return self._update_op

    @property
    def reset_op(self):
        return self._reset_op

    def compute(self, values, weights=None, scope=None):
        """Compute the streaming weighted average.

        Arguments:
          values:
          weights:
          scope:
        """
        # pylint: disable=I0011,E1129
        with tf.variable_scope(scope or self._name) as scope:
            self._count = _local_variable('count')
            self._total = _local_variable('total')

            if weights:
                values = tf.multiply(values, weights)
                self._batch_count = tf.reduce_sum(weights, 'batch_count')
            else:
                self._batch_count = tf.to_float(tf.size(values), name='batch_count')

            self._batch_total = tf.reduce_sum(values, name='batch_total')
            self._batch_value = ops.safe_div(
                self._batch_total, self._batch_count, name='batch_value')

            update_total_op = tf.assign_add(self._total, self._batch_total)
            with tf.control_dependencies([values]):
                update_count_op = tf.assign_add(self._count, self._batch_count)
            self._update_op = ops.safe_div(update_total_op, update_count_op, 'update_op')

            self._value = ops.safe_div(self._total, self._count, 'value')

            self._reset_op = tf.group(
                self._total.initializer,
                self._count.initializer,
            )

    # pylint: disable=I0011,W0221
    def __call__(self, values, weights=None, scope=None):
        """Build the streaming average.

        Arguments:

        Returns:

        """
        self.compute(values, weights=weights, scope=scope)
        return self.value, self.update_op
