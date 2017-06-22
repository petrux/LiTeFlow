"""Streaming metrics implementation."""

import abc

import tensorflow as tf

from liteflow import streaming
from liteflow import utils

class StreamingMetric(streaming.StreamingComputation):
    """Streaming metric base class."""

    def __init__(self, func, average=None, name=None):
        super(StreamingMetric, self).__init__(name=name)
        self._func = func
        self._avg = average or streaming.StreamingAverage()

    @property
    def value(self):
        return self._avg.value

    @property
    def count(self):
        return self._avg.count

    @property
    def total(self):
        """The total values of the metric summed up so far."""
        return self._avg.value

    @property
    def batch_value(self):
        return self._avg.batch_value

    @property
    def batch_count(self):
        return self._avg.batch_count

    @property
    def batch_total(self):
        """The total value of the metric summed up for the current batch."""
        return self._avg.batch_total

    @property
    def update_op(self):
        return self._avg.update_op

    @property
    def reset_op(self):
        return self._avg.reset_op

    def compute(self, target, predictions, weights,
                metrics_collections=None,
                updates_collections=None,
                scope=None):
        """Compute the streaming metric."""
        # pylint: disable=I0011,E1129
        with tf.variable_scope(scope or self._name) as scope:
            values = self._func(target, predictions, weights)
            self._avg.compute(values, scope=scope)

        if metrics_collections:
            utils.add_to_collections(metrics_collections, self.value)

        if updates_collections:
            utils.add_to_collections(updates_collections, self.update_op)

    # pylint: disable=I0011,W0221
    def __call__(self, target, predictions, weights,
                 metrics_collections=None,
                 updates_collections=None,
                 scope=None):
        """Build the streaming metric."""
        self.compute(target, predictions, weights,
                     metrics_collections=metrics_collections,
                     updates_collections=updates_collections,
                     scope=scope)
        return self.value, self.update_op
