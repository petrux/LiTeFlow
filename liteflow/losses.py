"""Streaming loss implementation."""

import tensorflow as tf

from liteflow import ops
from liteflow import streaming
from liteflow import utils


class StreamingLoss(streaming.StreamingComputation):
    """Streaming loss base class.

    This class implements the basic infrastructure to build a streaming
    average based loss. It works wrapping a loss function accepting as arguments
    the ground truth values, the predictions and a weight tensor, and returns a
    point-wise loss tensor and another weight tensor, to be used in the streaming
    average.

    The class has a method `compute` that builds the computational graph in charge
    of computing the streaming loss. Such method accepts the following arguments:
      targets: a `Tensor` representing the gold truth values.
      predictions: a `Tensor` representing the predicted values.
      weights: optional `Tensor` to weight the loss.
      loss_collections: a list of string representing the keys of the collections
        to which the `Tensor` in `self.value` *AND* the tensor in `self.batch_value`
        must be added.
      scope: the variable scope name to be used to build the portion of the graph.

    The class implements also a __call__ interface accepting the very same arguments
    and returning:
      values: a `Tensor` of the same shape of `targets` and of type `tf.float32`
        representing the weighted contribution to the loss of each element.
      weights: a `Tensor` of the same shape as `targets` and of type `tf.float32`
        representing the weight of each element in a weighted average (in this case,
        it's the same as the input, if provided).

    Please, note that the `targets`, `predictions` and `weights` argument will be
    passed as-is as the input argument of the wrapped function.

    Example:
    ```python
    def abs_error(targets, perdictions, weights=None):
        values = tf.abs(targets - predictions)
        if weights is not None:
            values = tf.multiply(weights, values)
        return values, weights

    targets = ...
    predictions = ...
    weights = ...
    loss = StreamingLoss(func=abs_error, name='AbsError')
    ```
    """

    def __init__(self, func, average=None, name=None):
        super(StreamingLoss, self).__init__(name=name)
        self._func = func
        self._avg = average or streaming.StreamingAverage()

    @property
    def value(self):
        """The current value of the loss."""
        return self._avg.value

    @property
    def count(self):
        """The number of elements seen so far."""
        return self._avg.count

    @property
    def total(self):
        """The total values of the loss summed up so far."""
        return self._avg.value

    @property
    def batch_value(self):
        """The value of the loss for the current batch."""
        return self._avg.batch_value

    @property
    def batch_count(self):
        """The number of elements in the current batch."""
        return self._avg.batch_count

    @property
    def batch_total(self):
        """The total value of the loss summed up for the current batch."""
        return self._avg.batch_total

    @property
    def update_op(self):
        """Updates the current value of the loss."""
        return self._avg.update_op

    @property
    def reset_op(self):
        """Reset the streaming computation of the loss."""
        return self._avg.reset_op

    def compute(self, targets, predictions, weights=None,
                loss_collections=None, scope=None):
        """Build the graph portion computing the streaming average loss.

        Arguments:
          targets: a `Tensor` representing the gold truth values.
          predictions: a `Tensor` representing the predicted values.
          weights: optional `Tensor` to weight the loss.
          loss_collections: a list of string representing the keys of the collections
            to which the `Tensor` in `self.value` *AND* the tensor in `self.batch_value`
            must be added.
          scope: the variable scope name to be used to build the portion of the graph.
        """

        # pylint: disable=I0011,E1129
        with tf.variable_scope(scope or self.name) as scope:
            values, weights = self._func(targets, predictions, weights)
            self._avg.compute(values, weights=weights, scope=scope)

        if loss_collections:
            utils.add_to_collections(loss_collections, self.value)
            utils.add_to_collections(loss_collections, self.batch_value)

    # pylint: disable=I0011,W0221
    def __call__(self, targets, predictions, weights=None,
                 loss_collections=None, scope=None):
        """Computes omputing the streaming average loss.

        Arguments:
          targets: a `Tensor` representing the gold truth values.
          predictions: a `Tensor` representing the predicted values.
          weights: optional `Tensor` to weight the loss.
          loss_collections: a list of string representing the keys of the collections
            to which the `Tensor` in `self.value` *AND* the tensor in `self.batch_value`
            must be added.
          scope: the variable scope name to be used to build the portion of the graph.

        Returns:
          value: a `Tensor` representing the streaming average value of the loss.
          update_op: an `Op` that updates the streaming average value.
        """
        self.compute(targets, predictions, weights=weights,
                     loss_collections=loss_collections, scope=scope)
        return self.value, self.update_op


def categorical_crossentropy(targets, predictions, weights=None):
    """Computes the categorical cross entropy.

    Arguments:
      targets: `Tensor` of shape `[d_0, d_1, ..., d_{r-1}]` and type `tf.int32`
        Each entry in `labels` must be an index in `[0, num_classes)`. Other
        values will raise an exception when this op is run on CPU, and return
        `NaN` for corresponding loss and gradient rows on GPU.
      preictions: `Tensor` of shape `[d_0, d_1, ..., d_{r-1}, num_classes]`
        and type `tf.float32` representing a probability distribution across
        the output classes.
      weights: a `Tensor` of the same shape of `targets` (or broadcastable)
        representing the weight of the element loss.

    Returns:
      values: a `Tensor` of the same shape of `targets` and of type `tf.float32`
        representing the weighted contribution to the loss of each element.
      weights: a `Tensor` of the same shape as `targets` and of type `tf.float32`
        representing the weight of each element in a weighted average (in this case,
        it's the same as the input, if provided).
    """
    eps = tf.convert_to_tensor(ops.EPSILON)
    logits = tf.log(tf.clip_by_value(predictions, eps, 1 - eps))
    values = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=targets, logits=logits, name='xentropy')
    if weights is not None:
        values = tf.multiply(values, weights)
    return values, weights
