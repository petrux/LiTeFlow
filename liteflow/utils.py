"""Various utilities."""

import tensorflow as tf


def as_scope(scope):
    """Get the proper variable scope.

    Given an object that can represent a `tf.VariableScope`,
    namely a `str` or a `tf.VariableScope`, performs type checking
    and return a proper `tf.VariableScope` object. Such function is
    hancy when a function accepts an argument serving as a variable
    scope but doesn's know its proper type.

    Arguments:
      scope: a `str` or a `tf.VariableScope` representing a variable scope.

    Returns:
      a `tf.VariableScope` instance.

    Raises:
      ValueError: if `scope` is `None`.
      TypeError: if `scope` is neither `str` or `tf.VariableScope`.

    Example:
    ```python
    from dket import utils

    def do_something(scope):
        scope = utils.as_scope(scope or 'DefaultScope')
        with tf.variable_scope(scope) as scope:
            # do something
            pass
    ```
    """
    if scope is None:
        raise ValueError('Cannot create a scope from a None.')
    if isinstance(scope, str):
        return next(tf.variable_scope(scope).gen)  # pylint: disable=I0011,E1101
    if isinstance(scope, tf.VariableScope):
        return scope
    raise TypeError("""`scope` argument can be of type str, """
                    """tf.VariableScope, while %s found.""",
                    (str(type(scope))))


def get_variables(prefix=None):
    """Get variables by their name prefix.

    Arguments:
      prefix: a `str` or a `tf.VariableScope` instance.

    Returns:
      a list of `tf.Variable` with their name starting with the
        given prefix, i.e. all those variables under the scope
        specified by the prefix.
    """
    prefix = prefix or tf.get_variable_scope().name
    return [var for var in tf.global_variables()
            if var.name.startswith(prefix)]


def dypes(tensors):
    """Get the `dtype` for tensors in a list.

    Arguments:
      tensors: an iterable of `tf.Tensor`.

    Returns:
      a `list` of `dtype`s, one for each tensor in `tensors`,
        representing their `dtype`.
    """
    return [t.dtype for t in tensors]


def shapes(tensors):
    """Get the static shapes of tensors in a list.

    Arguments:
      tensors: an iterable of `tf.Tensor`.

    Returns:
      a `list` of `tf.TensorShape`, one for each tensor in `tensors`,
        representing their static shape (via `tf.Tensor.get_shape()`).
    """
    return [t.get_shape() for t in tensors]


def get_dimension(tensor, dim, ensure_tensor=False):
    """Returns a dimension of a tensor.

    Given a `Tensor`, retutns the `dim`-th dimension. If the
    dimension is specified, the `int` value of the dimension from the
    static shape of the tensor will be returned, otherwise a tensor
    representing the `dim`-th slice of the dynamic shape. To force
    the result to be always a tensor, set the `ensure_tensor`
    argument to `True`.

    Arguments:
      tensor: a `Tensor`.
      dim: a `int` representing the 0-based index of one of the dimensions of `tensor`.
      ensure_tensor: default `False`, ensure the result to be a `Tensor`.

    Returns:
      a `int` or a `Tensor` representing the value of the`dim`-th
      dimension of the `tensor`. If `ensure_tensor` is set to `True`
      a `Tensor` is always returned.

    Raises:
      TypeError: if `tensor` or `dim` are None.
      IndexError: if `dim` not compatible with the rank (i.e. the number
        of dimensions) of `tensor`, if specified.

    Example:
    ```
    >>> import tensorflow as tf
    >>> from liteflow import utils
    ```
    """

    if tensor is None:
        raise TypeError('`tensor` cannot be of `None` type.')
    if dim is None:
        raise TypeError('`dim` cannot be of `None` type.')

    dimension = tensor.shape[dim]
    if dimension.value is None:
        return tf.shape(tensor)[dim]
    if ensure_tensor:
        return tf.convert_to_tensor(dimension.value)
    return dimension.value
