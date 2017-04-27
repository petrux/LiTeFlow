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
