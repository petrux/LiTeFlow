"""Various utilities."""

import tensorflow as tf


def as_scope(scope):
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
    """Get variables by their name prefix."""
    prefix = prefix or tf.get_variable_scope().name
    return [var for var in tf.global_variables()
            if var.name.startswith(prefix)]
