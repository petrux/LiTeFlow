"""Various utilities."""

import tensorflow as tf

def get_variables(prefix=None):
    """Get variables by their name prefix."""
    prefix = prefix or tf.get_variable_scope().name
    return [var for var in tf.global_variables()
            if var.name.startswith(prefix)]
