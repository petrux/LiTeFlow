"""Test the base class of the layers hierarchy."""

# Disable pylint warning about too many statements
# and local variables since we are dealing with tests.
# pylint: disable=R0914, R0915

import mock
import tensorflow as tf

from liteflow import layers, utils


class Layer(layers.Layer):
    """Dummy `layers.Layer` implementation."""

    def __init__(self, scope=None):
        super(Layer, self).__init__(scope=scope)

    def _call_helper(self, *args, **kwargs):  # pylint: disable=I0011,W0221
        return RuntimeError('This method should be patched!')


class ScopeTracker(object):
    """Function class that just tracks the tf.VariableScope."""

    def __init__(self, fn=None):
        self._scopes = []
        self._fn = fn if fn is not None else lambda *args, **kwargs: None

    def scopes(self):
        """Returns the list of tracked scopes."""
        return self._scopes

    def latest(self):
        """Returns the latest tracked scope (or None)."""
        if not self._scopes:
            return None
        return self._scopes[-1]

    def __call__(self, *args, **kwargs):
        scope = tf.get_variable_scope()
        self._scopes.append(scope)
        return self._fn(*args, **kwargs)

    @classmethod
    def empty(cls):
        """Create a ScopeTracker with an empty inner function."""
        def _empty(*args, **kwargs):
            _, _ = args, kwargs
        return ScopeTracker(_empty)

    @classmethod
    def identity(cls):
        """Create a ScopeTracker with an inner function that returns its only argument."""
        return ScopeTracker(lambda x: x)


def scopetracker(func):
    """Turns your function into a ScopeTracker.

    ```python
    # Use this function as a decorator

    import tensorflow as tf

    @scopetracker
    def fun(x, y, *args, **kwargs):
        return x + y

    witf tf.variable_scope('Scope') as scope:
        print(fun(22, 1))  #  23
    print fun.laters().name  # 'Scope'
    ```
    """
    return ScopeTracker(func)


class LayerTest(tf.test.TestCase):
    """Test case for a generic liteflow.layers.Layer implementation."""

    @mock.patch.object(Layer, '_call_helper')
    def _test_scope(self, scope, _call_helper=None):

        _call_helper.side_effect = ScopeTracker.identity()

        layer = Layer()
        self.assertIsNotNone(layer.name)
        self.assertRaises(ValueError, lambda: layer.scope)
        _call_helper.assert_not_called()

        layer(object(), scope=scope)
        _call_helper.assert_called_once()
        self.assertEqual(1, len(_call_helper.side_effect.scopes()))
        self.assertEqual(layer.scope, _call_helper.side_effect.latest().name)

        layer(object(), scope=scope)
        self.assertEqual(2, _call_helper.call_count)
        self.assertEqual(2, len(_call_helper.side_effect.scopes()))
        self.assertEqual(layer.scope, _call_helper.side_effect.latest().name)

        layer(object(), scope=None)
        self.assertEqual(3, _call_helper.call_count)
        self.assertEqual(3, len(_call_helper.side_effect.scopes()))
        self.assertEqual(layer.scope, _call_helper.side_effect.latest().name)

        layer(object(), scope=layer.scope + '__ANOTHER')
        self.assertEqual(4, _call_helper.call_count)
        self.assertEqual(4, len(_call_helper.side_effect.scopes()))
        self.assertEqual(layer.scope, _call_helper.side_effect.latest().name)

    def test_scope(self):
        """Test the default behaviour of a layer."""
        self._test_scope(None)
        self._test_scope('Scope')
        self._test_scope(utils.as_scope('Scope'))

    @mock.patch.object(Layer, '_call_helper')
    def test_reuse_variables(self, _call_helper=None):
        """Test that the first invocation of the layer doesn't reuse variables."""
        scope_name = 'Scope'
        var_name = 'Variable'
        def _call(*args, **kwargs):
            _, _ = args, kwargs
            _ = tf.get_variable(name=var_name, shape=0)
        _call_helper.side_effect = _call

        with tf.variable_scope(scope_name) as scope:
            _call()

        with tf.variable_scope(scope_name) as scope:
            self.assertRaises(ValueError, Layer(), scope=scope)

        with tf.variable_scope(scope_name) as scope:
            scope.reuse_variables()
            Layer().__call__(scope=scope)

        with tf.variable_scope(scope_name + '__ANOTHER') as scope:
            scope.reuse_variables()
            self.assertRaises(ValueError, Layer(), scope=scope)