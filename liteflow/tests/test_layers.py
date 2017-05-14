"""Test module for liteflow.layers module."""

import numpy as np
import tensorflow as tf

import mock

from liteflow import layers
from liteflow import utils


class Layer(layers.Layer):
    """Dummy `layers.Layer` implementation."""

    def __init__(self, scope=None):
        super(Layer, self).__init__(scope=scope)

    def _build(self, *args, **kwargs):
        return None

    def _call(self, inp, *args, **kwargs):
        return None


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
        if len(self._scopes) == 0:
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
        """Create a ScopeTracker with an empty inner function."""
        def _id(inp, *args, **kwargs):
            _, _ = args, kwargs
            return inp
        return ScopeTracker(_id)


def scopetracker(func):
    """Turns your function into a ScopeTracker.

    ```python
    # Use this function as a decorator

    import tensorflow as tf

    @scopetracker
    def fun(x, y, *args, **kwargs):
        return x + y

    witf tf.variable_scope('Scope') as scope:
        print(fun(22, 1))
    print fun.laters().name

    >>> 23
    >>> Scope
    ```
    """
    return ScopeTracker(func)


class LayerTest(tf.test.TestCase):
    """Test case for a generic liteflow.layers.Layer implementation."""

    @mock.patch.object(Layer, '_call')
    @mock.patch.object(Layer, '_build')
    def _test_init_scope(self, scope, _build=None, _call=None):
        """ Initialize a layer with a given scope and build it with no scope.

        1. Initializes the layer with the given scope;
        2. Build the layer without any scope argument;
        3. Assert that the _build() method has nee invoked once;
        4. Assert that the tracked scope during the _build() phase
           is coherent with the layer name ans its scope name.
        5. Call twice the .apply() method
        6. Assert each time that the scope is coherent with the initial
           one and that the _build() method has been invoked once (while
           the call count of _call() increases).
        """

        _build.side_effect = ScopeTracker.empty()
        _call.side_effect = ScopeTracker.identity()

        layer = Layer(scope=scope)

        self.assertFalse(layer.built)
        self.assertEquals(layer.name, layer.scope.name)
        _build.assert_not_called()
        _call.assert_not_called()

        layer.build()
        self.assertTrue(layer.built)
        _build.assert_called_once()
        _call.assert_not_called()
        self.assertEquals(1, len(_build.side_effect.scopes()))
        self.assertEquals(layer.scope.name, _build.side_effect.latest().name)

        inp = object()
        out = layer.apply(inp)
        self.assertEquals(inp, out)
        _call.assert_called_once()
        self.assertEquals(1, len(_call.side_effect.scopes()))
        self.assertEquals(layer.scope.name, _call.side_effect.latest().name)
        _build.assert_called_once()
        self.assertEquals(1, len(_build.side_effect.scopes()))
        self.assertEquals(layer.scope.name, _build.side_effect.latest().name)

        inp = object()
        out = layer.apply(inp)
        self.assertEquals(inp, out)
        self.assertEquals(2, _call.call_count)
        self.assertEquals(2, len(_call.side_effect.scopes()))
        self.assertEquals(layer.scope.name, _call.side_effect.latest().name)
        _build.assert_called_once()
        self.assertEquals(1, len(_build.side_effect.scopes()))
        self.assertEquals(layer.scope.name, _build.side_effect.latest().name)

    def test_init_scope(self):
        """Test the layer with the scope initialization."""
        self._test_init_scope('Scope')
        self._test_init_scope(utils.as_scope('Scope'))

    @mock.patch.object(Layer, '_call')
    @mock.patch.object(Layer, '_build')
    def _test_build_scope(self, scope, _build=None, _call=None):
        """ Initialize a layer without a scope and build it the given scope.

        1. Initializes the layer with no scope;
        2. Build the layer with the given scope argument;
        3. Assert that the _build() method has nee invoked once;
        4. Assert that the tracked scope during the _build() phase
           is coherent with the layer name ans its scope name.
        5. Call twice the .apply() method
        6. Assert each time that the scope is coherent with the initial
           one and that the _build() method has been invoked once (while
           the call count of _call() increases).
        """

        _build.side_effect = ScopeTracker.empty()
        _call.side_effect = ScopeTracker.identity()

        layer = Layer(scope=None)

        self.assertFalse(layer.built)
        self.assertIsNone(layer.scope)
        self.assertIsNotNone(layer.name)
        _build.assert_not_called()
        _call.assert_not_called()

        layer.build(scope=scope)
        self.assertTrue(layer.built)
        _build.assert_called_once()
        _call.assert_not_called()
        self.assertEquals(1, len(_build.side_effect.scopes()))
        self.assertEquals(layer.scope.name, _build.side_effect.latest().name)
        self.assertEquals(layer.scope.name, utils.as_scope(scope).name)

        inp = object()
        out = layer.apply(inp)
        self.assertEquals(inp, out)
        _call.assert_called_once()
        self.assertEquals(1, len(_call.side_effect.scopes()))
        self.assertEquals(layer.scope.name, _call.side_effect.latest().name)
        _build.assert_called_once()
        self.assertEquals(1, len(_build.side_effect.scopes()))
        self.assertEquals(layer.scope.name, _build.side_effect.latest().name)

        inp = object()
        out = layer.apply(inp)
        self.assertEquals(inp, out)
        self.assertEquals(2, _call.call_count)
        self.assertEquals(2, len(_call.side_effect.scopes()))
        self.assertEquals(layer.scope.name, _call.side_effect.latest().name)
        _build.assert_called_once()
        self.assertEquals(1, len(_build.side_effect.scopes()))
        self.assertEquals(layer.scope.name, _build.side_effect.latest().name)

    def test_build_scope(self):
        """Test the scope explicitly building the layer."""
        self._test_build_scope('Scope')
        self._test_build_scope(utils.as_scope('Scope'))

    @mock.patch.object(Layer, '_call')
    @mock.patch.object(Layer, '_build')
    def _test_call_scope(self, scope, _build=None, _call=None):
        """Init without a scope and use it in applying the layer.

        1. Initializes the layer with no scope;
        2. Assert that the _build()/_call() have been never invoked;
        3. Call the .apply() method with a given scope as a kwarg.
        4. Assert that _build()/_call() have been invoked once,
           that the scope has been set and its coherent with the given one.
        6. Run a second .apply()
        """
        _build.side_effect = ScopeTracker.empty()
        _call.side_effect = ScopeTracker.identity()

        layer = Layer()
        self.assertFalse(layer.built)
        self.assertIsNone(layer.scope)
        self.assertIsNotNone(layer.name)
        _build.assert_not_called()
        _call.assert_not_called()

        inp = object()
        out = layer.apply(inp, scope=scope)
        self.assertIsNotNone(layer.scope)
        self.assertEquals(utils.as_scope(scope).name, layer.scope.name)
        self.assertEquals(inp, out)
        _call.assert_called_once()
        self.assertEquals(1, len(_call.side_effect.scopes()))
        self.assertEquals(layer.scope.name, _call.side_effect.latest().name)
        _build.assert_called_once()
        self.assertEquals(1, len(_build.side_effect.scopes()))
        self.assertEquals(layer.scope.name, _build.side_effect.latest().name)

    def test_call_scope(self):
        """Test the scope straight calling the layer function."""
        self._test_call_scope('Scope')
        self._test_call_scope(utils.as_scope('Scope'))

    def test_build_twice(self):
        """Building the layer twice wil raise a RuntimeError."""
        layer = Layer()
        layer.build()
        self.assertRaises(RuntimeError, layer.build)

    @mock.patch.object(Layer, '_call')
    @mock.patch.object(Layer, '_build')
    def test_no_scope(self, _build, _call):
        """Run a layer without a scope.

        1. Initializes the layer with no scope;
        2. Build the layer with the given scope argument;
        3. Assert that the _build() method has been invoked once and
           that the scope name is the layer.name (default one);
        4. Call twice the .apply() method without scope;
        6. Assert each time that the scope is coherent with the layer.name
           one and that the _build() method has been invoked once (while
           the call count of _call() increases).
        """

        _build.side_effect = ScopeTracker.empty()
        _call.side_effect = ScopeTracker.identity()

        layer = Layer()
        self.assertFalse(layer.built)
        self.assertIsNone(layer.scope)
        self.assertIsNotNone(layer.name)
        _build.assert_not_called()
        _call.assert_not_called()

        layer.build()
        self.assertTrue(layer.built)
        _build.assert_called_once()
        _call.assert_not_called()
        self.assertIsNotNone(layer.scope)
        self.assertEquals(layer.name, layer.scope.name)
        self.assertEquals(1, len(_build.side_effect.scopes()))
        self.assertEquals(layer.scope.name, _build.side_effect.latest().name)

        inp = object()
        out = layer.apply(inp)
        self.assertEquals(inp, out)
        _call.assert_called_once()
        self.assertEquals(1, len(_call.side_effect.scopes()))
        self.assertEquals(layer.scope.name, _call.side_effect.latest().name)
        _build.assert_called_once()
        self.assertEquals(1, len(_build.side_effect.scopes()))
        self.assertEquals(layer.scope.name, _build.side_effect.latest().name)

        inp = object()
        out = layer.apply(inp)
        self.assertEquals(inp, out)
        self.assertEquals(2, _call.call_count)
        self.assertEquals(2, len(_call.side_effect.scopes()))
        self.assertEquals(layer.scope.name, _call.side_effect.latest().name)
        _build.assert_called_once()
        self.assertEquals(1, len(_build.side_effect.scopes()))
        self.assertEquals(layer.scope.name, _build.side_effect.latest().name)

    def test_reuse(self):
        """Test a layer with reusable scope.

        NOTA BENE: reuse a scope means that somehow the client
        class KNOWS the variable that will be created by the layer
        and its internal logic. So, the need of reusing a scope can
        be a strong signal that your design is violating the incapsulation
        and that you need a refactoring.
        """

        def _init_scope(scope, *args):
            layer = Layer(scope=scope)
            layer.build()
            for arg in args:
                layer.apply(arg)
        tf.reset_default_graph()
        self._test_reuse(_init_scope)

        def _build_scope(scope, *args):
            layer = Layer()
            layer.build(scope=scope)
            for arg in args:
                layer.apply(arg)
        tf.reset_default_graph()
        self._test_reuse(_build_scope)

        def _apply_scope(scope, *args):
            layer = Layer()
            for i, arg in enumerate(args):
                if i == 0:
                    layer.apply(arg, scope=scope)
                else:
                    layer.apply(arg)
        tf.reset_default_graph()
        self._test_reuse(_apply_scope)

    @mock.patch.object(Layer, '_call')
    @mock.patch.object(Layer, '_build')
    def _test_reuse(self, perform, _build=None, _call=None):
        """Test the layer with a reusable scope."""
        scope = 'Scope'
        build_var_name = 'BuildVar'
        build_var_op_name = scope + '/' + build_var_name
        call_var_name = 'CallVar'
        call_var_op_name = scope + '/' + call_var_name
        shape = [2, 2]

        @scopetracker
        def _do_build(*args, **kwargs):
            _, _ = args, kwargs
            _ = tf.get_variable(name=build_var_name, shape=shape)
        _build.side_effect = _do_build

        @scopetracker
        def _do_call(_=None, *args, **kwargs):
            _, _ = args, kwargs
            var = tf.get_variable(name=call_var_name, shape=shape)
            return var
        _call.side_effect = _do_call

        inputs = [object(), object()]

        # initialize the variables in the scope
        with tf.variable_scope(scope) as scope:
            _do_build()
            _do_call()
            scope.reuse_variables()
            perform(scope, *inputs)
            variables = utils.get_variables(scope.name)
            self.assertEquals(1, _build.call_count)
            self.assertEquals(len(inputs), _call.call_count)
            self.assertEquals(2, len(variables))
            actual = sorted([v.op.name for v in variables])
            expected = sorted([build_var_op_name, call_var_op_name])
            for act, exp in zip(actual, expected):
                self.assertEquals(act, exp)

    @mock.patch.object(Layer, '_build')
    def test_reuse_unexisting_build(self, _build):
        """Building with a scope were variables do not exist raise an Error."""
        def _do_build(*args, **kwargs):
            _, _ = args, kwargs
            _ = tf.get_variable('Variable', shape=[2, 2])
        _build.side_effect = _do_build

        with tf.variable_scope('Scope') as scope:
            scope.reuse_variables()
            layer = Layer(scope=scope)
            with self.assertRaises(ValueError) as context:
                layer.build()
            self.assertTrue(isinstance(context.exception, ValueError))
            self.assertTrue(context.exception.message.startswith(
                """Variable Scope/Variable does not exist, """ +
                """or was not created with tf.get_variable()"""))

    @mock.patch.object(Layer, '_call')
    def test_reuse_unexisting_call(self, _call):
        """Calling with a scope were variables do not exist raise an Error."""
        def _do_call(_=None, *args, **kwargs):
            _, _ = args, kwargs
            _ = tf.get_variable('Variable', shape=[2, 2])
        _call.side_effect = _do_call

        with tf.variable_scope('Scope') as scope:
            scope.reuse_variables()
            layer = Layer(scope=scope)
            layer.build()
            with self.assertRaises(ValueError) as context:
                layer.apply(object())
            self.assertTrue(isinstance(context.exception, ValueError))
            self.assertTrue(context.exception.message.startswith(
                """Variable Scope/Variable does not exist, """ +
                """or was not created with tf.get_variable()"""))

    @mock.patch.object(Layer, '_build')
    def test_reuse_unreusable_build(self, _build):
        """Using a scope where variables already exists raises an Error."""
        def _do_build(*args, **kwargs):
            _, _ = args, kwargs
            _ = tf.get_variable('Variable', shape=[2, 2])
        _build.side_effect = _do_build

        with tf.variable_scope('Scope') as scope:
            _do_build()
            layer = Layer(scope=scope)
            with self.assertRaises(ValueError) as context:
                layer.build()
            self.assertTrue(isinstance(context.exception, ValueError))
            self.assertTrue(context.exception.message.startswith(
                'Variable Scope/Variable already exists'))

    @mock.patch.object(Layer, '_call')
    def test_reuse_unreusable_call(self, _call):
        """Using a scope where variables already exists raises an Error."""
        def _do_call(_=None, *args, **kwargs):
            _, _ = args, kwargs
            _ = tf.get_variable('Variable', shape=[2, 2])
        _call.side_effect = _do_call

        with tf.variable_scope('Scope') as scope:
            _do_call()
            layer = Layer(scope=scope)
            with self.assertRaises(ValueError) as context:
                layer.apply(object())
            self.assertTrue(isinstance(context.exception, ValueError))
            self.assertTrue(context.exception.message.startswith(
                'Variable Scope/Variable already exists'))

    @mock.patch.object(Layer, '_call')
    @mock.patch.object(Layer, '_build')
    def test_change_scope(self, _build, _call):
        """Once set, the scope doesn't change.

        NOTA BENE: you should see some warning in the logs.
        """
        scope = utils.as_scope('Scope01')
        layer = Layer(scope=scope)
        self.assertEquals(scope.name, layer.scope.name)

        layer.build(scope='Scope02')
        self.assertEquals(scope.name, layer.scope.name)

        inp = object()
        _ = layer(inp, scope='Scope03')
        self.assertEquals(scope.name, layer.scope.name)
        self.assertEquals(1, _call.call_count)
        args, _ = _call.call_args
        self.assertEquals(inp, args[0])

        inp = object()
        _ = layer(inp, scope='Scope04')
        self.assertEquals(scope.name, layer.scope.name)
        self.assertEquals(2, _call.call_count)
        args, _ = _call.call_args
        self.assertEquals(inp, args[0])

        self.assertEquals(1, _build.call_count)


class BahdanauAttentionTest(tf.test.TestCase):
    """Test case for the liteflow.layers.BahdanauAttention class."""

    _SEED = 23

    def setUp(self):
        tf.reset_default_graph()
        tf.set_random_seed(seed=self._SEED)
        np.random.seed(seed=self._SEED)

    def _get_names(self, key):
        collection = tf.get_collection(key)
        names = [var.op.name for var in collection]
        return set(sorted(names))

    def _assert_all_in(self, variables, key):
        ref = self._get_names(key)
        for var in variables:
            self.assertIn(
                var.op.name, ref,
                var.op.name + ' not in ' + key)

    def _assert_none_in(self, variables, key):
        ref = self._get_names(key)
        for var in variables:
            self.assertNotIn(
                var.op.name, ref,
                var.op.name + ' in ' + key)

    def test_base(self):
        """Test the attention mechanism.

        1. Test that the attention scores that are computed
           are similar to the gold ones.
        2. check that all the attention layer variables are in the
           tf.GraphKeys.GLOBAL_VARIABLES
        3. check that the same variables are/are not in the
           tf.GraphKeys.TRAINABLE_VARIABLES, depending on if the
           layer is trainable or not.
        """
        self._test_base(trainable=True)
        self._test_base(trainable=False)

    def _test_base(self, trainable):
        state_size = 3
        query_size = 2
        attention_size = 4

        states = np.array([[[0.1, 0.1, 0.1],
                            [0.2, 0.2, 0.2],
                            [0.3, 0.3, 0.3],
                            [0.4, 0.4, 0.4],
                            [0.5, 0.5, 0.5]],
                           [[1., 1., 1.],
                            [0.9, 0.9, 0.9],
                            [0.8, 0.8, 0.8],
                            [0.7, 0.7, 0.7],
                            [0.6, 0.6, 0.6]]])
        queries = [np.asarray([[1, 1], [2, 2]]),
                   np.asarray([[.5, .5], [4, 4]])]
        scores = [np.array([[0.0904, 0.1017, 0.1128, 0.1238, 0.1345],
                            [0.2417, 0.2339, 0.2259, 0.2176, 0.2090]]),
                  np.array([[0.0517, 0.0634, 0.0750, 0.0866, 0.0979],
                            [0.3201, 0.3157, 0.3111, 0.3063, 0.3012]])]

        tf.reset_default_graph()
        self.assertEquals(0, len(self._get_names(
            tf.GraphKeys.TRAINABLE_VARIABLES)))
        init = tf.constant_initializer(0.1)
        with tf.variable_scope('Scope', initializer=init) as scope:
            states_ = tf.placeholder(dtype=tf.float32, shape=[
                                     None, None, state_size], name='S')
            queries_ = [tf.placeholder(dtype=tf.float32, shape=[None, query_size], name='q01'),
                        tf.placeholder(dtype=tf.float32, shape=[None, query_size], name='q02')]
            attention = layers.BahdanauAttention(
                states_, attention_size, trainable=trainable, scope=scope)
            self.assertEquals(trainable, attention.trainable)
            scores_ = [attention(q) for q in queries_]

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            results = sess.run(scores_, {
                states_: states,
                queries_[0]: queries[0],
                queries_[1]: queries[1]
            })

        for act, exp in zip(results, scores):
            self.assertAllClose(act, exp, rtol=1e-4, atol=1e-4)

        variables = attention.variables
        self._assert_all_in(variables, tf.GraphKeys.GLOBAL_VARIABLES)
        if trainable:
            self._assert_all_in(variables, tf.GraphKeys.TRAINABLE_VARIABLES)
        else:
            self._assert_none_in(variables, tf.GraphKeys.TRAINABLE_VARIABLES)


class TestPointingSoftmax(tf.test.TestCase):
    """Test case for the `liteflow.layers.PointingSoftmax` class."""

    _SEED = 23

    def setUp(self):
        tf.reset_default_graph()
        tf.set_random_seed(seed=self._SEED)
        np.random.seed(seed=self._SEED)

    def test_base(self):
        """Basic usage of the `liteflow.layers.PointingSoftmax` class."""

        query = tf.constant(
            [[0.05, 0.05, 0.05], [0.07, 0.07, 0.07]], dtype=tf.float32)

        states_np = np.asarray(
            [[[0.01, 0.01, 0.01], [0.02, 0.02, 0.02], [0.03, 0.03, 0.03], [0.04, 0.04, 0.04]],
             [[0.1, 0.1, 0.1], [0.2, 0.2, 0.2], [23.0, 23.0, 23.0], [23.0, 23.0, 23.0]]])
        states = tf.placeholder(dtype=tf.float32, shape=[None, None, 3])
        lengths = tf.constant([4, 2], dtype=tf.int32)
        mask = tf.cast(tf.sequence_mask(
            lengths, tf.shape(states)[1]), tf.float32)

        activations = tf.constant(
            [[1, 1, 1, 1], [1, 2, 10, 10]], dtype=tf.float32)

        exp_weights = np.asarray(
            [[0.25, 0.25, 0.25, 0.25], [0.2689414, 0.7310586, 0.0, 0.0]])
        exp_context = np.asarray(
            [[0.025, 0.025, 0.025], [0.17310574, 0.17310574, 0.17310574]])

        attention = mock.Mock()
        attention.states = states
        attention.apply.side_effect = [activations, activations]

        layer = layers.PointingSoftmax(attention, mask=mask)
        weights, context = layer(query)
        _, _ = layer(query)
        self.assertEqual(1, attention.build.call_count)
        self.assertTrue(layer.built)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            feed_dict = {
                states: states_np
            }
            act_weights, act_context = sess.run(
                [weights, context], feed_dict=feed_dict)

        self.assertAllClose(act_weights, exp_weights)
        self.assertAllClose(act_context, exp_context)

    def test_build(self):
        """Check that the build operation bounces on the injected attention."""

        attention = mock.Mock()
        layer = layers.PointingSoftmax(attention)
        self.assertFalse(layer.built)
        self.assertEqual(0, attention.build.call_count)

        layer.build()
        self.assertTrue(layer.built)
        self.assertEqual(1, attention.build.call_count)


class TestPointingSoftmaxOutput(tf.test.TestCase):
    """Test case for the PointingSoftmaxOutput layer."""

    def test_build(self):
        """Test that the  building phase goes upstream to the injected layer."""

        pointing = mock.Mock()
        layer = layers.PointingSoftmaxOutput(
            pointing=pointing,
            emission_size=10,
            feedback_size=None,
            decoder_out_size=7,
            attention_size=4)
        self.assertFalse(layer.built)
        self.assertEqual(0, pointing.build.call_count)

        layer.build()
        self.assertTrue(layer.built)
        self.assertEqual(1, pointing.build.call_count)


if __name__ == '__main__':
    tf.test.main()
