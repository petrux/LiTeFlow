"""Test module for liteflow.layers module."""

import numpy as np
import tensorflow as tf

import mock

from liteflow import layers
from liteflow import utils

# Disable pylint warning about too many statements
# and local variables since we are dealing with tests.
# pylint: disable=R0914, R0915


class Layer(layers.Layer):
    """Dummy `layers.Layer` implementation."""

    def __init__(self, scope=None):
        super(Layer, self).__init__(scope=scope)

    def _build(self, *args, **kwargs):
        return None

    def _call(self, inp, *args, **kwargs):  # pylint: disable=I0011,W0221
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
        self.assertEqual(layer.name, layer.scope.name)
        _build.assert_not_called()
        _call.assert_not_called()

        layer.build()
        self.assertTrue(layer.built)
        _build.assert_called_once()
        _call.assert_not_called()
        self.assertEqual(1, len(_build.side_effect.scopes()))
        self.assertEqual(layer.scope.name, _build.side_effect.latest().name)

        inp = object()
        out = layer.call(inp)
        self.assertEqual(inp, out)
        _call.assert_called_once()
        self.assertEqual(1, len(_call.side_effect.scopes()))
        self.assertEqual(layer.scope.name, _call.side_effect.latest().name)
        _build.assert_called_once()
        self.assertEqual(1, len(_build.side_effect.scopes()))
        self.assertEqual(layer.scope.name, _build.side_effect.latest().name)

        inp = object()
        out = layer.call(inp)
        self.assertEqual(inp, out)
        self.assertEqual(2, _call.call_count)
        self.assertEqual(2, len(_call.side_effect.scopes()))
        self.assertEqual(layer.scope.name, _call.side_effect.latest().name)
        _build.assert_called_once()
        self.assertEqual(1, len(_build.side_effect.scopes()))
        self.assertEqual(layer.scope.name, _build.side_effect.latest().name)

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
        self.assertEqual(1, len(_build.side_effect.scopes()))
        self.assertEqual(layer.scope.name, _build.side_effect.latest().name)
        self.assertEqual(layer.scope.name, utils.as_scope(scope).name)

        inp = object()
        out = layer.call(inp)
        self.assertEqual(inp, out)
        _call.assert_called_once()
        self.assertEqual(1, len(_call.side_effect.scopes()))
        self.assertEqual(layer.scope.name, _call.side_effect.latest().name)
        _build.assert_called_once()
        self.assertEqual(1, len(_build.side_effect.scopes()))
        self.assertEqual(layer.scope.name, _build.side_effect.latest().name)

        inp = object()
        out = layer.call(inp)
        self.assertEqual(inp, out)
        self.assertEqual(2, _call.call_count)
        self.assertEqual(2, len(_call.side_effect.scopes()))
        self.assertEqual(layer.scope.name, _call.side_effect.latest().name)
        _build.assert_called_once()
        self.assertEqual(1, len(_build.side_effect.scopes()))
        self.assertEqual(layer.scope.name, _build.side_effect.latest().name)

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
        out = layer.call(inp, scope=scope)
        self.assertIsNotNone(layer.scope)
        self.assertEqual(utils.as_scope(scope).name, layer.scope.name)
        self.assertEqual(inp, out)
        _call.assert_called_once()
        self.assertEqual(1, len(_call.side_effect.scopes()))
        self.assertEqual(layer.scope.name, _call.side_effect.latest().name)
        _build.assert_called_once()
        self.assertEqual(1, len(_build.side_effect.scopes()))
        self.assertEqual(layer.scope.name, _build.side_effect.latest().name)

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
        self.assertEqual(layer.name, layer.scope.name)
        self.assertEqual(1, len(_build.side_effect.scopes()))
        self.assertEqual(layer.scope.name, _build.side_effect.latest().name)

        inp = object()
        out = layer.call(inp)
        self.assertEqual(inp, out)
        _call.assert_called_once()
        self.assertEqual(1, len(_call.side_effect.scopes()))
        self.assertEqual(layer.scope.name, _call.side_effect.latest().name)
        _build.assert_called_once()
        self.assertEqual(1, len(_build.side_effect.scopes()))
        self.assertEqual(layer.scope.name, _build.side_effect.latest().name)

        inp = object()
        out = layer.call(inp)
        self.assertEqual(inp, out)
        self.assertEqual(2, _call.call_count)
        self.assertEqual(2, len(_call.side_effect.scopes()))
        self.assertEqual(layer.scope.name, _call.side_effect.latest().name)
        _build.assert_called_once()
        self.assertEqual(1, len(_build.side_effect.scopes()))
        self.assertEqual(layer.scope.name, _build.side_effect.latest().name)

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
                layer.call(arg)
        tf.reset_default_graph()
        self._test_reuse(_init_scope)

        def _build_scope(scope, *args):
            layer = Layer()
            layer.build(scope=scope)
            for arg in args:
                layer.call(arg)
        tf.reset_default_graph()
        self._test_reuse(_build_scope)

        def _apply_scope(scope, *args):
            layer = Layer()
            for i, arg in enumerate(args):
                if i == 0:
                    layer.call(arg, scope=scope)
                else:
                    layer.call(arg)
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
            self.assertEqual(1, _build.call_count)
            self.assertEqual(len(inputs), _call.call_count)
            self.assertEqual(2, len(variables))
            actual = sorted([v.op.name for v in variables])
            expected = sorted([build_var_op_name, call_var_op_name])
            for act, exp in zip(actual, expected):
                self.assertEqual(act, exp)

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
            self.assertTrue(str(context.exception).startswith(
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
                layer.call(object())
            self.assertTrue(isinstance(context.exception, ValueError))
            self.assertTrue(str(context.exception).startswith(
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
            self.assertTrue(str(context.exception).startswith(
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
                layer.call(object())
            self.assertTrue(isinstance(context.exception, ValueError))
            self.assertTrue(str(context.exception).startswith(
                'Variable Scope/Variable already exists'))

    @mock.patch.object(Layer, '_call')
    @mock.patch.object(Layer, '_build')
    def test_change_scope(self, _build, _call):
        """Once set, the scope doesn't change.

        NOTA BENE: you should see some warning in the logs.
        """
        scope = utils.as_scope('Scope01')
        layer = Layer(scope=scope)
        self.assertEqual(scope.name, layer.scope.name)

        layer.build(scope='Scope02')
        self.assertEqual(scope.name, layer.scope.name)

        inp = object()
        _ = layer(inp, scope='Scope03')
        self.assertEqual(scope.name, layer.scope.name)
        self.assertEqual(1, _call.call_count)
        args, _ = _call.call_args
        self.assertEqual(inp, args[0])

        inp = object()
        _ = layer(inp, scope='Scope04')
        self.assertEqual(scope.name, layer.scope.name)
        self.assertEqual(2, _call.call_count)
        args, _ = _call.call_args
        self.assertEqual(inp, args[0])

        self.assertEqual(1, _build.call_count)


class BahdanauAttentionTest(tf.test.TestCase):
    """Test case for the liteflow.layers.BahdanauAttention class."""

    _SEED = 23

    def setUp(self):
        tf.reset_default_graph()
        tf.set_random_seed(seed=self._SEED)
        np.random.seed(seed=self._SEED)  # pylint: disable=I0011,E1101

    def _get_names(self, key):  # pylint: disable=I0011,R0201
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
        self.assertEqual(0, len(self._get_names(
            tf.GraphKeys.TRAINABLE_VARIABLES)))
        initializer = tf.constant_initializer(0.1)
        with tf.variable_scope('Scope', initializer=initializer) as scope:
            states_ = tf.placeholder(
                dtype=tf.float32,
                shape=[None, None, state_size],
                name='S')
            queries_ = [tf.placeholder(dtype=tf.float32, shape=[None, query_size], name='q01'),
                        tf.placeholder(dtype=tf.float32, shape=[None, query_size], name='q02')]
            attention = layers.BahdanauAttention(
                states_, attention_size, trainable=trainable, scope=scope)
            self.assertEqual(trainable, attention.trainable)
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
        np.random.seed(seed=self._SEED)  # pylint: disable=I0011,E1101

    def test_base(self):
        """Basic usage of the `liteflow.layers.PointingSoftmax` class."""

        query = tf.constant(
            [[0.05, 0.05, 0.05], [0.07, 0.07, 0.07]], dtype=tf.float32)

        states_np = np.asarray(
            [[[0.01, 0.01, 0.01], [0.02, 0.02, 0.02], [0.03, 0.03, 0.03], [0.04, 0.04, 0.04]],
             [[0.1, 0.1, 0.1], [0.2, 0.2, 0.2], [23.0, 23.0, 23.0], [23.0, 23.0, 23.0]]])
        states = tf.placeholder(dtype=tf.float32, shape=[None, None, 3])
        lengths = tf.constant([4, 2], dtype=tf.int32)

        activations = tf.constant(
            [[1, 1, 1, 1], [1, 2, 10, 10]], dtype=tf.float32)

        exp_weights = np.asarray(
            [[0.25, 0.25, 0.25, 0.25], [0.2689414, 0.7310586, 0.0, 0.0]])
        exp_context = np.asarray(
            [[0.025, 0.025, 0.025], [0.17310574, 0.17310574, 0.17310574]])

        attention = mock.Mock()
        attention.states = states
        attention.built = False
        attention.apply.side_effect = [activations, activations]

        layer = layers.PointingSoftmax(attention, lengths)
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
        attention.built = False
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

        layer = layers.PointingSoftmaxOutput(
            emission_size=10,
            decoder_out_size=7,
            attention_size=4)
        self.assertFalse(layer.built)
        layer.build()
        self.assertTrue(layer.built)

    def test_base(self):
        """Base test for the PointingSoftmaxOutput layer."""
        decoder_out_size = 3
        attention_size = 4
        emission_size = 7

        decoder_out = tf.constant(
            [[1, 1, 1], [2, 2, 2]],
            dtype=tf.float32)
        pointing_scores = tf.constant(
            [[0.1, 0.1, 0.1, 0.2, 0.5],
             [0.2, 0.1, 0.5, 0.1, 0.1]],
            dtype=tf.float32)
        attention_context = tf.constant(
            [[3, 3, 3, 3], [4, 4, 4, 4]],
            dtype=tf.float32)
        initializer = tf.constant_initializer(value=0.1)

        with tf.variable_scope('', initializer=initializer):
            layer = layers.PointingSoftmaxOutput(
                emission_size=emission_size,
                decoder_out_size=decoder_out_size,
                attention_size=attention_size)
            output = layer.call(
                decoder_out=decoder_out,
                pointing_scores=pointing_scores,
                attention_context=attention_context)

        exp_output = np.asarray(
            [[0.49811914, 0.49811914, 0.49811914, 0.49811914, 0.49811914,
              0.49811914, 0.49811914, 0.01679816, 0.01679816, 0.01679816,
              0.03359632, 0.08399081],
             [0.60730052, 0.60730052, 0.60730052, 0.60730052, 0.60730052,
              0.60730052, 0.60730052, 0.01822459, 0.00911230, 0.04556148,
              0.00911230, 0.0091123]],
            dtype=np.float32)  # pylint: disable=I0011,E1101

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            act_output = sess.run(output)
        self.assertAllClose(exp_output, act_output)

    def test_zero_output(self):
        """Tests the zero-output."""
        decoder_out_size = 3
        attention_size = 4
        emission_size = 7
        batch_size = 2
        pointing_size = 10
        state_size = 5
        states = tf.placeholder(dtype=tf.float32, shape=[None, None, None])
        batch_size_tensor = tf.shape(states)[0]
        pointing_size_tensor = tf.shape(states)[1]
        layer = layers.PointingSoftmaxOutput(emission_size, decoder_out_size, attention_size)
        zero_output = layer.zero_output(batch_size_tensor, pointing_size_tensor)

        data = np.ones((batch_size, pointing_size, state_size))
        exp_shape = (batch_size, emission_size + pointing_size)
        exp_zero_output = np.zeros(exp_shape)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            act_zero_output = sess.run(zero_output, {states: data})
        self.assertAllEqual(exp_zero_output, act_zero_output)


class TestPointingDecoder(tf.test.TestCase):
    """Test case for the PointingDecoder class."""

    _SEED = 23

    def setUp(self):
        tf.reset_default_graph()
        tf.set_random_seed(self._SEED)
        np.random.seed(seed=self._SEED)  # pylint: disable=I0011,E1101

    def test_build_and_init(self):
        """Test the .build moethod and the default init tensors."""

        emit_size = 3
        batch_size = 2
        timesteps = 11
        state_size = 4
        cell_output_size = 5
        cell_state_size = 2 * cell_output_size

        states = tf.placeholder(dtype=tf.float32, shape=[None, None, None])
        zero_output = tf.zeros(tf.stack([
            utils.get_dimension(states, 0),
            emit_size + utils.get_dimension(states, 1)]))
        cell_zero_state = tf.zeros(tf.stack([utils.get_dimension(states, 0), cell_state_size]))

        decoder_cell = mock.Mock()
        decoder_cell.output_size = cell_output_size
        decoder_cell.zero_state.side_effect = [cell_zero_state]

        pointing_softmax = mock.Mock()
        pointing_softmax.attention.states = states

        pointing_softmax_output = mock.Mock()
        pointing_softmax_output.zero_output.side_effect = [zero_output]

        layer = layers.PointingDecoder(decoder_cell, pointing_softmax, pointing_softmax_output)

        self.assertIsNone(layer.emit_out_init)
        self.assertIsNone(layer.cell_out_init)
        self.assertIsNone(layer.cell_state_init)

        layer.build()
        self.assertIsNotNone(layer.emit_out_init)
        self.assertIsNotNone(layer.cell_out_init)
        self.assertIsNotNone(layer.cell_state_init)

        data = np.ones((batch_size, timesteps, state_size))
        exp_output_init = np.zeros((batch_size, emit_size + timesteps))
        exp_cell_out_init = np.zeros((batch_size, cell_output_size))
        exp_cell_state_init = np.zeros((batch_size, cell_state_size))

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            act_output_init = sess.run(layer.emit_out_init, {states: data})
            act_cell_out_init = sess.run(layer.cell_out_init, {states: data})
            act_cell_state_init = sess.run(layer.cell_state_init, {states: data})

        self.assertAllClose(act_output_init, exp_output_init)
        self.assertAllClose(act_cell_out_init, exp_cell_out_init)
        self.assertAllClose(act_cell_state_init, exp_cell_state_init)

    def test_loop_fn_init(self):
        """Test the initialization of the loop function."""

        emit_size = 3
        batch_size = 2
        timesteps = 11
        state_size = 4
        cell_output_size = 5
        cell_state_size = 2 * cell_output_size
        attention_size = 7

        states = tf.placeholder(dtype=tf.float32, shape=[None, None, None])
        sequence_length = tf.placeholder(dtype=tf.int32, shape=[None])

        # TODO(petrux): zero_output must become output_init with random values.
        batch_dim = utils.get_dimension(states, 0)
        timesteps_dim = utils.get_dimension(states, 1)
        zero_output = tf.zeros(tf.stack([batch_dim, emit_size + timesteps_dim]))
        cell_zero_state = tf.zeros(tf.stack([batch_dim, cell_state_size]))
        time = tf.constant(0, dtype=tf.int32)

        pointing_scores = tf.random_normal(shape=[batch_dim, timesteps_dim])
        attention_context = tf.random_normal(shape=[batch_dim, attention_size])

        decoder_cell = mock.Mock()
        decoder_cell.output_size = cell_output_size
        decoder_cell.zero_state.side_effect = [cell_zero_state]

        pointing_softmax = mock.Mock()
        pointing_softmax.attention.states = states
        pointing_softmax.sequence_length = sequence_length
        pointing_softmax.side_effect = [(pointing_scores, attention_context)]

        pointing_softmax_output = mock.Mock()
        pointing_softmax_output.zero_output.side_effect = [zero_output]

        layer = layers.PointingDecoder(decoder_cell, pointing_softmax, pointing_softmax_output)
        layer.build()

        results = layer._loop_fn(time, None, None, (None, None))  # pylint: disable=I0011,W0212
        elements_finished = results[0]
        next_cell_input = results[1]
        next_cell_state = results[2]
        emit_output = results[3]
        next_pointing_scores, next_attention_context = results[4]

        # Assertion of programatically returned tensors.
        self.assertEqual(next_cell_state, cell_zero_state)
        self.assertEqual(emit_output, zero_output)
        self.assertEqual(next_pointing_scores, pointing_scores)
        self.assertEqual(next_attention_context, attention_context)

        # Data for feeding placeholders and expected values.
        data = np.ones((batch_size, timesteps, state_size))
        lengths = [timesteps] * batch_size
        exp_elements_finished = np.asarray([False] * batch_size)

        fetches = [elements_finished,
                   next_cell_input,
                   layer.cell_out_init,
                   attention_context,
                   zero_output]
        feed_dict = {
            states: data,
            sequence_length: lengths}
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            actual = sess.run(fetches, feed_dict)
        act_elements_finished = actual[0]
        act_next_cell_input = actual[1]
        act_cell_out_init = actual[2]
        act_attention_context = actual[3]
        act_zero_output = actual[4]
        act_next_cell_input_rebuilt = np.concatenate(
            (act_cell_out_init, act_attention_context, act_zero_output), axis=1)

        self.assertAllEqual(exp_elements_finished, act_elements_finished)
        self.assertAllEqual(act_next_cell_input, act_next_cell_input_rebuilt)

    def test_loop_fn_step(self):
        """Test a regular step of the loop function."""

        emit_size = 3
        batch_size = 2
        timesteps = 11
        state_size = 4
        cell_output_size = 5
        cell_state_size = 2 * cell_output_size
        attention_size = 7
        current_time = 5
        lengths = [3, 10]

        # placeholders and dynamic shapes.
        states = tf.placeholder(dtype=tf.float32, shape=[None, None, None])
        sequence_length = tf.placeholder(dtype=tf.int32, shape=[None])
        batch_dim = utils.get_dimension(states, 0)
        timesteps_dim = utils.get_dimension(states, 1)

        # init values.
        emit_out_init = tf.zeros(tf.stack([batch_dim, emit_size + timesteps_dim]))
        cell_state_init = tf.random_normal(shape=tf.stack([batch_dim, cell_state_size]))

        # input tensors.
        time = tf.constant(current_time, dtype=tf.int32)
        cell_output = tf.random_normal(shape=tf.stack([batch_dim, cell_output_size]))
        cell_state = tf.random_normal(shape=tf.stack([batch_dim, cell_state_size]))
        pointing_scores = tf.random_normal(shape=[batch_dim, timesteps_dim])
        attention_context = tf.random_normal(shape=[batch_dim, attention_size])
        loop_state = (pointing_scores, attention_context)

        # output/next step tensors.
        emit_out = tf.ones(tf.stack([batch_dim, emit_size + timesteps_dim]))
        next_pointing_scores = tf.random_normal(shape=[batch_dim, timesteps_dim])
        next_attention_context = tf.random_normal(shape=[batch_dim, attention_size])
        next_loop_state = (next_pointing_scores, next_attention_context)

        decoder_cell = mock.Mock()
        decoder_cell.output_size = cell_output_size
        decoder_cell.zero_state.side_effect = [cell_state_init]

        pointing_softmax = mock.Mock()
        pointing_softmax.attention.states = states
        pointing_softmax.sequence_length = sequence_length
        pointing_softmax.side_effect = [next_loop_state]

        pointing_softmax_output = mock.Mock()
        pointing_softmax_output.zero_output.side_effect = [emit_out_init]
        pointing_softmax_output.side_effect = [emit_out]

        layer = layers.PointingDecoder(decoder_cell, pointing_softmax, pointing_softmax_output)
        layer.build()

        results = layer._loop_fn(time, cell_output, cell_state, loop_state)  # pylint: disable=I0011,W0212

        # Assertions on programatically returned tensors.
        self.assertEqual(cell_state, results[2])
        self.assertEqual(emit_out, results[3])
        self.assertEqual(next_loop_state, results[4])

        elements_finished = results[0]
        next_cell_input = results[1]
        data = np.ones((batch_size, timesteps, state_size))
        exp_elements_finished = np.asarray([current_time >= l for l in lengths])
        fetches = [elements_finished,
                   next_cell_input,
                   cell_output,
                   next_attention_context,
                   emit_out]
        feed_dict = {
            states: data,
            sequence_length: lengths
        }
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            actual = sess.run(fetches, feed_dict)

        act_elements_finished = actual[0]
        act_next_cell_input = actual[1]
        act_cell_out = actual[2]
        act_attention_context = actual[3]
        act_emit_out = actual[4]
        act_next_cell_input_rebuilt = np.concatenate([
            act_cell_out, act_attention_context, act_emit_out], axis=1)

        self.assertAllEqual(exp_elements_finished, act_elements_finished)
        self.assertAllEqual(act_next_cell_input, act_next_cell_input_rebuilt)


if __name__ == '__main__':
    tf.test.main()
