"""Test module for liteflow.layers module."""

# Disable pylint warning about too many statements
# and local variables since we are dealing with tests.
# pylint: disable=R0914, R0915

import mock
import numpy as np
import tensorflow as tf

from liteflow import layers, utils


class BahdanauAttentionTest(tf.test.TestCase):
    """Test case for the liteflow.layers.BahdanauAttention class."""

    _SEED = 23

    def setUp(self):
        tf.reset_default_graph()
        tf.set_random_seed(seed=self._SEED)
        np.random.seed(seed=self._SEED)  # pylint: disable=I0011,E1101

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
        self.setUp()
        self._test_base(trainable=True)
        self.setUp()
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

        self.assertTrue(len(tf.global_variables()) == 0)
        self.assertTrue(len(tf.trainable_variables()) == 0)

        initializer = tf.constant_initializer(0.1)
        with tf.variable_scope('Scope', initializer=initializer) as scope:
            states_ = tf.placeholder(
                dtype=tf.float32,
                shape=[None, None, state_size],
                name='States')
            queries_ = [tf.placeholder(dtype=tf.float32, shape=[None, query_size], name='q01'),
                        tf.placeholder(dtype=tf.float32, shape=[None, query_size], name='q02')]
            attention = layers.BahdanauAttention(
                states_, attention_size, trainable=trainable)
            self.assertEqual(trainable, attention.trainable)

            scores_ = []
            scores_.append(attention(queries_[0], scope=scope))
            self.assertEqual(scope.name, attention.scope)
            self.assertTrue(len(tf.global_variables()) > 0)
            if attention.trainable:
                self.assertTrue(tf.trainable_variables())
            variables = set([var.op.name for var in tf.global_variables()])
            scores_.append(attention(queries_[1], scope=scope))
            self.assertEqual(len(tf.global_variables()), len(variables))
            for var in tf.global_variables():
                self.assertIn(var.op.name, variables)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            results = sess.run(scores_, {
                states_: states,
                queries_[0]: queries[0],
                queries_[1]: queries[1]
            })

        for act, exp in zip(results, scores):
            self.assertAllClose(act, exp, rtol=1e-4, atol=1e-4)


class TestLocationSoftmax(tf.test.TestCase):
    """Test case for the `liteflow.layers.PointingSoftmax` class."""

    _SEED = 23

    def setUp(self):
        tf.reset_default_graph()
        tf.set_random_seed(seed=self._SEED)
        np.random.seed(seed=self._SEED)  # pylint: disable=I0011,E1101

    def test_base(self):
        """Basic usage of the `liteflow.layers.PointingSoftmax` class."""

        query = tf.constant([[0.05, 0.05, 0.05], [0.07, 0.07, 0.07]], dtype=tf.float32)
        states_np = np.asarray(
            [[[0.01, 0.01, 0.01], [0.02, 0.02, 0.02], [0.03, 0.03, 0.03], [0.04, 0.04, 0.04]],
             [[0.1, 0.1, 0.1], [0.2, 0.2, 0.2], [23.0, 23.0, 23.0], [23.0, 23.0, 23.0]]])
        states = tf.placeholder(dtype=tf.float32, shape=[None, None, 3])
        lengths = tf.constant([4, 2], dtype=tf.int32)

        activations = tf.constant([[1, 1, 1, 1], [1, 2, 10, 10]], dtype=tf.float32)
        exp_location = np.asarray([[0.25, 0.25, 0.25, 0.25], [0.2689414, 0.7310586, 0.0, 0.0]])
        exp_context = np.asarray([[0.025, 0.025, 0.025], [0.17310574, 0.17310574, 0.17310574]])

        attention = mock.Mock()
        attention.states = states
        attention.side_effect = [activations, activations]

        layer = layers.LocationSoftmax(attention, lengths)
        location, context = layer(query)
        _, _ = layer(query)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            feed_dict = {
                states: states_np
            }
            act_location, act_context = sess.run(
                [location, context], feed_dict=feed_dict)

        self.assertAllClose(act_location, exp_location)
        self.assertAllClose(act_context, exp_context)


class TestPointingSoftmaxOutput(tf.test.TestCase):
    """Test case for the PointingSoftmaxOutput layer."""

    def test_base(self):
        """Base test for the PointingSoftmaxOutput layer."""
        batch_size = 2
        decoder_out_size = 3
        state_size = 4
        shortlist_size = 7
        timesteps = 5
        location_size = timesteps
        output_size = shortlist_size + location_size

        decoder_out = tf.constant(
            [[1, 1, 1], [2, 2, 2]],
            dtype=tf.float32)  # [batch_size, decoder_out_size]
        location_softmax = tf.constant(
            [[0.1, 0.1, 0.1, 0.2, 0.5],
             [0.2, 0.1, 0.5, 0.1, 0.1]],
            dtype=tf.float32)  # [batch_size, location_size]
        attention_context = tf.constant(
            [[3, 3, 3, 3], [4, 4, 4, 4]],
            dtype=tf.float32)  # [batch_size, state_size]
        initializer = tf.constant_initializer(value=0.1)

        with tf.variable_scope('', initializer=initializer):
            layer = layers.PointingSoftmaxOutput(
                shortlist_size=shortlist_size,
                decoder_out_size=decoder_out_size,
                state_size=state_size)
            output = layer(
                decoder_out=decoder_out,
                location_softmax=location_softmax,
                attention_context=attention_context)

        # the expected output has shape [batch, output_size] where
        # output size is given by the sum:
        # output_size = emission_size + pointing_size
        exp_output = np.asarray(
            [[0.49811914, 0.49811914, 0.49811914, 0.49811914, 0.49811914,
              0.49811914, 0.49811914, 0.01679816, 0.01679816, 0.01679816,
              0.03359632, 0.08399081],
             [0.60730052, 0.60730052, 0.60730052, 0.60730052, 0.60730052,
              0.60730052, 0.60730052, 0.01822459, 0.00911230, 0.04556148,
              0.00911230, 0.0091123]],
            dtype=np.float32)  # pylint: disable=I0011,E1101
        exp_output_shape = (batch_size, output_size)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            act_output = sess.run(output)
        self.assertEqual(exp_output_shape, exp_output.shape)
        self.assertAllClose(exp_output, act_output)

    def test_zero_output(self):
        """Tests the zero-output."""
        batch_size = 2
        decoder_out_size = 3
        shortlist_size = 7
        timesteps = 7
        location_size = timesteps
        state_size = 5
        output_size = shortlist_size + location_size

        states = tf.placeholder(dtype=tf.float32, shape=[None, None, None])
        batch_dim = tf.shape(states)[0]
        location_dim = tf.shape(states)[1]
        layer = layers.PointingSoftmaxOutput(
            shortlist_size=shortlist_size,
            decoder_out_size=decoder_out_size,
            state_size=state_size)
        zero_output = layer.zero_output(batch_dim, location_dim)

        data = np.ones((batch_size, location_size, state_size))
        exp_shape = (batch_size, output_size)
        exp_zero_output = np.zeros(exp_shape)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            act_zero_output = sess.run(zero_output, {states: data})
        self.assertAllEqual(exp_zero_output, act_zero_output)


class TestPointingDecoder(tf.test.TestCase):
    """Test case for the PointingDecoder class."""

    import unittest

    _SEED = 23

    def setUp(self):
        tf.reset_default_graph()
        tf.set_random_seed(self._SEED)
        np.random.seed(seed=self._SEED)  # pylint: disable=I0011,E1101

    def test_step(self):
        """Test the single step."""

        batch_size = 2
        shortlist_size = 3
        timesteps = 11
        state_size = 4
        cell_output_size = 5
        cell_state_size = 2 * cell_output_size

        states = tf.placeholder(dtype=tf.float32, shape=[None, None, None])
        batch_dim = utils.get_dimension(states, 0)
        location_dim = utils.get_dimension(states, 1)
        out_sequence_length = location_dim * tf.ones(dtype=tf.int32, shape=[batch_dim])

        decoder_cell = mock.Mock()
        cell_out = 3 * tf.ones(shape=tf.stack([batch_dim, cell_output_size]))
        cell_state = 4 * tf.ones(shape=tf.stack([batch_dim, cell_state_size]))
        decoder_cell.side_effect = [(cell_out, cell_state)]

        location_softmax = mock.Mock()

        pointing_softmax_output = mock.Mock()
        emit_out_dim = shortlist_size + location_dim
        emit_out = 23 * tf.ones(tf.stack([batch_dim, emit_out_dim]))
        pointing_softmax_output.side_effect = [emit_out]

        layer = layers.PointingDecoder(
            decoder_cell=decoder_cell,
            out_sequence_length=out_sequence_length,
            location_softmax=location_softmax,
            pointing_softmax_output=pointing_softmax_output)

        prev_cell_out = 1 * tf.ones(shape=tf.stack([batch_dim, cell_output_size]))
        prev_cell_state = 2 * tf.ones(shape=tf.stack([batch_dim, cell_state_size]))
        prev_emit_out = 22 * tf.ones(tf.stack([batch_dim, emit_out_dim]))
        location = 10 * tf.ones(shape=tf.stack([batch_dim, location_dim]))
        attention = 11 * tf.ones(shape=tf.stack([batch_dim, state_size]))

        # pylint: disable=I0011,W0212
        act_cell_out_t, act_cell_state_t, act_emit_out_t =\
            layer._step(prev_cell_out, prev_cell_state, prev_emit_out, location, attention)

        # OUTPUT TENSORS ASSERTIONS.
        self.assertEqual(act_cell_out_t, cell_out)
        self.assertEqual(act_cell_state_t, cell_state)
        self.assertEqual(act_emit_out_t, emit_out)

        # DECODER CELL ASSERTIONS.
        # Since the first argument of the cell function is dynamically built
        # concatenating different tensor, we can do a tensor-based assertion
        # only on the second argument, i.e. the cell state.
        decoder_cell.assert_called_once()
        args, _ = tuple(decoder_cell.call_args)
        act_cell_in_t, act_prev_cell_state_t = args
        exp_cell_in_t = tf.concat([prev_cell_out, attention, prev_emit_out], axis=1)
        self.assertEqual(act_prev_cell_state_t, prev_cell_state)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            exp_cell_in, act_cell_in = sess.run(
                fetches=[act_cell_in_t, exp_cell_in_t],
                feed_dict={
                    states: np.ones((batch_size, timesteps, state_size))
                })
            self.assertAllClose(exp_cell_in, act_cell_in)

        # POINTING SOFTMAX OUTPUT ASSERTIONS.
        pointing_softmax_output.assert_called_once_with(cell_out, location, attention)

    @mock.patch.object(layers.PointingDecoder, '_step')
    def test_body(self, _step=None):
        """Test the loop body"""

        batch_size = 2
        shortlist_size = 3
        timesteps = 11
        location_size = timesteps
        state_size = 4
        cell_output_size = 5
        cell_state_size = 2 * cell_output_size
        emit_out_size = shortlist_size + location_size

        states = tf.placeholder(dtype=tf.float32, shape=[None, None, None])
        batch_dim = utils.get_dimension(states, 0)
        location_dim = utils.get_dimension(states, 1)
        emit_out_dim = shortlist_size + location_dim
        out_sequence_length = location_dim * tf.ones(dtype=tf.int32, shape=[batch_dim])

        decoder_cell = mock.Mock()
        pointing_softmax_output = mock.Mock()

        emit_out_feedback_fit = mock.Mock()
        def _feedback(value):
            return value
        emit_out_feedback_fit.side_effect = _feedback

        location_softmax = mock.Mock()
        next_location = 20 * tf.ones(shape=tf.stack([batch_dim, location_dim]))
        next_attention = 21 * tf.ones(shape=tf.stack([batch_dim, state_size]))
        location_softmax.side_effect = [(next_location, next_attention)]

        cell_out = 3 * tf.ones(shape=tf.stack([batch_dim, cell_output_size]))
        cell_state = 4 * tf.ones(shape=tf.stack([batch_dim, cell_state_size]))
        emit_out = 23 * tf.ones(tf.stack([batch_dim, emit_out_dim]))
        _step.side_effect = [(cell_out, cell_state, emit_out)]

        layer = layers.PointingDecoder(
            decoder_cell=decoder_cell,
            out_sequence_length=out_sequence_length,
            location_softmax=location_softmax,
            pointing_softmax_output=pointing_softmax_output,
            emit_out_feedback_fit=emit_out_feedback_fit)

        time = tf.constant(0, dtype=tf.int32)
        prev_cell_out = 1 * tf.ones(shape=tf.stack([batch_dim, cell_output_size]))
        prev_cell_state = 2 * tf.ones(shape=tf.stack([batch_dim, cell_state_size]))
        prev_emit_out = 22 * tf.ones(tf.stack([batch_dim, emit_out_dim]))
        location = 10 * tf.ones(shape=tf.stack([batch_dim, location_dim]))
        attention = 11 * tf.ones(shape=tf.stack([batch_dim, state_size]))
        emit_ta = tf.TensorArray(size=0, dynamic_size=True, dtype=tf.float32)

        # pylint: disable=I0011,W0212
        result = layer._body(time, prev_cell_out, prev_cell_state,
                             prev_emit_out, location, attention, emit_ta)
        self.assertEqual(7, len(result))
        self.assertEqual(result[0], time)
        self.assertEqual(result[1], cell_out)
        self.assertEqual(result[2], cell_state)
        self.assertEqual(result[3], emit_out)
        self.assertEqual(result[4], next_location)
        self.assertEqual(result[5], next_attention)

        # ASSERTIONS ON EMIT OUT
        act_emit_t = result[6].stack()
        self.assertEqual(3, act_emit_t.shape.ndims)
        exp_emit_length = 1
        exp_emit_first_t = emit_out
        exp_emit_shape = (exp_emit_length, batch_size, emit_out_size)
        feed = {states: np.ones((batch_size, timesteps, state_size))}
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            act_emit = sess.run(act_emit_t, feed)
            self.assertEqual(exp_emit_shape, act_emit.shape)
            act_emit_first = act_emit[0]
            exp_emit_first = sess.run(exp_emit_first_t, feed)
            self.assertAllEqual(exp_emit_first, act_emit_first)

        # ASSERTION ON FEEDBACK FIT.
        emit_out_feedback_fit.assert_called_once_with(emit_out)

        # ASSERTIONS ON _step.
        _step.assert_called_once_with(
            prev_cell_out, prev_cell_state, prev_emit_out, location, attention)

        # ASSERTIONS ON LOCATION SOFTMAX.
        location_softmax.assert_called_once()
        location_softmax_args = tuple(location_softmax.call_args)[0]
        act_query_t = location_softmax_args[0]
        exp_query_t = tf.concat([cell_out, emit_out], axis=1)
        feed = {states: np.ones((batch_size, timesteps, state_size))}
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            exp_query, act_query = sess.run([exp_query_t, act_query_t], feed)
            self.assertAllEqual(exp_query, act_query)


class _TestSmoke(tf.test.TestCase):
    """Smoke test for pointing decoder."""

    import unittest

    _SEED = 23

    def setUp(self):
        tf.reset_default_graph()
        tf.set_random_seed(self._SEED)
        np.random.seed(seed=self._SEED)  # pylint: disable=I0011,E1101

    @unittest.skip('Under refactoring')
    def test_smoke(self):
        """Build a pointer decoder and test that it works."""
        batch_size = 2
        timesteps = 10
        state_size = 4
        attention_inner_size = 7
        shortlist_size = 3
        # emit_out_feedback_size = 17
        decoder_out_size = 5

        attention_states = tf.placeholder(tf.float32, shape=[None, None, state_size])
        attention_sequence_length = tf.placeholder(tf.int32, [None])
        output_sequence_length = tf.placeholder(tf.int32, [None])
        decoder_cell = tf.contrib.rnn.GRUCell(decoder_out_size)

        decoder = layers.pointing_decoder(
            attention_states=attention_states,
            attention_inner_size=attention_inner_size,
            decoder_cell=decoder_cell,
            shortlist_size=shortlist_size,
            attention_sequence_length=attention_sequence_length,
            output_sequence_length=output_sequence_length,
            emit_out_feedback_size=None,  # emit_out_feedback_size,
            parallel_iterations=None,
            swap_memory=False,
            trainable=True)
        output = decoder()
        print(output)

        act_attention_states = np.ones((batch_size, timesteps, state_size))
        act_attention_sequence_lengths = [6, 8]
        act_output_sequence_length = [5, 7]

        feed_dict = {
            attention_states: act_attention_states,
            attention_sequence_length: act_attention_sequence_lengths,
            output_sequence_length: act_output_sequence_length
        }

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            print(sess.run(output, feed_dict))

if __name__ == '__main__':
    tf.test.main()
