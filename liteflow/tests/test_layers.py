"""Test module for liteflow.layers module."""

# Disable pylint warning about too many statements
# and local variables since we are dealing with tests.
# pylint: disable=R0914, R0915

import mock
import numpy as np
import tensorflow as tf

from liteflow import layers
from liteflow import utils

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


class TestTerminationHelper(tf.test.TestCase):
    """Test case for the TerminationHelper class."""

    def test_init_errors(self):
        """Test initialization errors."""
        lengths0D = tf.placeholder(dtype=tf.int32, shape=[])  # pylint: disable=C0103,I0011
        lengths1D = tf.placeholder(dtype=tf.int32, shape=[None])  # pylint: disable=C0103,I0011
        self.assertRaises(ValueError, layers.TerminationHelper, None)
        self.assertRaises(ValueError, layers.TerminationHelper, None, 23)
        layers.TerminationHelper(lengths0D, None)
        layers.TerminationHelper(lengths1D, None)

    def test_EOS(self):  # pylint: disable=C0103,I0011
        """Test that an EOS in the output triggers a finished flag."""
        time = tf.convert_to_tensor(5, dtype=tf.int32)
        maxlen = tf.convert_to_tensor(10, dtype=tf.int32)
        output = tf.constant([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=tf.float32)
        eos = layers.TerminationHelper.EOS
        finished = layers.TerminationHelper(maxlen, EOS=eos).finished(time, output)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            act_finished = sess.run(finished)
            
        exp_finished = [True, False]
        self.assertAllEqual(exp_finished, act_finished)

    def test_time(self):
        """Test that a `time` over the `length` triggers a finished flag."""
        tf.set_random_seed(23)
        time = tf.convert_to_tensor(5, dtype=tf.int32)
        lengths = tf.constant([4, 5, 6, 7])
        output = tf.random_normal([4, 10, 3], dtype=tf.float32)
        finished = layers.TerminationHelper(lengths).finished(time, output)
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            act_finished = sess.run(finished)

        # NOTA BENE: we have set that
        # time = 5
        # lengths = [4, 5, 6, 7]
        # 
        # Since the time is 0-based, having time=5 means that
        # we have alread scanned through 5 elements, so only
        # the last sequence in the batch is ongoing.
        exp_finished = [True, True, True, False]
        self.assertAllEqual(exp_finished, act_finished)


class TestDynamicDecoder(tf.test.TestCase):
    """Test case for the DynamicDecoder class."""

    def test_output(self):
        """Test the DynamicDecoder.output() method."""

        helper = mock.Mock()
        decoder = mock.Mock()
        zero_output = tf.constant([[0, 0, 0], [0, 0, 0]], dtype=tf.float32)
        decoder.zero_output.side_effect = [zero_output]

        output = tf.constant([[23, 23, 23], [23, 23, 23]], dtype=tf.float32)
        finished = tf.constant([True, False], dtype=tf.bool)

        dyndec = layers.DynamicDecoder(decoder, helper)
        act_output_t = dyndec.output(output, finished)
        exp_output = np.asarray([[0, 0, 0], [23, 23, 23]], dtype=np.float32) # pylint: disable=I0011,E1101

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            act_output = sess.run(act_output_t)

        helper.finished.assert_not_called()
        decoder.zero_output.assert_called_once()
        self.assertAllEqual(exp_output, act_output)

    def test_condition(self):
        """Test the DynamicDecoder.condition() method."""

        helper = mock.Mock()
        decoder = mock.Mock()

        dyndec = layers.DynamicDecoder(decoder, helper)
        finished = [(tf.constant([True], dtype=tf.bool), False),
                    (tf.constant([False], dtype=tf.bool), True),
                    (tf.constant([True, False], dtype=tf.bool), True),
                    (tf.constant([True, True], dtype=tf.bool), False)]

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for tensor, expected in finished:
                actual = sess.run(dyndec.cond(None, None, None, tensor, None))
                self.assertEqual(expected, actual)

        helper.assert_not_called()
        decoder.assert_not_called()

    def test_body(self):
        """Test the DynamicDecoder.body() method."""
        time = tf.constant(2, dtype=tf.int32)
        inp = tf.constant([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
        state = tf.constant([[5.0, 5.0, 5.0], [6.0, 6.0, 6.0], [7.0, 7.0, 7.0], [8.0, 8.0, 8.0]])
        finished = tf.constant([False, False, False, True], dtype=tf.bool)
        output_ta = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

        # Fill the output tensor array with some dummy
        # values for items 0 and 1
        dummy_out = tf.constant([[-.1, -.1], [-.2, -.2], [-.3, -.3], [-.4, -.4]])
        output_ta = output_ta.write(0, dummy_out).write(1, dummy_out)

        dec_out = tf.constant([[10.0, 10.0], [20.0, 20.0], [30.0, 30.0], [17.0, 17.0]])
        next_inp = 7.0 * inp
        next_state = 11.0 * state
        next_finished = tf.constant([False, False, True, True], dtype=tf.bool)
        zero_output = tf.zeros(dtype=tf.float32, shape=dec_out.get_shape())

        decoder = mock.Mock()
        decoder.step.side_effect = [(dec_out, next_inp, next_state, next_finished)]
        decoder.zero_output.side_effect = [zero_output]

        helper_finished = tf.constant([True, False, False, False], dtype=tf.bool)
        helper = mock.Mock()
        helper.finished.side_effect = [helper_finished]

        # pylint: disable=I0011,E1101
        # NOTA BENE: the only masked element is the one that is masked in the beginning.
        output_exp = np.asarray([[10.0, 10.0], [20.0, 20.0], [30.0, 30.0], [0.0, 0.0]])
        next_finished_exp = np.asarray([True, False, True, True], np.bool)
        # pylint: enable=I0011,E1101

        dyndec = layers.DynamicDecoder(decoder, helper)
        next_time_t, next_inp_t, next_state_t, next_finished_t, next_output_ta =\
            dyndec.body(time, inp, state, finished, output_ta)
        outputs_t = next_output_ta.stack()  # still time major, easier to check.

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            next_inp_exp, next_state_exp, dummy_out_act =\
                sess.run([next_inp, next_state, dummy_out])
            next_time_act, next_inp_act, next_state_act, next_finished_act, outputs_act =\
                sess.run([next_time_t, next_inp_t, next_state_t, next_finished_t, outputs_t])

        # assertions on tensors.
        self.assertEqual(next_inp, next_inp_t)
        self.assertEqual(next_state, next_state_t)

        # assertion on mocks.
        decoder.step.assert_called_once_with(time, inp, state)
        decoder.zero_output.assert_called_once()
        helper.finished.assert_called_once_with(time, dec_out)

        # assertion on calculated values.
        self.assertEqual(3, next_time_act)
        self.assertAllEqual(next_inp_exp, next_inp_act)
        self.assertAllEqual(next_state_exp, next_state_act)
        self.assertAllEqual(next_finished_exp, next_finished_act)
        self.assertAllEqual(dummy_out_act, outputs_act[0])
        self.assertAllEqual(dummy_out_act, outputs_act[1])
        self.assertAllEqual(output_exp, outputs_act[-1])

    def test_decode_one_step(self):
        """Default test for the DynamicDecoder.decode() method."""

        init_value = [[.1, .1], [.2, .2], [.3, .3]]
        init_input = tf.constant(init_value)
        init_state = 2 * init_input
        next_input = 3 * init_input
        next_state = 4 * init_input
        output = 10 * init_input
        finished = tf.constant([False, False, False], dtype=tf.bool)
        zero_output = tf.zeros_like(output)

        decoder = mock.Mock()
        decoder.init_input.side_effect = [init_input]
        decoder.init_state.side_effect = [init_state]
        decoder.zero_output.side_effect = [zero_output]
        decoder.step.side_effect = [(output, next_input, next_state, finished)]

        helper = mock.Mock()
        helper.finished.side_effect = [tf.logical_not(finished)]  # exit from the loop!

        dyndec = layers.DynamicDecoder(decoder, helper)
        output_t, state_t = dyndec.decode()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            output_act, state_act = sess.run([output_t, state_t])

            # assertions on output.
            output_exp = 10 * np.transpose(np.asarray([init_value]), (1, 0, 2))
            self.assertAllClose(output_exp, output_act)
            state_exp = 4 * np.asarray(init_value)
            self.assertAllClose(state_exp, state_act)

            # mock assertions.
            # we cannot assert more than this since the while
            # loop makes all the ops non-fetchable.
            decoder.init_input.assert_called_once()
            decoder.init_state.assert_called_once()
            decoder.zero_output.assert_called_once()
            decoder.step.assert_called_once()
            helper.finished.assert_called_once()

    def test_iterations_step_by_step(self):
        """Test the number of iterations (step by step)."""

        # pylint: disable=C0103,I0011
        T, F = True, False
        bt = lambda *args: tf.convert_to_tensor(list(args), dtype=tf.bool)
        # pylint: enable=C0103,I0011

        init_value = [[.1, .1], [.2, .2], [.3, .3]]
        inp = tf.placeholder(tf.float32, shape=[None, None])
        state = 2 * inp
        # next_input = 3 * init_input
        # next_state = 4 * init_input
        zero_output = tf.zeros_like(inp)

        out01 = 10 * inp
        out02 = 20 * inp
        out03 = 30 * inp

        decoder = mock.Mock()
        decoder.init_input.side_effect = [inp]
        decoder.init_state.side_effect = [state]
        decoder.zero_output.return_value = zero_output
        decoder_finished_00 = bt(F, F, F)
        decoder_finished_01 = bt(F, F, T)
        decoder_finished_02 = bt(F, F, T)
        decoder.step.side_effect = [
            (out01, inp, state, decoder_finished_00),  # time=0
            (out02, inp, state, decoder_finished_01),  # time=1
            (out03, inp, state, decoder_finished_02)]  # time=2

        helper = mock.Mock()
        helper_finished_00 = bt(F, F, F)
        helper_finished_01 = bt(F, T, F)
        helper_finished_02 = bt(T, F, F)
        helper.finished.side_effect = [
            helper_finished_00,  # time=0
            helper_finished_01,  # time=1
            helper_finished_02]  # time=2

        # STEP BY STEP EVALUATION OF `finished` flags.
        from liteflow import utils
        dyndec = layers.DynamicDecoder(decoder, helper)

        time = tf.constant(0, dtype=tf.int32)
        finished = tf.tile([F], [utils.get_dimension(inp, 0)])
        output_ta = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

        # time=0
        next_finished_00_exp = [F, F, F]
        results_00 = dyndec.body(time, inp, state, finished, output_ta)
        time = results_00[0]
        finished = results_00[3]
        results_00[-1].stack()
        
        feed = {inp: init_value}
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            next_finished_00_act = sess.run(finished, feed)
            self.assertEqual(1, sess.run(time, feed))
            self.assertAllEqual(next_finished_00_exp, next_finished_00_act)

        # time=1
        cond_01_t = dyndec.cond(*results_00)
        cond_01_exp = True
        feed = {inp: init_value}
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            cond_01_act = sess.run(cond_01_t, feed)
            self.assertEqual(cond_01_exp, cond_01_act)

        next_finished_01_exp = [F, T, T]
        results_01 = dyndec.body(time, inp, state, finished, output_ta)
        time = results_01[0]
        finished = results_01[3]
        results_01[-1].stack()

        feed = {inp: init_value}
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            self.assertEqual(2, sess.run(time, feed))
            next_finished_01_act = sess.run(finished, feed)
            self.assertAllEqual(next_finished_01_exp, next_finished_01_act)

        # time=2
        cond_02_t = dyndec.cond(*results_01)
        cond_02_exp = True
        feed = {inp: init_value}
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            cond_02_act = sess.run(cond_02_t, feed)
            self.assertEqual(cond_02_exp, cond_02_act)

        next_finished_02_exp = [T, T, T]
        results_02 = dyndec.body(time, inp, state, finished, output_ta)
        time = results_02[0]
        finished = results_02[3]
        results_02[-1].stack()

        feed = {inp: init_value}
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            self.assertEqual(3, sess.run(time, feed))
            next_finished_02_act = sess.run(finished, feed)
            self.assertAllEqual(next_finished_02_exp, next_finished_02_act)

        # time=3
        cond_03_t = dyndec.cond(*results_02)
        cond_03_exp = False  # STOP!
        feed = {inp: init_value}
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            cond_03_act = sess.run(cond_03_t, feed)
            self.assertEqual(cond_03_exp, cond_03_act)

    def test_iterations(self):
        """Test the number of iterations."""

        lengths = tf.constant([1, 2, 3], dtype=tf.int32)
        def _helper_finished(time, _):
            return tf.greater_equal(time + 1, lengths)
        helper = mock.Mock()
        helper.finished.side_effect = _helper_finished

        batch_size = utils.get_dimension(lengths, 0)
        inp_size, state_size, output_size = 2, 5, 2

        decoder = mock.Mock()
        decoder.init_input.side_effect = lambda: tf.zeros([batch_size, inp_size])
        decoder.init_state.side_effect = lambda: tf.zeros([batch_size, state_size])
        decoder.zero_output.side_effect = lambda: tf.zeros([batch_size, output_size])
        decoder.step.side_effect = lambda t, i, s: ((i + 1), 3 * (i + 1), s, tf.tile([False], [batch_size]))

        output_exp = np.asarray(
            [[[1, 1], [0, 0], [0, 0]],
             [[1, 1], [4, 4], [0, 0]],
             [[1, 1], [4, 4], [13, 13]]],
            dtype=np.float32)  # pylint: disable=E1101,I0011
        
        dyndec = layers.DynamicDecoder(decoder, helper)
        output_t, state_t = dyndec.decode()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            output_act, state_act = sess.run([output_t, state_t])
            self.assertAllEqual(output_exp, output_act)

if __name__ == '__main__':
    tf.test.main()
