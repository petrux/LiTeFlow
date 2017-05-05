"""Test module for the `liteflow.model` module."""

import unittest

import tensorflow as tf

from liteflow import graphutils
from liteflow import model

from liteflow.graphutils import GraphHelperKeys as GHK

class TestModel(unittest.TestCase):
    """Test case for the `liteflow.model.Model` class."""

    def test_base(self):
        """Base test for the `liteflow.model.Model` class."""

        # 1. BUILD THE MODEL.
        # The model is extremely simple: two 0/1 signals, one
        # is processed through a 1LP to predict the AND and the
        # other one through another 1LP to predict the OR.
        global_step = tf.train.get_or_create_global_step()
        with tf.variable_scope('Input'):
            inputs = {
                'X': tf.placeholder(tf.float32, shape=[128]),
                'Y': tf.placeholder(tf.float32, shape=[128])
            }
            joined = tf.concat(
                [tf.expand_dims(inputs['X'], 1),
                 tf.expand_dims(inputs['Y'], 1)],
                axis=-1)
        with tf.variable_scope('Targets'):
            targets = {
                'AND': inputs['X'] * inputs['Y'],
                'OR': tf.cast(tf.greater(inputs['X'] + inputs['Y'], 0), tf.float32)
            }
        outputs = {}
        loss_ops = {}
        train_ops = {}
        summary_ops = {}
        eval_ops = {}

        with tf.variable_scope('OR'):
            weights = tf.get_variable(name='w', dtype=tf.float32, shape=[2, 1])
            bias = tf.get_variable(name='b', dtype=tf.float32, shape=[])
            logits = tf.matmul(joined, weights) + bias
            pred = tf.squeeze(tf.sigmoid(logits))
            outputs['OR'] = pred

        with tf.variable_scope('AND'):
            weights = tf.get_variable(name='w', dtype=tf.float32, shape=[2, 1])
            bias = tf.get_variable(name='b', dtype=tf.float32, shape=[])
            logits = tf.matmul(joined, weights) + bias
            pred = tf.squeeze(tf.sigmoid(logits))
            outputs['AND'] = pred

        with tf.variable_scope('Loss'):
            or_loss = tf.losses.mean_squared_error(
                targets['OR'], outputs['OR'], scope='ORLoss')
            loss_ops['OR'] = or_loss
            and_loss = tf.losses.mean_squared_error(
                targets['AND'], outputs['AND'], scope='ANDLoss')
            loss_ops['AND'] = and_loss

        with tf.variable_scope('BackProp'):
            or_train = tf.train.GradientDescentOptimizer(0.1).minimize(or_loss)
            train_ops['OR'] = or_train
            and_train = tf.train.AdamOptimizer().minimize(and_loss)
            train_ops['AND'] = and_train

        with tf.variable_scope('Summaries'):
            or_loss_summary = tf.summary.scalar('ORLoss', or_loss)
            summary_ops['OR'] = or_loss_summary
            and_loss_summary = tf.summary.scalar('ANDLoss', and_loss)
            summary_ops['AND'] = and_loss_summary

        def _accuracy(truth, test):
            truth = tf.cast(truth, tf.int32)
            test = tf.cast(tf.greater(test, 0.5), tf.int32)
            both = truth * test
            correct = tf.reduce_sum(both)
            accuracy = correct / 128
            return accuracy

        with tf.variable_scope('Eval'):
            eval_ops['OR'] = _accuracy(targets['OR'], outputs['OR'])
            eval_ops['AND'] = _accuracy(targets['AND'], outputs['AND'])

        # 2. BUILD THE ModelBuilderOps.
        class _InvokeOnce(object):

            def __init__(self, func):
                self._func = func
                self._invoked = False

            @property
            def invoked(self):
                """True if the function has been invoked."""
                return self._invoked

            def __call__(self):
                if self._invoked:
                    raise RuntimeError('Already invoked!')
                self._invoked = True
                self._func()

            @staticmethod
            def invoke_once(func):
                """Wraps the function into a _InvokeOnce object."""
                return _InvokeOnce(func)

        @_InvokeOnce.invoke_once
        def _build_global_step():
            tf.add_to_collection(tf.GraphKeys.GLOBAL_STEP, global_step)
            self.assertTrue(_build_global_step.invoked)
            self.assertFalse(_build_inputs_and_targets.invoked)
            self.assertFalse(_build_outputs.invoked)
            self.assertFalse(_build_loss_ops.invoked)
            self.assertFalse(_build_train_ops.invoked)
            self.assertFalse(_build_summary_ops.invoked)
            self.assertFalse(_build_eval_ops.invoked)

        @_InvokeOnce.invoke_once
        def _build_inputs_and_targets():
            for key, value in inputs.iteritems():
                graphutils.put_in_map(GHK.INPUTS_MAP, key, value)
            for key, value in targets.iteritems():
                graphutils.put_in_map(GHK.TARGETS_MAP, key, value)
            self.assertTrue(_build_global_step.invoked)
            self.assertTrue(_build_inputs_and_targets.invoked)
            self.assertFalse(_build_outputs.invoked)
            self.assertFalse(_build_loss_ops.invoked)
            self.assertFalse(_build_train_ops.invoked)
            self.assertFalse(_build_summary_ops.invoked)
            self.assertFalse(_build_eval_ops.invoked)

        @_InvokeOnce.invoke_once
        def _build_outputs():
            for key, value in outputs.iteritems():
                graphutils.put_in_map(GHK.OUTPUTS_MAP, key, value)
            self.assertTrue(_build_global_step.invoked)
            self.assertTrue(_build_inputs_and_targets.invoked)
            self.assertTrue(_build_outputs.invoked)
            self.assertFalse(_build_loss_ops.invoked)
            self.assertFalse(_build_train_ops.invoked)
            self.assertFalse(_build_summary_ops.invoked)
            self.assertFalse(_build_eval_ops.invoked)

        @_InvokeOnce.invoke_once
        def _build_loss_ops():
            for key, value in loss_ops.iteritems():
                graphutils.put_in_map(GHK.LOSS_OPS_MAP, key, value)
            self.assertTrue(_build_global_step.invoked)
            self.assertTrue(_build_inputs_and_targets.invoked)
            self.assertTrue(_build_outputs.invoked)
            self.assertTrue(_build_loss_ops.invoked)
            self.assertFalse(_build_train_ops.invoked)
            self.assertFalse(_build_summary_ops.invoked)
            self.assertFalse(_build_eval_ops.invoked)

        @_InvokeOnce.invoke_once
        def _build_train_ops():
            for key, value in train_ops.iteritems():
                graphutils.put_in_map(GHK.TRAIN_OPS_MAP, key, value)
            self.assertTrue(_build_global_step.invoked)
            self.assertTrue(_build_inputs_and_targets.invoked)
            self.assertTrue(_build_outputs.invoked)
            self.assertTrue(_build_loss_ops.invoked)
            self.assertTrue(_build_train_ops.invoked)
            self.assertFalse(_build_summary_ops.invoked)
            self.assertFalse(_build_eval_ops.invoked)

        @_InvokeOnce.invoke_once
        def _build_summary_ops():
            for key, value in summary_ops.iteritems():
                graphutils.put_in_map(GHK.SUMMARY_OPS_MAP, key, value)
            self.assertTrue(_build_global_step.invoked)
            self.assertTrue(_build_inputs_and_targets.invoked)
            self.assertTrue(_build_outputs.invoked)
            self.assertTrue(_build_loss_ops.invoked)
            self.assertTrue(_build_train_ops.invoked)
            self.assertTrue(_build_summary_ops.invoked)
            self.assertFalse(_build_eval_ops.invoked)

        @_InvokeOnce.invoke_once
        def _build_eval_ops():
            for key, value in eval_ops.iteritems():
                graphutils.put_in_map(GHK.EVAL_OPS_MAP, key, value)
            self.assertTrue(_build_global_step.invoked)
            self.assertTrue(_build_inputs_and_targets.invoked)
            self.assertTrue(_build_outputs.invoked)
            self.assertTrue(_build_loss_ops.invoked)
            self.assertTrue(_build_train_ops.invoked)
            self.assertTrue(_build_summary_ops.invoked)
            self.assertTrue(_build_eval_ops.invoked)

        mbo = model.ModelBuildOps()
        mbo.build_global_step = _build_global_step
        mbo.build_inputs_and_targets = _build_inputs_and_targets
        mbo.build_outputs = _build_outputs
        mbo.build_loss_ops = _build_loss_ops
        mbo.build_train_ops = _build_train_ops
        mbo.build_summary_ops = _build_summary_ops
        mbo.build_eval_ops = _build_eval_ops

        # 3. BUILD THE MODEL and test your assertions.
        instance = model.Model.build(mbo)
        self.assertEquals(instance.global_step, global_step)
        self.assertEquals(instance.inputs, inputs)
        self.assertEquals(instance.targets, targets)
        self.assertEquals(instance.outputs, outputs)
        self.assertEquals(instance.loss_ops, loss_ops)
        self.assertEquals(instance.train_ops, train_ops)
        self.assertEquals(instance.summary_ops, summary_ops)
        self.assertEquals(instance.eval_ops, eval_ops)
