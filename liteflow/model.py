"""Basic workflow for a handy model creation.

To build a model, you have to define the network topology, the loss function
and many other ops. Since you might want to run several experiments with
different model configurations, ecc., you need a flexible way of changing the
behaviour of your building process. Such operations can be registered as
properties of the `ModelBuildOps` class. An instance of such class will be
passed to the factory method `Model.build` which will take care of invoking
such operations in the proper order and read in the context -- i.e. via
tf.Graph collections or liteflow.graphutils.GraphHelper maps -- the needed
objects.

Example:
```python
import tensorflow as tf
from liteflow import graphutils
from liteflow import model

# we want to easily switch between two different training algorithms.

def build_sgd_train_ops():
    loss_map = graphutils.get_map(graphutils.GraphHelperKeys.LOSS_OPS_MAP)
    loss_op = loss_map[graphutils.GraphHelperKeys.LOSS_OPS_DEFAULT_ITEM]
    trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    global_step = tf.get_collection(tf.GraphKeys.GLOBAL_STEP)[0]
    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train_op = optimizer.minimize(loss, var_list=trainable_vars)
    graphutils.put_in_map(
        graphutils.GraphHelperKeys.TRAIN_OPS_MAP,
        graphutils.GraphHelperKeys.TRAIN_OPS_DEFAULT_ITEM,
        train_op)

def build_adam_train_ops():
    loss_map = graphutils.get_map(graphutils.GraphHelperKeys.LOSS_OPS_MAP)
    loss_op = loss_map[graphutils.GraphHelperKeys.LOSS_OPS_DEFAULT_ITEM]
    trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    global_step = tf.get_collection(tf.GraphKeys.GLOBAL_STEP)[0]
    optimizer = tf.train.AdamOptimizer()  # the only different line
    train_op = optimizer.minimize(loss, var_list=trainable_vars)
    graphutils.put_in_map(
        graphutils.GraphHelperKeys.TRAIN_OPS_MAP,
        graphutils.GraphHelperKeys.TRAIN_OPS_DEFAULT_ITEM,
        train_op)

build_ops_01 = ModelBuildOps()
build_ops_01.build_train_ops = build_sgd_train_ops
# add different build ops like the following

model_01 = Model.build(build_ops_01, name='Model01',
                       description='Optimized with SGD')

build_ops_02 = ModelBuildOps()
build_ops_02.build_train_ops = build_adam_train_ops
# add different build ops like the following

model_02 = Model.build(build_ops_02, name='Model02',
                       description='Optimized with Adam')
```
"""


import tensorflow as tf
import liteflow.graphutils as gu


class ModelBuildOps(object):
    """Model building operations, all in one place.

    The ModelBuildOperation collects all the operations that
    need to be performed in order to build a Model. You can pass
    an instance of such class -- opportunely configured -- to the
    Model.build() class method in order to perform all the opearations
    in the right order.

    The ModelBuildeOps exposes a set of properties which are intended
    to be functions without arguments. Each function is intended to perform
    some actions in the context of the `tf.Graph`. You can set such functions
    just setting them as properties. Such mechanism -- which might change in
    the future -- provides you an easy way to change your building operations
    w.r.t. some configurations, ecc. with a minimum impact on the code.

    Example:
    ```python
    import tensorflow as tf
    from liteflow import graphutils
    from liteflow impoer model

    def build_inputs_and_targets():
        map_key = graphutils.GraphHelperKeys.INPUTS_MAP
        x = tf.placeholder(tf.float32, shape=[128, 1024])
        graphutils.get_helper().put_in_map(map_key, 'x', x)
        y = tf.placeholder(tf.float32, shape=[128, 1024])
        graphutils.get_helper().put_in_map(map_key, 'x', x)
        target = tf.placeholder(tf.float32, shape=[128, 1])
        graphutils.get_helper().put_in_map(map_key, 'target', target)


    build_ops = ModelBuildOps()
    build_ops.build_inputs_and_targets = build_inputs_and_targets
    # ...

    model = Model.build(build_ops)
    ```
    """

    def __init__(self):
        self._build_global_step = None
        self._build_inputs_and_targets = None
        self._build_outputs = None
        self._build_loss_ops = None
        self._build_train_ops = None
        self._build_summary_ops = None
        self._build_eval_ops = None

    @property
    def build_global_step(self):
        """Build the global step and place it in the `GLOBAL_STEP` collection."""
        return self._build_global_step

    @build_global_step.setter
    def build_global_step(self, func):
        self._build_global_step = func

    @property
    def build_inputs_and_targets(self):
        """Build the `INPUTS_MAP` and the `TARGETS_MAP` maps."""
        return self._build_inputs_and_targets

    @build_inputs_and_targets.setter
    def build_inputs_and_targets(self, func):
        self._build_inputs_and_targets = func

    @property
    def build_outputs(self):
        """Build the `OUTPUTS_MAP` map."""
        return self._build_outputs

    @build_outputs.setter
    def build_outputs(self, func):
        self._build_outputs = func

    @property
    def build_loss_ops(self):
        """Buld the `LOSS_OPS_MAP` map."""
        return self._build_loss_ops

    @build_loss_ops.setter
    def build_loss_ops(self, func):
        self._build_eval_ops = func

    @property
    def build_train_ops(self):
        """Build the `TRAIN_OPS_MAP` map."""
        return self._build_train_ops

    @build_train_ops.setter
    def build_train_ops(self, func):
        self._build_train_ops = func

    @property
    def build_summary_ops(self):
        """Build the `SUMMARY_OPS_MAP` map."""
        return self._build_summary_ops

    @build_summary_ops.setter
    def build_summary_ops(self, func):
        self._build_summary_ops = func

    @property
    def build_eval_ops(self):
        """Build the `EVAL_OPS_MAP` map."""
        return self._build_eval_ops

    @build_eval_ops.setter
    def build_eval_ops(self, func):
        self._build_eval_ops = func



class Model(object):
    """Basic model implementation.

    The Model class aims to collect some elements that can characterize
    a machine learning model, making easy to move them together back and
    forth and to interact with them in a object-oriented way.
    """
    def __init__(self, global_step, inputs, targets, outputs,
                 loss_ops=None, train_ops=None,
                 summary_ops=None, eval_ops=None,
                 name=None, description=None):
        """Initializes a new Model instance.
        """
        self._global_step = global_step
        self._inputs = inputs or {}
        self._targets = targets or {}
        self._outputs = outputs or {}
        self._loss_ops = loss_ops or {}
        self._train_ops = train_ops or {}
        self._summary_ops = summary_ops or {}
        self._eval_ops = eval_ops or {}
        self._name = name or 'Model_' + self.__class__.__name__
        self._description = description or self._name

    @property
    def global_step(self):
        """The global step of the model."""
        return self._global_step

    @property
    def inputs(self):
        """A dictionary with input tensors."""
        return self._inputs

    @property
    def targets(self):
        """A dictionary with target tensors."""
        return self._targets

    @property
    def outputs(self):
        """A dictionary with output tensors."""
        return self._outputs

    @property
    def loss_ops(self):
        """A dicitionary with the loss ops."""
        return self._loss_ops

    @property
    def train_ops(self):
        """A dictionary of train ops."""
        return self._train_ops

    @property
    def summary_ops(self):
        """A dictionary of summary ops."""
        return self._summary_ops

    @property
    def eval_ops(self):
        """A dictionary of ops used for evaluation."""
        return self._eval_ops

    @property
    def name(self):
        """The name of the mode."""
        return self._name

    @property
    def description(self):
        """A human redable description of the model."""
        return self._description


    @staticmethod
    def build(build_ops, name=None, description=None):
        """Build a model instance.

        The `Model.build` method is a factory method for the model.
        The `build_ops` argument is an instance of the `ModelBuildOps`
        class that collects all the operations to be invoked to build
        a model. This method implement the proper plan to build the model,
        invoking such operations in the right order.

        Arguments:
          build_ops: a `ModelBuildOps` instance containing the actual
            building operations to be invoked.
          name: a `str` representing the name for the model.
          description: a `str` representing a description of the model.

        Returns:
          a `Model` instance.
        """

        build_ops.build_global_step()
        global_step = tf.get_collection(tf.GraphKeys.GLOBAL_STEP)[0]

        build_ops.build_inputs_and_targets()
        inputs = gu.get_map(gu.GraphHelperKeys.INPUTS_MAP)
        targets = gu.get_map(gu.GraphHelperKeys.TARGETS_MAP)

        build_ops.build_outputs()
        outputs = gu.get_map(gu.GraphHelperKeys.OUTPUTS_MAP)

        build_ops.build_loss_ops()
        loss_ops = gu.get_map(gu.GraphHelperKeys.LOSS_OPS_MAP)

        build_ops.build_train_ops()
        train_ops = gu.get_map(gu.GraphHelperKeys.TRAIN_OPS_MAP)

        build_ops.build_summary_ops()
        summary_ops = gu.get_map(gu.GraphHelperKeys.SUMMARY_OPS_MAP)

        build_ops.build_eval_ops()
        eval_ops = gu.get_map(gu.GraphHelperKeys.EVAL_OPS_MAP)

        model = Model(
            global_step, inputs, targets, outputs,
            loss_ops, train_ops, summary_ops, eval_ops,
            name=name, description=description)
        return model
