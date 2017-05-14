"""Base contract for lite layers implementation."""

import abc

import tensorflow as tf

from liteflow import ops
from liteflow import utils


class Layer(object):
    """Base class for the layer implementation.

    NOTA BENE: subclassing of this class requires a certain
    understanding of the design choices that have been made
    so far. So, please, read carefully the documentation, the
    tests, the comments and (of course) the code.

    ```python
    # Example of implementation of liteflow.layer.Layer

    import tensorflow as tf
    import liteflow.layers as layers

    class FooLayer(layers.Layer):

        def __init__(self, trainable=True, scope='MyScope', other_param=None):
            super(FooLayer, self).__init__(trainable=trainable, scope=scope)
            self._useless = other_param
            # ...do the rest of your stuff.

        def _build(*args, **kwargs):
            print self.__class__.__name__ + '._build()'

        def _call(inp, *args, **kwargs):
            self.__class__.__name__ + '._build(' + str(inp) + ')'


    tensor = tf.get_variable(name='Variable', shape=[2, 2])
    foo = FooLayer()
    foo(tensor)
    ```
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, trainable=True, scope=None):
        """Initliazes a new Layer instance.

        Arguments:
          trainable: a boolean flag indicating if the variables created bytearray
            the layer should be considered as trainable.
          scope: the scope to be used in the layer. Must be `None`,
            `str` or `tf.VariableScope`.

        Raises:
          TypeError: if the `scope` argument is not `None`, `str` or `tf.VariableScope`.

        The scope of a layer does not have to be set in the initialisation phase. It can
        be set in an explicit call of the `build()` or the `__call__()` methods. It can also
        never be explicitly set. In this case, the layer will use a default scope. The default
        behaviour is to use a scope named with the `self.name` property value.
        """
        self._trainable = trainable
        self._built = False
        self._scope = None
        self._name = self.__class__.__name__
        self._variables = []

        if scope is not None:
            self._scope = utils.as_scope(scope)
            self._name = self._scope.name

    @property
    def name(self):
        """The name of the layer."""
        return self._name

    @property
    def scope(self):
        """The variable scope bound to the layer."""
        return self._scope

    @property
    def trainable(self):
        """True if the layer is trainable."""
        return self._trainable

    @property
    def variables(self):
        """Return all the variables used by the layer."""
        return self._variables

    @property
    def built(self):
        """True if the layer has already been built."""
        return self._built

    def _default_scope(self):

        """Returns the default scope."""
        return utils.as_scope(self._name)

    def _set_scope(self, scope):
        """Set the given scope as the scope of the layer.

        Set the scope for the layer, that will be accessible through the
        `self.scope` property.

        Argsuments:
          scope: the given scope, of type `str` of `tf.VariableScope`. If `None`,
            the one returned from the `self._default_scope()` method will be used.
        """
        if self._scope is None:
            if scope is None:
                self._scope = self._default_scope()
            else:
                self._scope = utils.as_scope(scope)
            return
        if scope is not None:
            tf.logging.warn(
                """Trying to set the scope %s while"""
                """ %s has already been set.""",
                utils.as_scope(scope).name, self._scope.name)

    @abc.abstractmethod
    def _build(self, *args, **kwargs):
        """Build the layer.

        NOTA BENE: this is an abstract method that must be implemented
        by concrete subclasses.
        """
        raise NotImplementedError(
            'This method must be implemented in subclasses.')

    def build(self, *args, **kwargs):
        """Build the layer.

        Build the inner state of the layer, with all the variables that
        will be used by the layer itself. Wraps `self._build()` applying
        pre- and post-processing.

        Arguments:
          *args: positional arguments to be passed to `self._build()`.
          **kwargs: keyword arguments to be passed to `self._build()`.
            **Note**, the kwarg `scope` is reserved for use by the layer.

        Raises:        print 'TRAINABLE: ' + str(self._trainable)


          RuntimeError: if the layer has already been built, i.e. if `self.built`
            is equals to `True`.

        NOTA BENE: this method works like this:
        1. setup the scope (if not set yet);
        2. within the scope context, invoke the `self._build()` template method;
        3. set the `self.built` property to `True`.
        When subclassing the `Layer` class, you shall provide an implementation of
        the `Layer._build()` method so that all the other boilerplate stuff (namely
        item 1. and 3. from the above list) are performed by the `Layer.build()`
        outer method.
        """
        if self._built:
            raise RuntimeError(
                'The layer %s has already been built.' %
                self._name)

        self._set_scope(kwargs.pop('scope', None))
        with tf.variable_scope(self._scope) as _:
            self._build(*args, **kwargs)
        self._built = True

    @abc.abstractmethod
    def _call(self, inp, *args, **kwargs):
        """Run the layer logic.

        NOTA BENE: this is an abstract method that must be implemented
        by concrete subclasses.
        """
        raise NotImplementedError(
            'This method must be implemented in subclasses.')

    def __call__(self, inp, *args, **kwargs):
        """Wraps `self._call()`, applying pre- and post-processing steps.

        Arguments:
          inputs: input tensor that will be passed to `self._call()`.
          *args: additional positional arguments to be passed to `self._call()`.
          **kwargs: additional keyword arguments to be passed to `self._call()`.
            **Note**, the kwarg `scope` is reserved for use by the layer.

        Returns:
          Output tensor.
        """
        self._set_scope(kwargs.pop('scope', None))
        with tf.variable_scope(self._scope) as scope:
            if not self._built:
                self._build()
                self._built = True
            else:
                scope.reuse_variables()
            return self._call(inp, *args, **kwargs)

    def apply(self, inp, *args, **kwargs):
        """Wrapper for the `self.__call__()` method."""
        return self.__call__(inp, *args, **kwargs)


class BahdanauAttention(Layer):
    """Attention mechanism as in Bahdanau et al. 2015.

    The attention mechanism implemented in this class is the one
    described by Bahdanau et al. here: https://arxiv.org/abs/1409.0473.
    The attention states and the query are projected to the attention
    inner size, then summed together and processed with a tanh and
    finally dot producted with an attention vector.
    """

    _KERNEL_NAME = 'Kernel'
    _VECTOR_NAME = 'Vector'
    _WEIGHTS_NAME = 'Weights'

    def __init__(self, states, size, trainable=True, scope=None):
        """Initiailzes a new instance of the BahdanauAttention class.

        The attention mechanism implemented in this class is the one
        described by Bahdanau et al. here: https://arxiv.org/abs/1409.0473.
        The attention states and the query are projected to the attention
        inner size, then summed together and processed with a tanh and
        finally dot producted with an attention vector. All the operations
        are performed on a reference size, named as the attention size, which
        must be set during the initialization phase (with the `size` argument).

        Arguments:
          states: 3-D Tensor of shape [batch, timesteps, state] representing the
            states on which the attention scores will be computed; the third dimension
            of the tensor must be statically determined.
          size: int representing the inner attention size;
          trainable: if True, variables will be trainable;
          scope: None, str or tf.VariableScope representing the variable scope
            of the layer which will be used to create all the needed variables.

        Raises:
          ValueError: if the last dimension of the `state` argument is not
            statically determined.
        """
        super(BahdanauAttention, self).__init__(trainable=trainable, scope=scope)
        self._states = states
        self._size = size
        self._memory = None
        self._vector = None
        self._var_op_names = set()

        # check that the last dimension of the `states`
        # variable is fully defined.
        state_size = states.get_shape()[-1].value
        if state_size is None:
            raise ValueError('Last dimension of `states` must be defined, found %s'
                             % str(tf.shape(states)))
        self._state_size = state_size

    @property
    def states(self):
        """The attention states."""
        return self._states

    @property
    def size(self):
        """The attention size."""
        return self._size

    def _add_var(self, var):
        key = var.op.name
        if key not in self._var_op_names:
            self._var_op_names.add(key)
            self._variables.append(var)

    def _build(self, *args, **kwargs):
        """Implement the layer building logic."""
        batch = tf.shape(self._states)[0]
        length = tf.shape(self._states)[1]
        shape = [batch, length, 1, self._state_size]
        states = tf.reshape(self._states, shape=shape)
        kernel = tf.get_variable(self._KERNEL_NAME,
                                 [1, 1, self._state_size, self._size],
                                 trainable=self._trainable)
        self._memory = tf.nn.conv2d(states, kernel, [1, 1, 1, 1], "SAME")
        self._vector = tf.get_variable(self._VECTOR_NAME,
                                       shape=[self._size],
                                       trainable=self._trainable)
        self._add_var(kernel)
        self._add_var(self._vector)

    def _call(self, query, *args, **kwargs):
        """Implement the layer logic."""
        query_size = query.get_shape()[-1].value
        if query_size is None:
            raise ValueError(
                'Last dimension of `query` must be defined, found %s'
                % str(tf.shape(query)))

        weights = tf.get_variable(self._WEIGHTS_NAME,
                                  shape=[query_size, self._size],
                                  trainable=self._trainable)
        self._add_var(weights)
        features = tf.reshape(tf.matmul(query, weights), [-1, 1, 1, self._size])
        activations = self._vector * tf.tanh(self._memory + features)
        activations = tf.reduce_sum(activations, [2, 3])
        return activations

    def __call__(self, query, *args, **kwargs):
        """Calculate the attention scores over `self.states` for the given queries.

        Arguments:
          query: a 2-D Tensor of shape [batch, query]; the last dimension must
            be statically determined.
          *args: additional positional arguments to be passed to `self._call()`.
          **kwargs: additional keyword arguments to be passed to `self._call()`.
            **Note**, the kwarg `scope` is reserved for use by the layer.

        Returns:
          A 2-D tensor of shape [batch, timesteps] where `timesteps` is the second
            dimension of the `self.states` tensor.

        Raises:
          ValueError: if the last dimension of the `query` argument
            is not statically determined.
        """
        return super(BahdanauAttention, self).__call__(query, *args, **kwargs)

    def apply(self, query, *args, **kwargs):
        """Wrapper for the `self.__call__()` method."""
        return super(BahdanauAttention, self).__call__(query, *args, **kwargs)


class PointingSoftmax(Layer):
    """Implements a PointingSoftmax over a set of attention states."""

    def __init__(self, attention, mask=None, scope='PointingSoftmax'):
        super(PointingSoftmax, self).__init__(trainable=attention.trainable, scope=scope)
        self._attention = attention
        self._mask = mask
        # TODO(petrux): check dimension of the mask w.r.t. attention states.
        # TODO(petrux): check that the injected `attentio` has not been already built.

    def _build(self, *args, **kwargs):
        """This method is just a wrapper so no operation is actually performed."""
        self._attention.build()

    def _call(self, query, *args, **kwargs):
        activations = self._attention.apply(query, *args, **kwargs)
        weights = ops.softmax(activations, self._mask)
        eweights = tf.expand_dims(weights, axis=2)
        context = tf.reduce_sum(self._attention.states * eweights, axis=1)
        return weights, context

    def apply(self, query, *args, **kwargs):
        """Wrapper for the __call__() method."""  # TODO(petrux): add actual documentation
        return super(PointingSoftmax, self).__call__(self, query, *args, **kwargs)