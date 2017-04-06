"""Base contract for lite layers implementation."""

import abc

import tensorflow as tf

import liteflow.utils as utils


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

        Raises:
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
