"""Base contract for lite layers implementation."""
# TODO(petrux): consider about getting rid of the Layer super class.
#   Such class is extremely useful when dealing with unpacked/unstacked
#   tensors, but since the tensor array and symbolic looping ops are now
#   alive an' kickin' in TF, there is no need anymore of such stuff.

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

    def __init__(self, trainable=True, name=None, **kwargs):
        """Initliazes a new Layer instance.

        Arguments:
          trainable: a boolean flag indicating if the variables created bytearray
            the layer should be considered as trainable.
        """
        self._trainable = trainable
        self._name = name or self.__class__.__name__
        self._scope = None
        self._reuse = False
        self._variables = []

        scope = kwargs.get('_scope')
        if scope is not None:
            self._scope = utils.as_scope(scope)
        reuse = kwargs.get('_reuse')
        if reuse:
            self._reuse = True

    @property
    def name(self):
        """The name of the layer."""
        return self._name

    @property
    def scope(self):
        """The name of variable scope bound to the layer.

        Raises:
          ValueError: if the layer has not be called yet.
        """
        if not self._scope:
            raise ValueError('No name available for layer scope because the layer "' +
                             self.name + '" has not been used yet. The scope name ' +
                             ' is determined the first time the layer instance is ' +
                             'called. You must therefore call the layer before ' +
                             'querying `scope`.')
        return self._scope.name

    @property
    def trainable(self):
        """True if the layer is trainable."""
        return self._trainable

    def _default_scope(self):
        """Returns the default scope."""
        return utils.as_scope(self._name)

    def _set_scope(self, scope):
        """Set the given scope as the scope of the layer.

        If not already present, set the scope for the layer. The name of such scope
         will be accessible through the `self.scope` property.

        Argsuments:
          scope: the given scope, of type `str` of `tf.VariableScope`. If `None`,
            the one returned from the `self._default_scope()` method will be used.
        """
        if self._scope is None:
            if self._reuse:
                self._scope = next(tf.variable_scope(  # pylint: disable=I0011,E1101
                    scope if scope is not None else self._default_scope()).gen)
            else:
                self._scope = next(tf.variable_scope(  # pylint: disable=I0011,E1101
                    scope, default_name=self._default_scope().name).gen)

    @abc.abstractmethod
    def _call_helper(self, *args, **kwargs):
        """Run the layer logic.

        NOTA BENE: this is an abstract method that must be implemented
        by concrete subclasses.
        """
        raise NotImplementedError(
            'This method must be implemented in subclasses.')

    def __call__(self, *args, **kwargs):
        """Execute the layer logics, applying pre- and post-processing steps.

        The layer logic is supposed to happen in the concrete implementation
        of the abstract method `_call_helper()`. This method executes some pre- and
        post-processing. We can sketch the internal workflow of the method as:
        1. get the scope (as the `scope` kwarg) and check if variables should be reused
        2. invoke the `_call_helper()` method with the very same arguments (with the only
           exception of the `scope` argument, if present).
        3. set the future scope instances to be reused.
        4. return the result of invokation of `_call_helper()` (from 2).

        Arguments:
          *args: additional positional arguments.
          **kwargs: additional keyword arguments. **Note** that the
            keyword argument `scope` is reserved for use by the layer.

        Returns:
          a tuple of output tensors (or `None`).
        """
        self._set_scope(kwargs.pop('scope', None))
        # pylint: disable=E1129
        with tf.variable_scope(self._scope) as scope:
            if self._reuse:
                scope.reuse_variables()
            result = self._call_helper(*args, **kwargs)
            self._reuse = True
        return result


class BahdanauAttention(Layer):
    """Attention mechanism as in Bahdanau et al. 2015.

    Calculate the attention scores over `self.states` for the given queries.
    The attention mechanism implemented in this class is the one
    described by Bahdanau et al. here: https://arxiv.org/abs/1409.0473.
    The attention states and the query are projected to the attention
    inner size, then summed together and processed with a tanh and
    finally dot producted with an attention vector.

    Arguments:
      query: a 2-D Tensor of shape [batch_size, query_size]; the last dimension
        must be statically determined.
      scope: a `str` or `tf.VariableScope`; if the scope has already been set in a
        previous layer invocation, will be ignored.

    Returns:
      A 2-D tensor of shape [batch, timesteps] where `timesteps` is the second
        dimension of the `self.states` tensor.

    Raises:
      ValueError: if the last dimension of the `query` argument
        is not statically determined.
    """

    _KERNEL_NAME = 'Kernel'
    _VECTOR_NAME = 'Vector'
    _WEIGHTS_NAME = 'Weights'

    def __init__(self, states, inner_size, trainable=True, scope=None):
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
          inner_size: int representing the inner attention size;
          trainable: if True, variables will be trainable;
          scope: None, str or tf.VariableScope representing the variable scope
            of the layer which will be used to create all the needed variables.

        Raises:
          ValueError: if the last dimension of the `state` argument is not
            statically determined.
        """
        super(BahdanauAttention, self).__init__(trainable=trainable, scope=scope)
        self._states = states
        self._size = inner_size
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
    def inner_size(self):
        """The inner attention size."""
        return self._size

    def _add_var(self, var):
        key = var.op.name
        if key not in self._var_op_names:
            self._var_op_names.add(key)
            self._variables.append(var)

    def _call_helper(self, query):    # pylint: disable=I0011,W0221
        query_size = query.get_shape()[-1].value
        if query_size is None:
            raise ValueError(
                'Last dimension of `query` must be defined, found %s'
                % str(tf.shape(query)))

        batch = tf.shape(self._states)[0]
        length = tf.shape(self._states)[1]
        states_shape = [batch, length, 1, self._state_size]
        states = tf.reshape(self._states, shape=states_shape)
        kernel = tf.get_variable(self._KERNEL_NAME,
                                 [1, 1, self._state_size, self._size],
                                 trainable=self._trainable)
        self._memory = tf.nn.conv2d(states, kernel, [1, 1, 1, 1], "SAME")
        self._vector = tf.get_variable(self._VECTOR_NAME,
                                       shape=[self._size],
                                       trainable=self._trainable)

        weights = tf.get_variable(self._WEIGHTS_NAME,
                                  shape=[query_size, self._size],
                                  trainable=self._trainable)
        self._add_var(weights)
        features = tf.reshape(tf.matmul(query, weights), [-1, 1, 1, self._size])
        activations = self._vector * tf.tanh(self._memory + features)
        activations = tf.reduce_sum(activations, [2, 3])
        return activations

    def __call__(self, query, scope=None):  # pylint: disable=I0011,W0221
        return super(BahdanauAttention, self).__call__(query, scope=scope)


class LocationSoftmax(Layer):
    """Implements a LocationSoftmax over a set of attention states.

    The logics implemented in this class is described in this paper by Caglar
    Gulcehre et al., https://arxiv.org/abs/1603.08148

    Given a set of attention states as a 3D tensor of shape [batch_size, timesteps,
    state_size] and an attention mechanism over them, this layer returns a 2D tensor of
    shape [batch_size, timesteps] where the i-th element of the batch represents a
    probability distribution over the states of the i-th element of the input batch.
    Since the sequences in the input batch can have actual different lengths, such
    distributions are computed only over the actual elements.

    Arguments:
      query: a 2-D Tensor of shape [batch_size, query_size]; the last dimension
        must be statically determined.

    Returns:
      a tuple of 2 tensors:
        location_softmax: A 2D tensor of shape [batch_size, timesteps] where
          `timesteps` is the second dimension of the attention states; such
          tensor is intended to represent a probability distribution across
          the states.
        attention_context: a 2D tenros of shape [batch_size, state_size] where
          `state_size` is the size of the attention states (the 3rd dimension);
          such tensor is intended to represent the attention context vector over
          the attention states  where the `location_softmax` tensor has been used
          as the coeffiecient vector.
    """

    def __init__(self, attention, sequence_length=None, scope='LocationSoftmax'):
        """Initialize a LocationSoftmax layer.

        Arguments:
          attention: an attention layer (such as BahdanauAttention).
          sequente_length: a `1D` Tensor of shape [batch_size] where each element
            represent the number of actual meaningful elements in each sequence
            of the attention states.
        """
        super(LocationSoftmax, self).__init__(trainable=attention.trainable, scope=scope)
        self._attention = attention
        self._sequence_length = sequence_length

    @property
    def attention(self):
        """The attention layer used by the pointing softmax."""
        return self._attention

    @property
    def sequence_length(self):
        """A tensor representing the actual lenght of sequences of states in a batch."""
        return self._sequence_length

    def zero_location_softmax(self, batch_size):
        """A tensor representign the zero location softmax."""
        location_size = utils.get_dimension(self.attention.states, 1)
        shape = tf.stack([batch_size, location_size])
        return tf.zeros(shape=shape, dtype=tf.float32)

    def zero_attention_context(self, batch_size):
        """A tensor representing the zero attention context."""
        state_size = utils.get_dimension(self.attention.states, 2)
        shape = tf.stack([batch_size, state_size])
        return tf.zeros(shape=shape, dtype=tf.float32)

    def _call_helper(self, query):  # pylint: disable=I0011,W0221
        activations = self._attention(query)
        maxlen = utils.get_dimension(activations, -1)
        if self._sequence_length is not None:
            mask = tf.cast(tf.sequence_mask(self._sequence_length, maxlen), tf.float32)
        else:
            mask = None
        location = ops.softmax(activations, mask)
        weights = tf.expand_dims(location, axis=2)
        context = tf.reduce_sum(self._attention.states * weights, axis=1)
        return location, context

    def __call__(self, query, scope=None):  # pylint: disable=I0011,W0221
        return super(LocationSoftmax, self).__call__(query, scope=scope)


class PointingSoftmaxOutput(Layer):
    """Implements the output layer for a PointingDecoder.

    The logics implemented in this class is described in this paper by Caglar
    Gulcehre et al., https://arxiv.org/abs/1603.

    Arguments:
      decoder_out: a 2D tensor of shape [batch_size, decoder_out_size] representing
        the output of some decoding operation
      location_softmax: a 2D tensor of shape [batch_size, timesteps] that represents the
        location softmax (i.e. a probability distribution) over a set of attention states
      attention_context: a 2D tensor of shape [batch_size, state_size] representing the
        attention context vector for `decoder_out` w.r.t. to a set of input states calculated
        with the `location_softmax` as coeffiecients

    Returns:
      and returns a 2D tensor of shape [batch_size, shortlist_size + timesteps] where
      `shortlist_size` is the dimension of the outout vocabulary of the network and
      `timesteps` is the number of states, or the length, of the attention states.
      The output tensor is a probability distribution over a set of items represented
      by the elements in the shortlist (the first part) or some item from the input
      sequence, based on the position of the top scoring element in the second part
      of the outout vector. It means that:
      1. if the argmax of the output vector is in the interval [0, shortlist-1],
         then outout symbol will be taken from the shortlist
      2. otherwise, it is in the interval [shortlist, shortlist + timesteps - 1]
         and it will be considered a "position" in the [0, timesteps - 1] interval.
    """

    _SWITCH_KERNEL_NAME = 'SwitchKernel'
    _SWITCH_BIAS_NAME = 'SwitchBias'
    _SHORTLIST_KERNEL_NAME = 'ShortlistKernel'
    _SHORTLIST_BIAS_NAME = 'ShortlistBias'

    # pylint: disable=R0913
    def __init__(self, shortlist_size, decoder_out_size, state_size,
                 trainable=True, scope='PointingSoftmaxOutput'):
        """Initializes a new instance.

        Arguments:
          shorlist_size: a `int` representing the dimension of the known output vocabulary.
          decoder_out_size: a `int` representing the output size of the recoder.
          state_size: a `int` representing the size of the attention states.
          trainable: if `True`, the created variables will be trainable.
          scope: VariableScope for the created subgraph;.
        """
        super(PointingSoftmaxOutput, self).__init__(trainable=trainable, scope=scope)
        self._shortlist_size = shortlist_size
        self._decoder_out_size = decoder_out_size
        self._state_size = state_size

    @property
    def shortlist_size(self):
        """The size of the output shortlist."""
        return self._shortlist_size

    @property
    def decoder_out_size(self):
        """The decoder output size."""
        return self._decoder_out_size

    @property
    def state_size(self):
        """The attention context vector size."""
        return self._state_size

    def zero_output(self, batch_size, pointing_size):
        """Zero-state output of the layer.

        Arguments:
          batch_size: a `int` or unit Tensor representing the batch size.
          pointing_size: a `int` or unit Tensor representing the pointing scores size.

        Returns:
          a 2D tesor of size [batch_size, self.emission_size + pointing_size]
          of tf.float32 zeros, represernting the default first output tensor of the layer.
        """
        shape = tf.stack([batch_size, self._shortlist_size + pointing_size])
        return tf.zeros(shape, dtype=tf.float32)

    def _call_helper(self, decoder_out, location_scores, attention_context):  # pylint: disable=I0011,W0221

        switch_kernel = tf.get_variable(
            name=self._SWITCH_KERNEL_NAME,
            shape=[self._decoder_out_size + self._state_size, 1],
            trainable=self.trainable)
        switch_bias = tf.get_variable(
            name=self._SWITCH_BIAS_NAME,
            shape=[1],
            trainable=self.trainable)
        shortlist_kernel = tf.get_variable(
            self._SHORTLIST_KERNEL_NAME,
            shape=[self._decoder_out_size, self._shortlist_size],
            trainable=self.trainable)
        shortlist_bias = tf.get_variable(
            name=self._SHORTLIST_BIAS_NAME,
            shape=[self._shortlist_size],
            trainable=self.trainable)

        switch_in = tf.concat([decoder_out, attention_context], axis=1)
        switch = tf.matmul(switch_in, switch_kernel) + switch_bias
        switch = tf.nn.sigmoid(switch)
        shortlist = tf.matmul(decoder_out, shortlist_kernel) + shortlist_bias
        shortlist = tf.nn.sigmoid(shortlist)
        output = tf.concat([switch * shortlist, (1 - switch) * location_scores], axis=1)
        return output

    # pylint: disable=I0011,W0221,W0235
    def __call__(self, decoder_out, location_softmax, attention_context):
        return super(PointingSoftmaxOutput, self).__call__(
            decoder_out, location_softmax, attention_context)


class DecoderBase(object):
    """Base contract for a decoder."""

    __metaclass__ = abc.ABCMeta

    def __init__(self, trainable=True, name=None, **kwargs):  # pylint: disable=W0613
        self._trainable = trainable
        self._name = name or self.__class__.__name__

    @property
    def trainable(self):
        """True if the decoder is trainable."""
        return self._trainable

    @abc.abstractmethod
    def init_input(self):
        """Initial input state for the decoder.

        Returns:
          a `Tensor` of shape `[batch_size, input_size]` representing the initial
          input tensor for the decoder.
        """
        pass

    @abc.abstractmethod
    def init_state(self):
        """Initial inner state for the decoder.

        Returns:
          a `Tensor` or a tuple of `Tensor`s of arbitrary rand and `dtype` and
          shape [batch_size, ...] representing the initial inner state of the decoder.
        """
        pass

    @abc.abstractmethod
    def zero_output(self):
        """After-termination output.

        Returns:
          a `Tensor` of shape `[batch_size, output_size]` representing the output
          of the decoder after the termination of actual length.
        """
        pass

    @abc.abstractmethod
    def step(self, time, inp, state):
        """Decoder step.

        Arguments:
          time: a `0D` (i.e. scalar) `Tensor` of `dtype=tf.int32` representing
            the 0-based value of the current step in the loop.
          inp: a `2D` `Tensor` of shape [batch_size, input_size] representing
            the decoder input for the current step in the loop.
          state: a `Tensor` or a tuple of `Tensor`s of arbitrary rand and `dtype` and
            shape [batch_size, ...] representing the inner state of the decoder for
            the current loop step.

        Returns:
          a 4-tuple of tensor made of:
            output: a `Tensor` of shape `[batch_size, output_size]` representing the output
              of the decoder for the current step in the loop.
            next_inp: a `Tensor` of shape `[batch_size, input_size]` representing the
              input for the decoder at the next step in the loop.
            next_state: a `Tensor` or a tuple of `Tensor`s of arbitrary rand and `dtype` and
              shape [batch_size, ...] representing the inner state of the decoder for the
              next step in the loop.
            finished: a `Tensor` of shape `[batch_size]` and `dtype=tf.bool` representing
              which sequence in the batch has already reached the termination (according
              to some custom internal logic).
        """
        pass


class PointingSoftmaxDecoder(DecoderBase):
    """Pointing softmax decoder.

    The PointingSoftmaxDecoder implements a single decoding step using
    a pointing softmax that can emit a symbol from a shortlist or point
    one of the words in the input sentence.

    To build a PointingSoftmaxDecoder instance, you need to initialize
    the instance the following arguments:
      cell: a `tf.nn.RNNCell`.
      location_softmax: a `LocationSoftmax` instance.
      pointing_output: a `PointingSoftmaxOutput` instance.
      input_size: an `int` representing the desires shape of the input size.
      decoder_inputs: the (optional) gold decoder inputs to be used in training
        to feed back the decoder at each timestep.
      trainable: if `True`, the decoder is trainable.
      name: a string representing the name of the instance.

    *NOTE* that `input_size` must be an integere and must be the same both in
    training and in inference mode since it defines the dimension of the weights
    of the network. All inputs, both the provided ones and those that are fed
    back from the output, will be fit (trimmed or padded) to such value along
    their last axis.

    Once built, you cam evaluate the output of the decoder together with
    the next input and the next state with the PointingSoftmaxDecoder.step()
    method, that takes as arguments:
      time: a unit `Tensor` representing the 0-based index of the current timestep.
      inp: a `2D Tenros` of shape `[batch_size, input_size]` representing the
        current input.
      state: a 2 items tuple of elements. The first one is a `2D Tensor of shape
        `[batch_size, cell_output_size]` and represents the recurrent cell output.
        The second is the recurrent cell state and its format depends on the cell.

    The method returns:
      output: a `2D Tensor` of shape `[batch_size, shortlist_size + timesteps]`
        representing the current output of the decoder. It represents a probability
        distribution over the items in the shortlist (the first items) and the
        symbols in the input sequence. So, please, note that the length of this
        tensor along the last axis may actually change from batch to batch.
      next_inp: the next `2D Tensor` to be provided as input to the same method.
      next_state: the next tuple to be passed as the decoder state at the next step.
      finished: a `1D Tensor` of `tf.bool` items of shape `[batch_size]` that says,
        for each sequence in the batch, if it should be considered as finished or not.

    *NOTE* that the `inp` argument for the first step can be built with the method
    PointingSoftmaxDecoder.init_input(), and the same holds for the `state` argument
    with the PointingSoftmaxDecoder.init_state() one.
    """

    # pylint: disable=R0913
    def __init__(self, cell, location_softmax, pointing_output,
                 input_size, decoder_inputs=None,
                 trainable=True, name=None, **kwargs):
        """Initializes a new PointingSoftmaxDecoder instance.

        See the class documentation for the escription of all the arguments.
        """
        super(PointingSoftmaxDecoder, self).__init__(
            trainable=trainable, name=name, **kwargs)
        self._cell = cell
        self._loc = location_softmax
        self._out = pointing_output
        self._inp_size = input_size

        if decoder_inputs is not None:
            tensors = tf.transpose(decoder_inputs, [1, 0, 2])
            dtype = tensors.dtype
            size = tf.shape(tensors)[0]
            element_shape = tensors.get_shape()[1:]
            tensor_array = tf.TensorArray(dtype=dtype, size=size, element_shape=element_shape)
            decoder_inputs = tensor_array.unstack(tensors)
        self._inputs_ta = decoder_inputs

        # infer the batch/location size from the `states` tensor
        # of the attention layer of the injected location softmax.
        states = self._loc.attention.states
        self._batch_size = utils.get_dimension(states, 0)
        self._loc_size = utils.get_dimension(states, 1)

    def init_input(self):
        input_shape = tf.stack([self._batch_size, self._inp_size])
        init_input = tf.zeros(input_shape, dtype=tf.float32)
        return init_input

    def init_state(self):
        """A tuple of tensors.

        The state of a pointing softmax decoder is made of a 2 elements tuple
        carring the output and the inner state of the internal recurrent cell
        (which can be a structure of tensors iteself).
        """
        cell_output = tf.zeros(tf.stack([self._batch_size, self._cell.output_size]))
        cell_state = self._cell.zero_state(self._batch_size, tf.float32)
        return (cell_output, cell_state)

    def zero_output(self):
        return self._out.zero_output(
            self._batch_size, self._loc_size)

    def next_inp(self, time, output):
        """Returns the next input.

        Arguments:
          time: a `int` or unit `Tensor` representing the current timestep.
          output: a `2D Tensor` of shape `[batch_size, output_size]` representing
            the current output.

        *NOTE* that at time `t+1` the desired decoder input is the output
        from the previous step, `t`, it means that at timestep `t` the next
        input is the desired output for the very same timestep, if decoder
        inputs have been provided -- otherwise is just the current output.
        """
        if self._inputs_ta:
            output = tf.cond(
                time < self._inputs_ta.size(),
                lambda: self._inputs_ta.read(time),
                lambda: self.zero_output())  # pylint: disable=W0108
        next_inp = ops.fit(output, self._inp_size)
        return next_inp

    def finished(self, time):
        """flags the finished sequence in the batch."""
        flag = False
        if self._inputs_ta:
            flag = (time + 1) >= self._inputs_ta.size()
        return tf.tile([flag], [self._batch_size])

    def step(self, time, inp, state):
        """Pointing softmax decoder step.

        Arguments:
          time: a unit `Tensor` representing the 0-based index of the current timestep.
          inp: a `2D Tenros` of shape `[batch_size, input_size]` representing the
            current input.
          state: a 2 items tuple of elements. The first one is a `2D Tensor of shape
            `[batch_size, cell_output_size]` and represents the recurrent cell output.
            The second is the recurrent cell state and its format depends on the cell.

        Returns:
          output: a `2D Tensor` of shape `[batch_size, shortlist_size + timesteps]`
            representing the current output of the decoder. It represents a probability
            distribution over the items in the shortlist (the first items) and the
            symbols in the input sequence. So, please, note that the length of this
            tensor along the last axis may actually change from batch to batch.
          next_inp: the next `2D Tensor` to be provided as input to the same method.
          next_state: the next tuple to be passed as the decoder state at the next step.
          finished: a `1D Tensor` of `tf.bool` items of shape `[batch_size]` that says,
            for each sequence in the batch, if it should be considered as finished or not.

        *NOTE* that the `inp` argument for the first step can be built with the method
        PointingSoftmaxDecoder.init_input(), and the same holds for the `state` argument
        with the PointingSoftmaxDecoder.init_state() one.
        """
        # Unpack the state into:
        # `in_cell_out`: the cell output at the previous step,
        # `in_cell_state`: the cell inner stata at the previous step.
        in_cell_out, in_cell_state = state

        # Use the previous cell output `in_cell_out` as a query
        # to compute the location scores and the attention context.
        location, attention = self._loc(in_cell_out)

        # Concatenate the previous cell output, the attention context
        # and the current input as the cell current input and feed the
        # cell to compute the current cell output and the current cell state
        # which will be packed in the `next_state` tuple.
        cell_input = tf.concat([in_cell_out, attention, inp], axis=1)
        cell_out, cell_state = self._cell(cell_input, in_cell_state)
        next_state = (cell_out, cell_state)

        # Project the cell output into the pointing softmax output layer
        # together with the attention and the location tensor.
        output = self._out(cell_out, location, attention)

        # Get the next input: if the decoder inputs have been provided,
        # returns the (time + 1)-th if available, otherwise, fit the
        # current output to the input_size.
        next_inp = self.next_inp(time, output)

        # Get the termination flag tensor for the current timestep.
        finished = self.finished(time)

        # Return the outputs.
        return output, next_inp, next_state, finished


class TerminationHelper(object):  # pylint: disable=R0903
    """Helps the termination for a loop over a batch of sequences.

    Given the current time of the loop and the current output, returns a
    tensor of boolean flags indicating for which sequence in the batch, the
    end has been reached.

    Arguments:
      lengths: a `Tensor` of rank `0D` or `1D`, with shape [batch_size]. Is the hard
        limit of the number of iteration for each sequence in the batch. If a `0D` tensor
        is passed, this limit will be applied to all the sequences.
      EOS: an (optional) `int` representing the index value for the `end-of-sequence`
        symbol. The class field `EOS` can be used as the default index for the
        `end-of-sequence`, so that a certain sequence is considered as terminated if
        the current emission is equal to such symbol.

    The i-th sequence in the batch is considered as terminated if the current step
    is greater or equal the number of steps allowed (defined in the `lengths` input
    argument) and if the `argmax` over the output probability distribution ends up
    in the class that has id equal to the `EOS` symbol (if provided).
    """

    EOS = 0

    def __init__(self, lengths, EOS=None):
        """Initialise a new TerminationHelper instance."""
        self._EOS = EOS  # pylint: disable=I0011,C0103
        if lengths is None:
            raise ValueError("`lengths` must be a 0D or 1D Tensor, not `None`")
        self._lengths = lengths

    def finished(self, time, output):
        """Check which sentences are finished.

        Arguments:
          time: a `Tensor` of rank `0D` (i.e. a scalar) with the 0-based value of the
            current step in the loop.
          output: a `Tensor` of rank `2D` and shape `[batch_size, num_classes]` representing
            the current output of the model, i.e. abatch of probability distribution estimations
            over the output classes.

        Returns:
          a `Tensor` of shape `[batch_size]` of `tf.bool` elements, indicating for each
          position if the corresponding sequence has terminated or not. A sequence is
          has terminated if the current step is greater or equal the number of steps allowed
          (defined in the `lengths` input argument) and if the `argmax` over the output
          probability distribution ends up in the class that has id equal to the `EOS` symbol
          (if provided).
        """
        length = time + 1
        finished = tf.greater_equal(length, self._lengths)
        if finished.get_shape().ndims == 0:
            batch = [utils.get_dimension(output, 0)]
            finished = tf.tile([finished], batch)
        if self._EOS is not None:
            ids = tf.cast(tf.argmax(output, axis=-1), tf.int32)
            eos = tf.equal(ids, self._EOS)
            finished = tf.logical_or(finished, eos)
        return finished


class DynamicDecoder(Layer):
    """Dynamic decoder implementation.

    The DynamicDecoder class implements a dynamic decoding loop around
    a DecoderBase instance. To build a DynamicDecoder instance you must
    pass to the initializer the following arguments:
      decoder: any object implementing the `DecoderBase` type and implementing
        the single decoding step logics.
      helper: any object implementing the `TerminationHelper` type.
      parallel_iterations: (Default: 10). The number of iterations to run in parallel.
        Those operations which do not have any temporal dependency and can be run in
        parallel, will be. This parameter trades off time for space. Values >> 1 use more
        memory but take less time, while smaller values use less memory but computations
        take longer.
      pad_to: a (optional) `int` or unit `Tensor` indicating the length along the time
        axis to pad to. If `None`, the output sequence is terminated as soon as the decoding
        operation reaches its natural termination. Please note that the maximum length (which
        is determined by the termination helper, `helper`) can be greater than the padding
        length.
      swap_memory: Transparently swap the tensors produced in forward inference but needed
        for back prop from GPU to CPU. This allows training RNNs which would typically not
        fit on a single GPU, with very minimal (or no) performance penalty.
        trainable: if `True`, the created variables will be trainable.
        scope: VariableScope for the created subgraph.
      name: (optional) a string representing the name of the instance.

    Invoking the (parameterless) `decode()` argument, will return a 2 element tuple
    made of the following items:
      output: a `3D Tensor` of shape `[batch_size, sequence_length, output_size]`
        representing the output of the decoding phase.
      state: the final state of the injected decoder -- which can be an arbitrary
        structure of `Tensor`s.
    """

    # pylint: disable=R0913
    def __init__(self, decoder, helper, pad_to=None, parallel_iterations=10,
                 swap_memory=False, name='DynamicDecoder', **kwargs):
        """Initializes a DynamicDecoder instance."""
        self._decoder = decoder
        self._helper = helper
        self._pad_to = pad_to
        self._parallel_iterations = parallel_iterations
        self._swap_memory = swap_memory
        super(DynamicDecoder, self).__init__(
            trainable=decoder.trainable, name=name, **kwargs)

    def output(self, output, finished):
        """Filter the output tensor w.r.t. which sequence in the batch is finished."""
        zoutput = self._decoder.zero_output()
        return tf.where(finished, zoutput, output)

    # pylint: disable=R0913
    def body(self, time, inp, state, finished, output_ta):
        """Body of the dynamic decoding phase."""
        # invoke the decoder step.
        output, next_inp, next_state, decoder_finished = self._decoder.step(time, inp, state)

        # check the termination status and filter the output.
        next_finished = tf.logical_or(finished, decoder_finished)
        next_finished = tf.logical_or(next_finished, self._helper.finished(time, output))
        output = self.output(output, finished)

        output_ta = output_ta.write(time, output)
        ntime = tf.add(time, 1)

        return ntime, next_inp, next_state, next_finished, output_ta

    # pylint: disable=W0613,I0011,R0913
    # disable unused arguments (needed for the loop).
    def cond(self, time, inp, state, finished, output_ta):
        """Logical contidion for termination."""
        continuation = tf.logical_not(tf.reduce_all(finished))
        if self._pad_to is None:
            return continuation
        padding = time < self._pad_to
        return tf.logical_or(continuation, padding)

    # pylint: disable=W0221,I0011
    # disable the changed signature of the method.
    def _call_helper(self):
        time = tf.constant(0, dtype=tf.int32)
        inp = self._decoder.init_input()
        state = self._decoder.init_state()
        finished = tf.tile([False], [utils.get_dimension(inp, 0)])
        output_ta = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        loop_vars = [time, inp, state, finished, output_ta]
        results = tf.while_loop(
            cond=self.cond, body=self.body, loop_vars=loop_vars,
            parallel_iterations=self._parallel_iterations,
            swap_memory=self._swap_memory)
        output_ta = results[-1]
        output = output_ta.stack()
        output = tf.transpose(output, [1, 0, 2])
        state = results[2]
        return output, state

    def decode(self):
        """Run the dynamic decoding."""
        return self.__call__()
