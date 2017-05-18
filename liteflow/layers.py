"""Base contract for lite layers implementation."""

import abc
import functools
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

    @property
    def variables(self):
        """Return all the variables used by the layer."""
        return self._variables

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
                    scope, default_name=self._default_scope()).gen)

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
        with tf.variable_scope(self._scope) as scope:
            if self._reuse:
                scope.reuse_variables()
            result = self._call_helper(*args, **kwargs)
            self._reuse = True
        return result

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

    def _call(self, query):  # pylint: disable=I0011,W0221
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

    def call(self, query):  # pylint: disable=I0011,W0221
        """Calculate the attention scores over `self.states` for the given queries.

        Arguments:
          query: a 2-D Tensor of shape [batch_size, query_size]; the last dimension
            must be statically determined.

        Returns:
          A 2-D tensor of shape [batch, timesteps] where `timesteps` is the second
            dimension of the `self.states` tensor.

        Raises:
          ValueError: if the last dimension of the `query` argument
            is not statically determined.
        """
        return super(BahdanauAttention, self).__call__(query)


class LocationSoftmax(Layer):
    """Implements a PointingSoftmax over a set of attention states.

    The logics implemented in this class is described in this paper by Caglar
    Gulcehre et al., https://arxiv.org/abs/1603.08148

    Given a set of attention states as a 3D tensor of shape [batch_size, timesteps,
    state_size] and an attention mechanism over them, this layer returns a 2D tensor of
    shape [batch_size, timesteps] where the i-th element of the batch represents a
    probability distribution over the states of the i-th element of the input batch.
    Since the sequences in the input batch can have actual different lengths, such
    distributions are computed only over the actual elements.

    An invocation to this layer, given a 2D tensor, called `query`, of shape
    [batch_size, query_size], returns a tuple of two tensors:
      `location_softmax` is a 2D tensor of shape [batch_size, timesteps] that
        represent a probability distribution over the attention states;
      `attention_context` is a 2D tesnor of shape [batch_size, state_size] that
        represent the attention vector for the query over the attention states where
        the `location_softmax` tensor has been used as the coeffiecient vector.
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

    def _build(self, *args, **kwargs):
        if not self._attention.built:
            self._attention.build()

    def _call(self, query):  # pylint: disable=I0011,W0221
        activations = self._attention(query)
        maxlen = utils.get_dimension(activations, -1)
        if self._sequence_length is not None:
            mask = tf.cast(tf.sequence_mask(self._sequence_length, maxlen), tf.float32)
        else:
            mask = None
        weights = ops.softmax(activations, mask)
        eweights = tf.expand_dims(weights, axis=2)
        context = tf.reduce_sum(self._attention.states * eweights, axis=1)
        return weights, context

    def call(self, query):  # pylint: disable=I0011,W0221
        """Calculate the location softmax and attention context over a set of input states.

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
        return super(LocationSoftmax, self).__call__(self, query)


class PointingSoftmaxOutput(Layer):
    """Implements the output layer for a PointingDecoder.

    The logics implemented in this class is described in this paper by Caglar
    Gulcehre et al., https://arxiv.org/abs/1603.

    An invocation to this layer acceprs as arguments:
      decoder_out: a 2D tensor of shape [batch_size, decoder_out_size] representing
        the output of some decoding operation
      location_softmax: a 2D tensor of shape [batch_size, timesteps] that represents the
        location softmax (i.e. a probability distribution) over a set of attention states
      attention_context: a 2D tensor of shape [batch_size, state_size] representing the
      attention context vector for `decoder_out` w.r.t. to a set of input states calculated
      with the `location_softmax` as coeffiecients

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
        self._switch_kernel = None
        self._switch_bias = None
        self._shortlist_kernel = None
        self._shortlist_bias = None

    def _build(self, *args, **kwargs):
        self._switch_kernel = tf.get_variable(
            name=self._SWITCH_KERNEL_NAME,
            shape=[self._decoder_out_size + self._state_size, 1],
            trainable=self.trainable)
        self._switch_bias = tf.get_variable(
            name=self._SWITCH_BIAS_NAME,
            shape=[1],
            trainable=self.trainable)
        self._shortlist_kernel = tf.get_variable(
            self._SHORTLIST_KERNEL_NAME,
            shape=[self._decoder_out_size, self._shortlist_size],
            trainable=self.trainable)
        self._shortlist_bias = tf.get_variable(
            name=self._SHORTLIST_BIAS_NAME,
            shape=[self._shortlist_size],
            trainable=self.trainable)

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

    def _call(self, decoder_out, location_scores, attention_context):  # pylint: disable=I0011,W0221
        switch_in = tf.concat([decoder_out, attention_context], axis=1)
        switch = tf.matmul(switch_in, self._switch_kernel) + self._switch_bias
        switch = tf.nn.sigmoid(switch)
        shortlist = tf.matmul(decoder_out, self._shortlist_kernel) + self._shortlist_bias
        shortlist = tf.nn.sigmoid(shortlist)
        output = tf.concat([switch * shortlist, (1 - switch) * location_scores], axis=1)
        return output

    def call(self, decoder_out, location_softmax, attention_context):  # pylint: disable=I0011,W0221
        """Computes the output of a pointing decoder.

        Arguments:
          decoder_out: a 2D tensor of shape [batch_size, decoder_output_size]
            representing the output of the recurrent decoder.
          location_softmax: a 2D tensor of shape [batch_size, timesteps] representing
            a probability distribution over the attention states.
          attention_context: a 2D tensor of shape [batch_size, state_size] representing
            the attention context vector over the attention states for `decoder_outout`
            calculated with `location_softmax` as the coefficients.

        Returns:
          a 2D tensor of shape [batch_size, shortlist_size + timesteps] representing
          a probability distribution over the output shortlist **and** the attention states.
        """
        return super(PointingSoftmaxOutput, self).__call__(
            decoder_out, location_softmax, attention_context)


class PointingDecoder(Layer):
    """PointingDecoder layer.

    This class implements a decoder that, in a sequence-to-sequence scenario
    has the capability of gnerating symbols from a shortlist (i.e. an output
    vocabulary) or pointing a position within a sequence of elements so that
    the propert symbol can be copied from the input.
    """

    def __init__(self, decoder_cell,
                 location_softmax, pointing_softmax_output,
                 out_sequence_length,
                 emit_out_feedback_fit=None,
                 emit_out_init=None,
                 cell_out_init=None, cell_state_init=None,
                 parallel_iterations=None, swap_memory=False,
                 trainable=True, scope='PointingDecoder'):
        """Initialize a new instance.

        Arguments:
          decoder_cell: a cell used as a decoder, implementing `tf.contrib.rnn.RNNCell`.
          location_softmax: a LocationSoftmax instance.
          pointing_softmax_output: a PointingSoftmaxOutput instance
          out_sequence_length: a 1D tensor of shape [batch_size] with the actual length
            of each output (i.e. decoded) sequence in the batch.
          emit_out_feedback_fit: since the output can have different dimension for
            each batch, this function should process the output and fit it to a fixed
            length so that it can be fed back to the decoder cell, concatenated with the
            previous output and the attention context. If None, the output is fed
            as-is but this scenario will surely work with fixed length input states.
          emit_out_init: a 2D tensor of shape [batch_size, shortlist_size + timesteps]
            representing the initialization value for the output.
          cell_out_init: a 2D tensor of shape [batch_size, decoder_out_size]
            representing the initialization value for the decoder output.
          cell_state_init: a 2D tensor of shape [batch_size, decoder_cell_state]
            representing the initialization value for the decoder cell state.
          parallel_iterations: (Default: 32). The number of iterations to run in parallel.
            Those operations which do not have any temporal dependency and can be run in
            parallel, will be. This parameter trades off time for space. Values >> 1 use more
            memory but take less time, while smaller values use less memory but computations
            take longer.
          swap_memory: Transparently swap the tensors produced in forward inference but needed
            for back prop from GPU to CPU. This allows training RNNs which would typically not
            fit on a single GPU, with very minimal (or no) performance penalty.
          trainable: if `True`, the created variables will be trainable.
          scope: VariableScope for the created subgraph;.
        """

        super(PointingDecoder, self).__init__(trainable=trainable, scope=scope)
        self._decoder_cell = decoder_cell
        self._sequence_length = out_sequence_length
        self._location_softmax = location_softmax
        self._pointing_softmax_output = pointing_softmax_output
        self._emit_out_init = emit_out_init
        self._emit_out_feedback_fit = emit_out_feedback_fit
        self._cell_out_init = cell_out_init
        self._cell_state_init = cell_state_init
        self._parallel_iterations = parallel_iterations
        self._swap_memory = swap_memory
        self._batch_size = None
        self._pointing_size = None

    @property
    def emit_out_init(self):
        """Initialization for the output signal."""
        return self._emit_out_init

    @property
    def cell_out_init(self):
        """Initialization for the decoder cell output signal."""
        return self._cell_out_init

    @property
    def cell_state_init(self):
        """Initialization for the cell state signal."""
        return self._cell_state_init

    def _build(self, *args, **kwargs):
        if not self._location_softmax.built:
            self._location_softmax.build()
        if not self._pointing_softmax_output.built:
            self._pointing_softmax_output.build()

        states = self._location_softmax.attention.states
        self._batch_size = utils.get_dimension(states, 0)
        self._pointing_size = utils.get_dimension(states, 1)
        if self._emit_out_init is None:
            self._emit_out_init = self._pointing_softmax_output.zero_output(
                self._batch_size, self._pointing_size)

        if self._cell_out_init is None:
            cell_out_shape = tf.stack([self._batch_size, self._decoder_cell.output_size])
            self._cell_out_init = tf.zeros(cell_out_shape)

        if self._cell_state_init is None:
            self._cell_state_init = self._decoder_cell.zero_state(
                self._batch_size, dtype=tf.float32)

        if self._emit_out_feedback_fit is None:
            self._emit_out_feedback_fit = lambda tensor: tensor

    def _loop_fn(self, time, cell_output, cell_state, loop_state):

        # Determin how many sequences are actually over and define
        # a flag to check if all of them have been fully scanned.
        if self._sequence_length is None:
            batch_dim = utils.get_dimension(cell_output, 0)
            elements_finished = tf.ones([batch_dim], tf.bool)
        else:
            elements_finished = (time >= self._sequence_length)

        # Declare and initialize to `None` all the return arguments.
        next_cell_input = None
        next_cell_state = None
        emit_output = None
        next_loop_state = (None, None)
        feedback = None

        if cell_output is None:
            # If the `cell_output` is None, it means we are at the very
            # first iteration. In this case, we need to initialize the
            # return arguments to their initial values.
            cell_output = self._cell_out_init
            next_cell_state = self._cell_state_init
            emit_output = self._emit_out_init
        else:
            pointing_softmax, attention_context = loop_state
            next_cell_state = cell_state
            emit_output = self._pointing_softmax_output(
                cell_output, pointing_softmax, attention_context)

        # Evaluate the pointing scores and the attention context for
        # the next step and pack them into the loop state.
        pointing_softmax, attention_context = self._location_softmax(cell_output)
        next_loop_state = (pointing_softmax, attention_context)

        # Fit the output to be fed back.
        feedback = self._emit_out_feedback_fit(emit_output)

        # Pack the next input for the decoder cell. Such input is
        # the concatenation of the current cell output (since it is
        # a recurrent scenario), the current attention_context and
        # the feedback coming from the output signal.
        next_cell_input = tf.concat(
            [cell_output, attention_context, feedback],
            axis=1)

        return (elements_finished, next_cell_input, next_cell_state,
                emit_output, next_loop_state)

    def _call(self):    # pylint: disable=I0011,W0221
        outputs_ta, _, _ = tf.nn.raw_rnn(self._decoder_cell, self._loop_fn)
        outputs = outputs_ta.pack()
        return outputs

    def call(self):    # pylint: disable=I0011,W0221
        """Decode using pointing softmax.

        Decode a 3D tensor os shape [batch_size, length, shortlist_size + timesteps].
        Each outout vector will represent a probability distribution over a set of
        symbols representing a shortlist of symbols and a list of positions over
        the a set of attention states.

        Returns:
          a 3D tensor of shape [batch_size, length, shortlist_size + timesteps]
          representing the decoder output.
        """
        return super(PointingDecoder, self).__call__()


def pointing_decoder(attention_states,
                     attention_inner_size,
                     decoder_cell,
                     shortlist_size,
                     attention_sequence_length=None,
                     output_sequence_length=None,
                     emit_out_feedback_size=None,
                     parallel_iterations=None,
                     swap_memory=False,
                     trainable=True):
    """Creates a PointingDecoder layer.

    Arguments:
      attention_states: a 3D Tensor of shape [batch_size, timesteps, state_size]
        representing the states on which the attention scores will be computed;
        the third dimension of the tensor must be statically determined.
      inner_size: `int` representing the inner attention size of a BahdanauAttention.
      decoder_cell: a cell used as a decoder, implementing `tf.contrib.rnn.RNNCell`.
      shortlist_size:
      attention_sequence_length: a 1D Tensor of shape [batch_size] where each element
        represent the number of actual meaningful elements in each sequence of the
        attention states.
      output_sequence_length: a 1D Tensor of shape [batch_size] where each element
        represent the number of actual meaningful elements in each sequence of the
        output tensor.
      emit_out_feedback_fit: since the output can have different dimension for
        each batch, this function should process the output and fit it to a fixed
        length so that it can be fed back to the decoder cell, concatenated with the
        previous output and the attention context. If None, the output is fed
        as-is but this scenario will surely work with fixed length input states.
      parallel_iterations: (Default: 32). The number of iterations to run in parallel.
        Those operations which do not have any temporal dependency and can be run in
        parallel, will be. This parameter trades off time for space. Values >> 1 use more
        memory but take less time, while smaller values use less memory but computations
        take longer.
      swap_memory: Transparently swap the tensors produced in forward inference but needed
        for back prop from GPU to CPU. This allows training RNNs which would typically not
        fit on a single GPU, with very minimal (or no) performance penalty.
        trainable: if `True`, the created variables will be trainable.
      trainable: if True, variables will be trainable;

    Returns:
      a new instance of the PointingDecoder layer.
    """

    attention = BahdanauAttention(
        states=attention_states,
        inner_size=attention_inner_size,
        trainable=trainable)

    location = LocationSoftmax(
        attention=attention,
        sequence_length=attention_sequence_length)

    state_size = attention_states.shape[-1].value
    output = PointingSoftmaxOutput(
        shortlist_size=shortlist_size,
        decoder_out_size=decoder_cell.output_size,
        state_size=state_size,
        trainable=trainable)

    fit = None
    if emit_out_feedback_size is not None:
        fit = functools.partial(ops.fit, width=emit_out_feedback_size)
    decoder = PointingDecoder(
        decoder_cell=decoder_cell,
        location_softmax=location,
        out_sequence_length=output_sequence_length,
        pointing_softmax_output=output,
        emit_out_feedback_fit=fit,
        parallel_iterations=parallel_iterations,
        swap_memory=swap_memory)
    return decoder
