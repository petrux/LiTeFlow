"""Utilities for input pipelines."""

import tensorflow as tf


def shuffle(tensors,
            capacity=32,
            min_after_dequeue=64,
            num_threads=1,
            dtypes=None,
            shapes=None,
            seed=None,
            shared_name=None,
            name='shuffle'):
    """Wrapper around a `tf.RandomShuffleQueue` creation.

    Return a dequeue op that dequeues elements from `tensors` in a
    random order, through a `tf.RandomShuffleQueue` -- see for further
    documentation.

    Arguments:
      tensors: an iterable of tensors.
      capacity: (Optional) the capacity of the queue; default value set to 32.
      num_threads: (Optional) the number of threads to be used fo the queue runner;
        default value set to 1.
      min_after_dequeue: (Optional) minimum number of elements to remain in the
        queue after a `dequeue` or `dequeu_many` has been performend,
        in order to ensure better mixing of elements; default value set to 64.
      dtypes: (Optional) list of `DType` objects, one for each tensor in `tensors`;
        if not provided, will be inferred from `tensors`.
      shapes: (Optional) list of shapes, one for each tensor in `tensors`.
      seed: (Optional) seed for random shuffling.
      shared_name: (Optional) If non-empty, this queue will be shared under
        the given name across multiple sessions.
      name: Optional name scope for the ops.

    Returns:
      The tuple of tensors that was randomly dequeued from `tensors`.
    """

    tensors = list(tensors)
    with tf.name_scope(name, tensors):
        dtypes = dtypes or list([t.dtype for t in tensors])
        queue = tf.RandomShuffleQueue(
            seed=seed,
            shared_name=shared_name,
            name='random_shuffle_queue',
            dtypes=dtypes,
            shapes=shapes,
            capacity=capacity,
            min_after_dequeue=min_after_dequeue)
        enqueue = queue.enqueue(tensors)
        runner = tf.train.QueueRunner(queue, [enqueue] * num_threads)
        tf.train.add_queue_runner(runner)
        dequeue = queue.dequeue()
        return dequeue


def shuffle_batch(tensors,
                  batch_size,
                  capacity=32,
                  num_threads=1,
                  min_after_dequeue=64,
                  dtypes=None,
                  shapes=None,
                  seed=None,
                  enqueue_many=False,
                  dynamic_pad=True,
                  allow_smaller_final_batch=False,
                  shared_name=None,
                  name='shuffle_batch'):
    """Create shuffled and padded batches of tensors in `tensors`.

    Dequeue elements from `tensors` shuffling, batching and dynamically
    padding them. First a `tf.RandomShuffleQueue` is created and fed with
    `tensors` (using the `dket.input.shuffle` function); the dequeued tensors
    shapes are then set and fed into a `tf.train.batch` function that provides
    batching and dynamic padding.


    Arguments:
      tensors: an iterable of tensors.
      batch_size: an `int` representing th batch size.
      capacity: (Optional) the capacity of the queues; default value set to 32.
      num_threads: (Optional) the number of threads to be used fo the queue runner;
        default value set to 1.
      min_after_dequeue: (Optional) minimum number of elements to remain in the
        shuffling queue after a `dequeue` or `dequeu_many` has been performend,
        in order to ensure better mixing of elements; default value set to 64.
      dtypes: (Optional) list of `DType` objects, one for each tensor in `tensors`;
        if not provided, will be inferred from `tensors`.
      shapes: (Optional) list of shapes, one for each tensor in `tensors`.
      seed: (Optional) seed for random shuffling.
      enqueue_many: Whether each tensor in tensors is a single example.
      dynamic_pad: Boolean. Allow variable dimensions in input shapes.
        The given dimensions are padded upon dequeue so that tensors within
        a batch have the same shapes.
      allow_smaller_final_batch: (Optional) Boolean. If True, allow the final
        batch to be smaller if there are insufficient items left in the queue.
      shared_name: if set, the queues will be shared under the given name
        across different sessions.
      name: scope name for the given ops.

    Returns:
      A batch of tensors from `tensors`, shuffled and padded.
    """

    tensors = list(tensors)
    with tf.name_scope(name, tensors):
        dtypes = dtypes or list([t.dtype for t in tensors])
        shapes = shapes or list([t.get_shape() for t in tensors])
        inputs = shuffle(tensors,
                         seed=seed,
                         dtypes=dtypes,
                         capacity=capacity,
                         num_threads=num_threads,
                         min_after_dequeue=min_after_dequeue,
                         shared_name=shared_name,
                         name='shuffle')

        # fix the shapes
        for tensor, shape in zip(inputs, shapes):
            tensor.set_shape(shape)

        minibatch = tf.train.batch(
            tensors=inputs,
            batch_size=batch_size,
            num_threads=num_threads,
            capacity=capacity,
            dynamic_pad=dynamic_pad,
            allow_smaller_final_batch=allow_smaller_final_batch,
            shared_name=shared_name,
            enqueue_many=enqueue_many,
            name='batch')
        return minibatch
