"""Additional utilities to improve the usage of the `tf.Graph`."""

import copy

import tensorflow as tf


class GraphHelperMap(object):
    """Map a `tf.Graph` instance to a corresponding `GraphHelper`.

    This class is intended to implement a single point of
    access map to bing `tf.Graph` instances to corresponding
    `GraphHelper` instances. It provides only one static method
    `get()` that returns the proper `GraphHelper` instance for
    a given graph object -- or the default one, if `None` is
    passed as an argument.

    The class has only one internal field `_MAP`, which is a
    dictionary used to map the `tf.Graph` instances, acting
    as keys, to the corresponding `GraphHelper`, as values.

    Example:
    >>> import tensorflow as tf
    >>> from liteflow import graphutils
    >>> ghelper = graphutils.GraphHelperMap.get()
    >>> assert ghelper.graph == tf.get_default_graph()
    ```
    """

    _MAP = {}

    @staticmethod
    def get(graph=None):
        """Get the GraphHelper instance bound to a given graph.

        If the graph has not any helper bound to it, a brand
        new one will be created and returned. Such helper will
        be consistently returned all the times the method is
        invoked with the same graph instance as argument.

        Arguments:
          graph: a `tf.Graph` instance, if `None` the default
            graph will be used.

        Returns:
          an instance of `GraphHelper`.
        """
        key = graph or tf.get_default_graph()
        if key not in GraphHelperMap._MAP:
            helper = GraphHelper(key)
            GraphHelperMap._MAP[key] = helper
        return GraphHelperMap._MAP[key]


class GraphHelperKeys(object):
    """Keys for the GraphHelper maps."""
    INPUTS_MAP = 'inputs'
    TARGETS_MAP = 'targets'
    OUTPUTS_MAP = 'outputs'


class GraphHelper(object):
    """Aggregate floags and maps for the current graph."""

    def __init__(self, graph):
        """Initialize a new helper for a given graph.

        Arguments:
          graph: a `tf.Graph` instance.

        Raises:
          ValueError: if `graph` is `None`.
        """

        if graph is None:
            raise ValueError('`graph` cannot be `None`.')
        self._graph = graph
        self._trainable = False
        self._maps = {
            GraphHelperKeys.INPUTS_MAP: {},
            GraphHelperKeys.TARGETS_MAP: {},
            GraphHelperKeys.OUTPUTS_MAP: {}
        }

    @property
    def graph(self):
        """The  associated graph."""
        return self._graph

    @property
    def trainable(self):
        """A boolean flag indicating if the graph is trainable."""
        return self._trainable

    @trainable.setter
    def trainable(self, value):
        self._trainable = value

    def get_all_map_keys(self):
        """Get a tuple with all the map keys."""
        return tuple(self._maps.iterkeys())

    def get_map(self, key):
        """Get a copy of the map associated with the given key.

        Arguments:
          key: a `str` which is a key for a map.

        Returns:
          a dictionary which is a copy of the map associated
            to the given `key`.
        """
        return copy.copy(self._maps[key])

    def get_map_ref(self, key):
        """Get a reference to the map associated with the given key.

        Arguments:
          key: a `str` which is a key for a map.

        Returns:
          a dictionary which is the actual map associated
            to the given `key`.
        """
        return self._maps[key]

    def put_in_map(self, key, item_key, item_value):
        """Add a `item_key`, `item_value` pair to the map associated to `key`."""
        self._maps[key][item_key] = item_value
