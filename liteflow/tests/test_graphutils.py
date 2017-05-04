"""Test module for the `dket.graphutils` module."""

import unittest

import tensorflow as tf

from liteflow import graphutils


class GraphHelperMapTest(unittest.TestCase):
    """Test case for the `liteflow.graphutils.GraphHelperMap` class."""

    def test_default(self):
        """Test the .get() method with `None` or default graph."""

        graph = tf.get_default_graph()
        helper = graphutils.GraphHelperMap.get()
        self.assertEquals(graph, helper.graph)
        self.assertEquals(helper, graphutils.GraphHelperMap.get(graph))

    def test_graph(self):
        """Test the .get() method with a `tf.Graph` object."""

        graph = tf.Graph()
        helper = graphutils.GraphHelperMap.get(graph)
        self.assertEquals(graph, helper.graph)
        self.assertEquals(helper, graphutils.GraphHelperMap.get(graph))

    def test_graph_as_default(self):
        """Test the .get() method within a graph context."""

        def_graph = tf.get_default_graph()
        def_helper = graphutils.GraphHelperMap.get()
        self.assertEquals(def_helper.graph, def_graph)
        with tf.Graph().as_default() as ctx_graph:
            ctx_helper = graphutils.GraphHelperMap.get()
            self.assertEquals(ctx_graph, ctx_helper.graph)
            self.assertNotEquals(def_helper, ctx_helper)
            self.assertNotEquals(def_helper.graph, ctx_helper.graph)


class GraphHelperTest(unittest.TestCase):
    """Test case for the `liteflow.graphutils.GraphHelper` class."""

    def test_null_graph(self):
        """Initialize a new helpe with a `None` graph instance."""
        self.assertRaises(ValueError, lambda: graphutils.GraphHelper(None))

    def _test_map(self, key):
        tf.reset_default_graph()
        helper = graphutils.GraphHelper(tf.get_default_graph())
        item_key = 'X'
        item_value = object()

        # Get the map and add a key-value pair; since the
        # map is a copy, if we get the map again, the key-value
        # pair won't be there.
        map_ = helper.get_map(key)
        map_[item_key] = item_value
        self.assertNotIn(item_key, helper.get_map(key))

        # Get the reference to the map and add the pais;
        # since we got the reference to the actual object,
        # the pair will be there even if the map reference
        # and e map returned from the .get_map() invocation
        # are different -- i.e. not the same object.
        rmap = helper.get_map_ref(key)
        rmap[item_key] = item_value
        map_ = helper.get_map(key)
        self.assertNotEquals(id(rmap), id(map_))
        self.assertIn(item_key, map_)

        # Invoking twice the .get_map_ref() method
        # you shall obtain twice the same object, while
        # this is not the case for the get_map() one.
        self.assertEquals(id(helper.get_map_ref(key)), id(helper.get_map_ref(key)))
        self.assertNotEquals(id(helper.get_map(key)), id(helper.get_map(key)))

        # Putting an element in a map from and helper will
        # make it appear in the map. The only way to remove
        # an element from a map is passing through the
        # actual reference to the map.
        helper.put_in_map(key, item_key, item_value)
        self.assertIn(item_key, helper.get_map(key))
        self.assertIn(item_key, helper.get_map_ref(key))
        self.assertEquals(item_value, helper.get_map_ref(key).pop(item_key))
        self.assertNotIn(item_key, helper.get_map(key))
        self.assertNotIn(item_key, helper.get_map_ref(key))


    def test_inputs(self):
        """Test the inputs map."""
        self._test_map(graphutils.GraphHelperKeys.INPUTS_MAP)

    def test_targets(self):
        """Test the targets map."""
        self._test_map(graphutils.GraphHelperKeys.TARGETS_MAP)

    def test_outputs(self):
        """Test the outputs map."""
        self._test_map(graphutils.GraphHelperKeys.TARGETS_MAP)

    def test_loss(self):
        """Test the loss ops map."""
        self._test_map(graphutils.GraphHelperKeys.LOSS_OPS_MAP)

    def test_train(self):
        """Test the train ops map."""
        self._test_map(graphutils.GraphHelperKeys.LOSS_OPS_MAP)

    def test_summary(self):
        """Test the summary ops map."""
        self._test_map(graphutils.GraphHelperKeys.SUMMARY_OPS_MAP)

    def test_eval(self):
        """Test the eval ops map."""
        self._test_map(graphutils.GraphHelperKeys.EVAL_OPS_MAP)

class DefaultTest(unittest.TestCase):
    """Test case for the module level functions.
    
    Remarks: as you can see in the implementation of the
    `liteflow.graphutils` module, such functions don't worth
    unit tests, since they just bounce on the default graph
    helper. Still, tests may be added in the future.
    """

    def test_get_helper(self):
        """Test for the `liteflow.graphutils.get_helper` function."""
        graph = tf.get_default_graph()
        helper = graphutils.get_helper()
        self.assertIsNotNone(helper)
        self.assertEquals(helper.graph, graph)
        self.assertEquals(helper, graphutils.get_helper(graph))

        xgraph = tf.Graph()
        xhelper = graphutils.get_helper(xgraph)
        self.assertIsNotNone(xhelper)
        self.assertEquals(xhelper, graphutils.get_helper(xgraph))

        with xgraph.as_default():
            self.assertEquals(xhelper, graphutils.get_helper())
        
        
if __name__ == '__main__':
    unittest.main()
