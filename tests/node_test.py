import unittest
from Node import Node

class TestNode(unittest.TestCase):

    def test_node_initialization(self):
        """Test if a Node is initialized correctly."""
        node = Node(id=1, ttl=20, active_status=True)
        self.assertEqual(node.id, 1)
        self.assertEqual(node.ttl, 20)
        self.assertTrue(node.is_active)
        self.assertEqual(node.finger_table['successors'], [])
        self.assertEqual(node.finger_table['predecessors'], [])

    def test_set_active_status(self):
        """Test setting a node's active status."""
        node = Node(id=2, ttl=30, active_status=False)
        node.set_active_status(True)
        self.assertTrue(node.is_active)

    def test_reset_timer(self):
        """Test resetting a node's timer."""
        node = Node(id=3, ttl=10, active_status=True)
        node.ttl = 5
        node.reset_timer()
        self.assertEqual(node.ttl, 10)


if __name__ == "__main__":
    unittest.main()
