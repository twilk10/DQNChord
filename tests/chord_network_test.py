import unittest
from Chord import ChordNetwork

class TestChordNetwork(unittest.TestCase):

    def setUp(self):
        """Set up a Chord network before each test."""
        self.network = ChordNetwork(size=10, r=2, bank_size=20) 

    # def test_initialize_node_bank(self):
    #     """Test if the node bank is correctly initialized."""
    #     self.assertEqual(len(self.network.node_bank), 20)
    #     self.assertTrue(self.network.node_bank[0].is_active)  # Node 0 should be active
    
    def test_basic_lookup(self):
        """
        Test a lookup for a key that exists in the network and is directly in the finger table.
        """
        key = 3  # Assuming key 3 falls directly in the finger table of node 0
        result = self.network.lookup(key)
        self.assertIsNotNone(result, "Lookup should return a node ID.")

    # def test_lookup_multiple_hops(self):
    #     """
    #     Test a lookup that requires traversing multiple nodes.
    #     """
    #     key = 7  # Assuming key 7 requires multiple hops
    #     result = self.network.lookup(key)
    #     self.assertIsNotNone(result, "Lookup should return a node ID.")
    #     self.assertEqual(result, 7, "Lookup should find the correct node.")

    def test_key_not_found(self):
        """
        Test a lookup for a key that does not exist in the network.
        """
        key = 100  # Assuming key 100 is out of the range of all node IDs
        result = self.network.lookup(key)
        self.assertIsNone(result, "Lookup for a nonexistent key should return None.")

    def test_edge_case_min_key(self):
        """
        Test a lookup for the smallest key in the network.
        """
        key = 0  # Assuming key 0 exists in the network
        result = self.network.lookup(key)
        self.assertIsNotNone(result, "Lookup should return a node ID.")
        self.assertEqual(result, 0, "Lookup should find the correct node.")

    def test_edge_case_max_key(self):
        """
        Test a lookup for the largest key in the network.
        """
        network_size = len([n for n in self.network.node_bank.values() if n.is_active])
        key = network_size - 1  # Assuming the largest key is equal to network size - 1
        result = self.network.lookup(key)
        # print(f'Node responsible for key {key}: is {result}')
        self.assertIsNotNone(result, "Lookup should return a node ID.")


    def test_successors_and_predecessors(self):
        """Test if successors and predecessors are correctly assigned."""
        id = 1
        actual_predecessor = 0
        actual_successors = [2, 3, 5]
        r = 3 # Max number of successors set as 3 

        node = self.network.node_bank[id]
        self.network.assign_successors_and_predecessors(node, r)
        self.assertIn(actual_predecessor , node.finger_table['predecessors'])
        self.assertEqual(len(node.finger_table['successors']), r)
        self.assertEqual(node.finger_table['successors'], actual_successors)

    # def test_node_join(self):
    #     """Test if a node can successfully join the network."""
    #     node = self.network.node_bank[6]
    #     node.set_active_status(False)  # Ensure the node starts as inactive
    #     self.network.join_network(node)
    #     self.assertTrue(node.is_active)  # Node should be active after joining
    #     self.assertGreater(len(node.finger_table['successors']), 0)

    # def test_node_leave(self):
    #     """Test if a node can successfully leave the network."""
    #     node = self.network.node_bank[3]
    #     node.set_active_status(True)  # Ensure the node starts as active
    #     self.network.leave_network(node)
    #     self.assertFalse(node.is_active)  # Node should be inactive after leaving
    #     self.assertEqual(len(node.finger_table['successors']), 0)

    # def test_random_churn_rate(self):
    #     """Test if random churn rates are assigned correctly."""
    #     rates = {20, 50, 100}
    #     churn_rate = self.network.random_churn_rate()
    #     self.assertIn(churn_rate, rates)

    # def test_stabilize_network(self):
    #     """Test if all finger tables are updated correctly."""
    #     self.network.update_all_finger_tables()
    #     for node in self.network.node_bank.values():
    #         if node.is_active:
    #             self.assertGreater(len(node.finger_table['successors']), 0)

if __name__ == "__main__":
    unittest.main() 
