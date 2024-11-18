from Node  import Node

class NetworkClass:
    def __init__(self, node_bank):
        self.node_bank = node_bank

    def assign_successors_and_predecessors(self, node: Node, r: int):
        # Get the list of active node IDs in ascending order
        active_nodes_ids = sorted([n.id for n in self.node_bank.values() if n.is_active])
        total_number_of_active_nodes = len(active_nodes_ids)

        # Validate input
        if node.id not in active_nodes_ids:
            raise ValueError(f"Node {node.id} is not active or missing from the active nodes.")
        if r > total_number_of_active_nodes:
            raise ValueError(f"Not enough active nodes to assign {r} successors/predecessors.")
        
        # Get the index of the current node in the sorted active nodes list
        node_index = active_nodes_ids.index(node.id)
        
        # Assign successors (based on Chord's finger table definition)
        node.finger_table['successors'] = []
        for i in range(r):
            successor_index = (node_index + pow(2, i)) % total_number_of_active_nodes
            successor_node_id = active_nodes_ids[successor_index]
            node.finger_table['successors'].append(successor_node_id)
        
        # Assign predecessor (only one predecessor for this use case)
        predecessor_index = (node_index - 1) % total_number_of_active_nodes
        predecessor_node_id = active_nodes_ids[predecessor_index]
        node.finger_table['predecessors'] = [predecessor_node_id]




node_bank = {
    1: Node(1, 10, True),
    2: Node(2, 10, True),
    3: Node(3, 10, True),  # Inactive
    4: Node(4, 10, True),
    5: Node(5, 10, True),
    6: Node(6, 10, True),
    7: Node(7, 10, True),
    8: Node(8, 10, True),
    9: Node(9, 10, True),
    10: Node(10, 10, True),
}
network = NetworkClass(node_bank)

# Assign successors and predecessors for Node 1 with r=2
# network.assign_successors_and_predecessors(node_bank[1], r=2)
network.assign_successors_and_predecessors(node_bank[2], r=4)

# Output the finger table of Node 1
print(node_bank[2])