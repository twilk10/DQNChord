import random
import time
import threading

class Node:
    def __init__(self, identifier):
        self.identifier = identifier
        self.data_keys = set()  # Set of data keys managed by this node

    def __repr__(self):
        return f"Node({self.identifier}, Data: {sorted(self.data_keys)})"

class ChordRing:
    def __init__(self, total_nodes=16, max_identifier=16):
        self.nodes = []  # List of active nodes in the ring
        self.node_bank = []  # List of all potential node IDs (0 to max_identifier - 1)
        self.max_identifier = max_identifier

        # Initialize node bank with all possible identifiers
        for i in range(max_identifier):
            self.node_bank.append(i)

        # Create initial nodes
        for _ in range(total_nodes):
            self.join_node()

    def find_predecessor(self, node_id):
        """Finds the predecessor of a given node ID in the sorted list of nodes."""
        if len(self.nodes) == 1:
            return self.nodes[0]

        for i in range(len(self.nodes)):
            if self.nodes[i].identifier == node_id:
                return self.nodes[i - 1] if i > 0 else self.nodes[-1]

        return None

    def join_node(self):
        if self.node_bank:
            new_node_id = self.node_bank.pop(random.randint(0, len(self.node_bank) - 1))
            new_node = Node(new_node_id)

            # Assign initial data to the new node (correlating to its identifier)
            new_node.data_keys.add(new_node_id)

            # Inherit data keys from its predecessor
            predecessor = self.find_predecessor(new_node_id)
            if predecessor:
                inherited_keys = {key for key in predecessor.data_keys if key <= new_node_id}
                new_node.data_keys.update(inherited_keys)
                predecessor.data_keys.difference_update(inherited_keys)

            self.nodes.append(new_node)
            self.nodes.sort(key=lambda x: x.identifier)
            print(f"Node {new_node_id} joined the ring with data {sorted(new_node.data_keys)}")

    def leave_node(self):
        if self.nodes:
            leaving_node = self.nodes.pop(random.randint(0, len(self.nodes) - 1))
            self.node_bank.append(leaving_node.identifier)

            # Redistribute data to its successor
            if self.nodes:
                successor = self.nodes[0] if len(self.nodes) == 1 else self.nodes[0 if self.nodes[0].identifier > leaving_node.identifier else 1]
                successor.data_keys.update(leaving_node.data_keys)
                print(f"Node {leaving_node.identifier} left the ring and transferred data to Node {successor.identifier}")

            print(f"Node {leaving_node.identifier} left the ring")

    def display_ring_state(self):
        print("\nCurrent state of the Chord ring:")
        for node in self.nodes:
            print(node)

    def simulate(self):
        while True:
            # Randomly decide if a node will join or leave
            action = random.choice(['join', 'leave'])
            if action == 'join':
                self.join_node()
            elif action == 'leave':
                self.leave_node()

            # Display the current state every 5 seconds
            self.display_ring_state()
            time.sleep(5)

# Create and start the Chord simulation with 16 nodes and max identifier space of 16
chord_ring = ChordRing(total_nodes=16, max_identifier=16)
simulation_thread = threading.Thread(target=chord_ring.simulate)
simulation_thread.daemon = True  # Run in background
simulation_thread.start()

# Run the simulation for a set duration (e.g., 60 seconds)
time.sleep(60)

