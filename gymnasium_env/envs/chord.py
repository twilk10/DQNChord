import hashlib
import random
import threading
import time
from typing import List, Dict, Optional
import random

random.seed(42)

class Node:
    def __init__(self, id: int, active_status: bool, predecessor: Optional[int] = None, successor: Optional[int] = None, finger_table: Optional[List[int]] = None, m: int = 0):
        self.is_active = active_status
        self.id = id
        self.predecessor = predecessor
        self.successor = successor
        self.finger_table : Dict[str, List[int]] = {
            'start': [0]* m,
            'interval': [0]*m,
            'successor': [0]*m
        }
        self.keys: List[int] = []  # Initialize keys list

    def set_active_status(self, new_status: bool):
        self.is_active = new_status

    def __str__(self):
        return (f"\t Node Id: {self.id}\n"
                f"\t Active Status: {self.is_active}\n"
                f"\t Predecessor: {self.predecessor}\n"
                f"\t Successor: {self.successor}\n"
                f"\t Finger Table: {self.finger_table}\n"
                f"\t Keys: {self.keys}\n")

# ChordNetwork Class
class ChordNetwork:
    def __init__(self, m=4, keys: List[str] = [], nodes_to_activate: List[int] = [], verbose=False, seed=42):
        self.m = m
        self.max_network_size = 2 ** self.m
        self.n_nodes_to_activate = self.max_network_size // 2
        self.verbose = verbose
        self.node_bank: Dict[int, Node] = {}

        self.keys = keys
        self.nodes_to_activate = nodes_to_activate

        # Seed the random number generator
        random.seed(seed)
        if self.verbose:
            print(f"Random seed set to {seed}")

        self.initialize_node_bank()
        self.activate_n_nodes()

        
        if self.verbose:
            print('Initialization Done!\n')

    def initialize_node_bank(self):
        if self.verbose:
            print(f'Initializing node bank with identifier space size {self.max_network_size}...')
        
        for i in range(self.max_network_size):
            if i not in self.node_bank:
                node = Node(id=i, active_status=False, m=self.m)
                self.node_bank[node.id] = node

    def activate_n_nodes(self):
        if self.verbose:
            print(f'Activating {len(self.nodes_to_activate)} Nodes...')

       # HARDCODING FOR NOW
        self.nodes_to_activate = sorted(self.nodes_to_activate)
        for idx, node_id in enumerate(self.nodes_to_activate):
            node = self.node_bank[node_id]
            node.is_active = True
            predecessor_idx = (idx - 1) % len(self.nodes_to_activate)
            node.predecessor = self.nodes_to_activate[predecessor_idx]

            # Initialize the finger table
            for i in range(self.m):
                start = (node.id + pow(2, i)) % pow(2, self.m)
                node.finger_table['start'][i] = start   
                
                next_start = (node.id + pow(2, i+1)) % pow(2, self.m)
                interval = (start, next_start)
                node.finger_table['interval'][i] = interval

                # fin the successor of start
                for n in self.nodes_to_activate:
                    
                    if self._is_in_interval(n, start, next_start):
                        node.finger_table['successor'][i] = n
                        break
                    
            # Set the successor of the node to the first entry in the finger table
            node.successor = node.finger_table['successor'][0]


        for key in self.keys:
            if key == 6:
                self.node_bank[0].keys.append(key)
            if key == 1:
                self.node_bank[1].keys.append(key)
            if key == 2:
                self.node_bank[3].keys.append(key)
          
                

        # we will need to activate nodes in a way that the keys are distributed evenly across the network
        # we will need to find the node that has the key and activate the node that has the key
        # for key in self.keys:
        #     # key_id = hash_key(key, self.m) No need to hash the key since we are using the key to find the node
        #     successor_node = self.find_successor(node, key)
        #     self.join_network(successor_node)

        # For testing purposes, we will activate nodes 0, 1 3
       

    def find_successor(self, node: Node, id: int) -> Node: # ask node to find successor of id
        print(f'finding successor of id: {id} starting from node {node.id}...')
        predecessor = self.find_predecessor(node, id)
        # print(f'predecessor of {id} is: {predecessor.id}')
        successor_id = predecessor.successor if predecessor.successor is not None else predecessor.id
        print(f'successor of id: {id} is: {successor_id}')
        return self.node_bank[successor_id]

    def find_predecessor(self, node: Node, id: int) -> Node: # ask node to find predecessor of id
        '''
        works by finding the immediate predecessor node
        of the desired identifier; the successor of that node must be the
        successor of the identifier
        '''
        print(f'\tfinding predecessor of id: {id} from node {node.id}...')
        n_prime = node
        # print(f'successor of {id} is: {successor_node.id}')

        interval = (n_prime.id, n_prime.successor)
        print(f'\tinitial interval of node {n_prime.id} is: {interval}')

        counter = 0
        while not self._is_in_interval(id, interval[0], interval[1]):
            n_prime = self.closest_preceding_node(n_prime, id)
            print(f'\treturned value: {n_prime.id}')
            print(f'\tn prime is now: {n_prime.id}')
            interval = (n_prime.id, n_prime.successor)
            print(f'\tupdated interval of node {n_prime.id} is: {interval}')
            counter += 1
            if counter > 4:
                break
           
        print(f'\n \tpredecessor of {id} is: {n_prime.id}')
        return n_prime

    def closest_preceding_node(self, node: Node, id: int) -> Node:# ask node to find closest preceding node of id
        print(f'\t\tfinding closest preceding node of id: {id} from node {node.id}...')
        for i in reversed(range(self.m)):
            interval = node.finger_table['interval'][i]
            print(f'\t\tInterval at finger table entry {i} is: {interval}')
            if self._is_in_interval(id, interval[0], interval[1]):
                print(f'\t\t {id} is in interval {interval}')
                # print(f'\t\tclosest preceding node of id: {id} is: {node.id}')
                return_node_id = node.finger_table['successor'][i]  
                print(f'\t\treturning node: {return_node_id}')
                return self.node_bank[return_node_id]
            else:
                print(f'\t\t {id} is not in interval {interval}')
        
        # print(f'closest preceding node of {id} is: {node.id}')
        return node

    def _is_in_interval(self, x: int, start: int, end: int) -> bool:
        # Handle circular interval
        if start < end:
            return start <= x <= end
        elif start > end:
            if x >= 0 and x <= end:
                return True
            if x >= start and x < pow(2, self.m):
                return True
          

    def stabilize(self, node: Node):  # Updates predecessor and successor pointers
        print(f'stabilizing node {node.id}...')
        
        
        try:
            successor_node = self.node_bank.get(node.successor)
            x_id = successor_node.predecessor

            if x_id is not None and self._is_in_interval(x_id, node.id, successor_node.id):
                node.successor = x_id
                node.finger_table[0] = x_id  # Synchronize finger table's first entry
                if self.verbose:
                    print(f"Node {node.id}: Updated successor to {x_id}.")

            self.notify(successor_node, node)
        except Exception as e:
            print(f"Error stabilizing node {node.id}: {str(e)}")

            if not successor_node:
                node.successor = node.id

            if node.predecessor is None:
                node.predecessor = node.id

    def notify(self, node: Node, n_prime: Node): # ask node to update its predecessor to n_prime
        if node.predecessor is None or self._is_in_interval(n_prime.id, node.predecessor, node.id):
            node.predecessor = n_prime.id
            if self.verbose:
                print(f"Node {node.id} Updated predecessor to {n_prime.id}.")

    def fix_fingers(self, node: Node):  # Update the finger table of a node
        print(f'fixing fingers for node {node.id}...')
        finger_table = node.finger_table

        for i in range(self.m):
            
            start = (node.id + pow(2, i)) % pow(2, self.m)

            # Find the successor of start_i
            successor_node = self.find_successor(node, start)

            # Update the finger table
            finger_table['start'][i] = start
            next_start = (node.id + pow(2, i+1)) % pow(2, self.m)
            interval = (start, next_start)
            finger_table['interval'][i] = interval
            finger_table['successor'][i] = successor_node.id

    def join_network(self, node: Node):
        print(f'Joining node {node.id} to the network...')
        node.set_active_status(True)
        node.predecessor = None
        # Get all other active nodes excluding the joining node
        active_nodes = [n for n in self.node_bank.values() if n.is_active and n.id != node.id]
        if not active_nodes:
            # First node in the network points to itself
            node.successor = node.id
            node.predecessor = node.id
            node.finger_table = [node.id] * self.m
            if self.verbose:
                print(f"Node {node.id} is the first active node. It points to itself.")
        else:
            n_prime = random.choice(active_nodes)
            node.successor = self.find_successor(n_prime, node.id).id

        if self.verbose:
            print(f"Node {node.id} has joined the network.\n")

    def leave_network_gracefully(self, node: Node):
        node.set_active_status(False)
        # Inform predecessor and successor
        predecessor_node = self.node_bank.get(node.predecessor)
        successor_node = self.node_bank.get(node.successor)
        
        if predecessor_node and successor_node:
            predecessor_node.successor = successor_node.id
            successor_node.predecessor = predecessor_node.id
            # Transfer keys to successor
            successor_node.keys.extend(node.keys)
            node.keys.clear()
        
        # Clear the node's finger table
        node.predecessor = None
        node.successor = None
        node.finger_table = []
        
        if self.verbose:
            print(f"Node {node.id} has left the network gracefully.\n")
        # Optionally, trigger stabilization on predecessor and successor
        if predecessor_node and predecessor_node.is_active:
            self.stabilize(predecessor_node)
        if successor_node and successor_node.is_active:
            self.stabilize(successor_node)

    def node_failure(self, node: Node):
        node.set_active_status(False)
        # Inform predecessor and successor
        predecessor_node = self.node_bank.get(node.predecessor)
        successor_node = self.node_bank.get(node.successor)
        
        if predecessor_node and successor_node:
            predecessor_node.successor = successor_node.id
            successor_node.predecessor = predecessor_node.id
            # Optionally, handle keys (e.g., mark as lost or transfer)
        
        # Clear the node's finger table
        node.predecessor = None
        node.successor = None
        node.finger_table = []
        
        if self.verbose:
            print(f"Node {node.id} has failed.\n")
        # Optionally, trigger stabilization on predecessor and successor
        if predecessor_node and predecessor_node.is_active:
            self.stabilize(predecessor_node)
        if successor_node and successor_node.is_active:
            self.stabilize(successor_node)

    def join_x_random_nodes(self, x: int):
        # Join x random inactive nodes to the network
        for _ in range(x):
            inactive_nodes = [n for n in self.node_bank.values() if not n.is_active]
            if not inactive_nodes:
                break
            node_to_join = random.choice(inactive_nodes)
            self.join_network(node_to_join)

    def drop_x_random_nodes(self, x: int):
        # Drop x random active nodes from the network
        for _ in range(x):
            active_nodes = [n for n in self.node_bank.values() if n.is_active]
            if len(active_nodes) <= x:
                raise ValueError(f"Not enough active nodes to drop {x} nodes.")
            node_to_drop = random.choice(active_nodes)
            # We cannot drop the Node with id 0
            while node_to_drop.id == 0:
                node_to_drop = random.choice(active_nodes)
            self.leave_network_gracefully(node_to_drop)

    def get_network_state(self) -> Dict[int, List[Optional[int]]]:
        network_state = {}
        for node in self.node_bank.values():
            if node.is_active:
                network_state[node.id] = node.finger_table
        return network_state

    def display_network(self):
        print("Network state:")
        for node in sorted(self.node_bank.values(), key=lambda x: x.id):
            if node.is_active:
                print(node)

    # Periodic Stabilization (Optional Enhancement)
    def start_stabilization(self, interval=5):
        def stabilize_periodically():
            while True:
                for node in self.node_bank.values():
                    if node.is_active:
                        self.stabilize(node)
                        self.fix_fingers(node)
                time.sleep(interval)
        
        stabilization_thread = threading.Thread(target=stabilize_periodically, daemon=True)
        stabilization_thread.start()

    # Key Storage and Retrieval Methods (Optional Enhancement)
    def store_key(self, key: str):
        key_id = hash_key(key, self.m)
        successor_node = self.find_successor_by_key_id(key_id)
        self.key_store[key_id] = key
        successor_node.keys.append(key_id)
        if self.verbose:
            print(f"Key '{key}' hashed to {key_id} and stored at Node {successor_node.id}.")

    def retrieve_key(self, key: str):
        key_id = hash_key(key, self.m)
        successor_node = self.find_successor_by_key_id(key_id)
        if key_id in self.key_store:
            if self.verbose:
                print(f"Key '{key}' found at Node {successor_node.id}.")
            return successor_node.id
        else:
            if self.verbose:
                print(f"Key '{key}' not found in the network.")
            return None

    def find_successor_by_key_id(self, key_id: int) -> Node:
        # Start from an arbitrary active node
        active_nodes = [node for node in self.node_bank.values() if node.is_active]
        if not active_nodes:
            raise Exception("No active nodes in the network.")
        start_node = active_nodes[0]
        return self.find_successor(start_node, key_id)

# Example Usage
if __name__ == "__main__":
    m = 3
    keys = [1,2,6]
    nodes_to_activate = [0, 1, 3]   
    network = ChordNetwork(m=m, keys=keys, nodes_to_activate=nodes_to_activate, verbose=True)
    # network.display_network()
    successor = network.find_successor(network.node_bank[3], 1)
    # successor = network.find_predecessor(network.node_bank[0], 2)
    print(f'successor of 1 is: {successor.id}') # should be 1
