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

    def set_active_status(self, new_status: bool):
        self.is_active = new_status

    def __str__(self):
        return (f"\t Node Id: {self.id}\n"
                f"\t Active Status: {self.is_active}\n"
                f"\t Predecessor: {self.predecessor}\n"
                f"\t Successor: {self.successor}\n"
                f"\t Finger Table: {self.finger_table}\n")

# ChordNetwork Class
class ChordNetwork:
    def __init__(self, m=4, nodes_to_activate: List[int] = [], verbose=False, seed=42):
        self.m = m
        self.max_network_size = 2 ** self.m
        self.n_nodes_to_activate = self.max_network_size // 2
        self.verbose = verbose
        self.node_bank: Dict[int, Node] = {}
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
        
        if not self.nodes_to_activate or len(self.nodes_to_activate) == 0:
            n = self.max_network_size // 2
            self.nodes_to_activate = random.sample([i for i in range(self.max_network_size)], n)    

        # initialize finger tables  
        for node_id in self.nodes_to_activate:
            node = self.node_bank[node_id]
            node.is_active = True
            finger_table = node.finger_table
            for i in range(self.m):
                start: int = (node.id + pow(2, i)) % pow(2, self.m)
                finger_table['start'][i] = start
                finger_table['interval'][i] = (start, (node.id + pow(2, i+1)) % pow(2, self.m))

        # manually assign successors and predecessors   
        for node_id in self.nodes_to_activate:
            node = self.node_bank[node_id]
            self.manually_assign_successors_and_predecessors(node)
            
       # get the identifier successor id map
        identifier_successor_id_map = {}
        sorted_node_ids = sorted(self.node_bank.keys())
        print('my sorted node ids are: ', sorted_node_ids)
        for index, identifier in enumerate(sorted_node_ids):
            idx = index
            while not self.node_bank[sorted_node_ids[idx]].is_active:
                idx = (idx + 1) % len(sorted_node_ids)
            identifier_successor_id_map[identifier] = sorted_node_ids[idx]
                
        # initialize the successor nodes in the finger tables of active nodes
        print(f'identifier_successor_id_map is: {identifier_successor_id_map}')
        for node_id in self.nodes_to_activate:
            node = self.node_bank[node_id]
            finger_table = node.finger_table
            for i in range(self.m):
                start = finger_table['start'][i]
                print(f'start is: {start}')
                finger_table['successor'][i] = identifier_successor_id_map[start]


    def manually_assign_successors_and_predecessors(self, node: Node):
        # Get the list of active node IDs in ascending order
        active_nodes_ids = sorted([n.id for n in self.node_bank.values() if n.is_active])
        total_number_of_active_nodes = len(active_nodes_ids)
        
        
        # Get the index of the current node in the sorted active nodes list
        node_index = active_nodes_ids.index(node.id)
    

        # Assign predecessor (only one predecessor for this use case)
        predecessor_index = (node_index - 1) % total_number_of_active_nodes
        predecessor_node_id = active_nodes_ids[predecessor_index]
        node.predecessor = predecessor_node_id

        successor_index = (node_index + 1) % total_number_of_active_nodes
        successor_node_id = active_nodes_ids[successor_index]
        node.successor = successor_node_id
   

    def find_successor(self, node: Node, id: int) -> Node: # ask node to find successor of id
        n_prime = self.find_predecessor(node, id)
        print(f'+++++ received predecessor of {id} as: {n_prime.id}')
      
        
        return self.node_bank[n_prime.finger_table['successor'][0]]

    def find_predecessor(self, node: Node, id: int) -> Node: # ask node to find predecessor of id
        '''
        works by finding the immediate predecessor node
        of the desired identifier; the successor of that node must be the
        successor of the identifier
        '''
        n_prime = node
    
        if self._is_in_left_closed_right_open_interval(id, n_prime.id, n_prime.successor, verbose = False):
                if id == n_prime.id:
                    return self.node_bank[n_prime.predecessor]
                return n_prime
        

        interval = (n_prime.id, n_prime.finger_table['successor'][0])
        while not self._is_in_left_open_right_closed_interval(id, interval[0], interval[1]):
           
            # check if the id is in nprime's interval   
            interval = (n_prime.id, n_prime.finger_table['successor'][0])
            if self._is_in_closed_interval(id, interval[0], interval[1], verbose = False):
                if id == n_prime.id:
                    return self.node_bank[n_prime.predecessor]
                return n_prime
            
            returned_node = self.closest_preceding_node(n_prime, id)
            n_prime = returned_node
            interval = (returned_node.id, returned_node.successor)

        return n_prime

    def closest_preceding_node(self, node: Node, id: int) -> Node:# ask node to find closest preceding node of id
        for i in reversed(range(self.m)):
        
            x = node.finger_table['successor'][i]
            if self._is_in_open_interval(x, node.id, id,verbose = False):
                return self.node_bank[x]
        
        max_closest_preceding_node_id = node.id
        for i in reversed(range(self.m)):
            if(node.finger_table['successor'][i] == node.id):
                continue    
            # choose the largest of the successors whose value is less than id
            if id > node.finger_table['successor'][i]:
                max_closest_preceding_node_id = max(max_closest_preceding_node_id, node.finger_table['successor'][i])

        return self.node_bank[max_closest_preceding_node_id]



    def _is_in_closed_interval(self, x: int, start: int, end: int, verbose = True) -> bool:
        # Handle circular interval
        #  left <= x <= right
        # print('\t\tin closed interval')
        if start < end:
            return start <= x <= end
        elif start > end:
            # if verbose:
                # print(f'start is greater than end')
            if x >= 0 and x <= end:
                # if verbose:
                    # print(f'{x} is in closed interval ({start}, {end})')
                return True
            if x >= start and x < pow(2, self.m):
                # if verbose:
                    # print(f'{x} is in closed interval ({start}, {end})')
                return True
            
            # print(f'{x} is not in closed interval ({start}, {end})')    
            return False
        else:
            # start is equal to end
            # if verbose:
                # print(f'start is equal to end')
            
            if x >= start and x <= end:
                # if verbose:
                    # print(f'{x} is in closed interval ({start}, {end})')
                return True 
        return False
        
    def _is_in_left_open_right_closed_interval(self, x: int, left: int, right: int, verbose = False) -> bool:
        # left < x <= right  
        # Handle circular interval
        # print('\t\tin left open right closed interval')
        if left < right:
            # print(f'\t\tleft is less than right')
            # print(f'\t\t{x} is in interval ({left}, {right})')
            # print(f'\t\treturning {left < x <= right}')
            return left < x <= right
        elif left > right:
            # print(f'\t\tleft is greater than right')
            if x >= 0 and x <= right:
                # print(f'\t\t{x} is in interval ({left}, {right})')
                return True
            elif x > left and x <= pow(2, self.m):
                # print(f'\t\t{x} is in interval ({left}, {right})')
                return True
            else:
                # print(f'\t\t{x} is not in interval ({left}, {right})')
                return False
        else:
            # print(f'\t\tleft is equal to right')
            if x == right:
                # print(f'\t\t{x} is in interval ({left}, {right})')
                return True 
            else:
                # print(f'\t\t{x} is not in interval ({left}, {right})')
                return False
        return False
    
    def _is_in_left_closed_right_open_interval(self, x: int, left: int, right: int, verbose = False) -> bool:
        # left <= x < right  
        # Handle circular interval
        # print('in left closed right open interval')
        if left < right:
            # print(f'left is less than right')
            if left <= x < right:
                # print(f'{x} is in the interval ({left}, {right})')
                return True
            else:
                # print(f'{x} is not in the interval ({left}, {right})')
                return False
        elif left > right:
            # print(f'left is greater than right')
            if x >= 0 and x < right:
                # print(f'{x} is in interval ({left}, {right})')
                return True
            if x >= left and x <= pow(2, self.m):
                # print(f'{x} is in interval ({left}, {right})')
                return True
        else:
            # print(f'left is equal to right')
            if x == left:
                # print(f'{x} is in interval ({left}, {right})')
                return True 
        # print(f'{x} is not in interval ({left}, {right})')
        return False
    
    def _is_in_open_interval(self, x: int, left: int, right: int, verbose = False) -> bool:
        # left < x < right  
        # Handle circular interval
        if left < right:
            return left < x < right
        elif left > right:
            if x >= 0 and x < right:
                return True
            if x > left and x <= pow(2, self.m):
                # print(f'\t\t{x} is in interval ({left}, {right})')
                return True
        else:
            return False
        
        
        return False
    

    def stabilize(self, node: Node):  # Updates predecessor and successor pointers of a node
        print(f'stabilizing node {node.id}...')
        
        
        try:
            # ask the node's successor for its predecessor
            successor_node = self.node_bank[node.finger_table['successor'][0]]
            x = successor_node.predecessor
            # print(f'x is: {x}')

            interval = (node.id, successor_node.id)
            # print(f'interval is: {interval}')
            if x is not None and self._is_in_open_interval(x, interval[0], interval[1]):
                # print(f'{x} is in interval ({interval[0]}, {interval[1]})')     
                node.successor = x
                # if self.verbose:
                    # print(f"Node {node.id}: Updated successor to {x}.")

            # print(f' {node.successor} notifies {node.id} that it is {node.id}\'s successor')
            self.notify(self.node_bank[node.successor], node)
        except Exception as e:
            # print(f"Error stabilizing node {node.id}: {str(e)}")
            pass


    def notify(self, node: Node, n_prime: Node): # ask node to update its predecessor to n_prime
        
        # print(f'notifying node {node.id} to update its predecessor to {n_prime.id}')
        predecessor = node.predecessor
        # print(f'\tpredecessor is: {predecessor}')
        if predecessor is None or self._is_in_open_interval(n_prime.id, predecessor, node.id):
            node.predecessor = n_prime.id
            # print(f"Node {node.id} Updated predecessor to {n_prime.id}.")
        else:
            # print(f'\t\tDid not update predecessor of {node.id} to {n_prime.id}')
            pass

    def fix_fingers(self, node: Node):  # Update the finger table of a node
        print(f'fixing fingers for node {node.id}...')
        finger_table = node.finger_table

        # random index from 0 to len(finger_table['start']) - 1
        for i in range(self.m):
            start = (node.id + pow(2, i)) % pow(2, self.m)
            finger_table['start'][i] = start
            # successor = self.find_successor(node, start)
            # finger_table['successor'][i] = successor.id
            finger_table['interval'][i] = (start, (node.id + pow(2, i+1)) % pow(2, self.m))

        
        # fixing the successor nodes randomly to account for periods of instability 
        # i = random.randint(1, len(finger_table['start']) - 1)
        for i in range(1, len(finger_table['start'])):
            print(f'fixing successor at index {i}') 
            print(f'finding successor for identifier {finger_table["start"][i]}')
            successor = self.find_successor(node, finger_table['start'][i])
            print(f'found successor: {successor.id}')
            print('----------------------------------------------------------------')
            finger_table['successor'][i] = successor.id

    def manually_fix_fingers(self, node: Node):
        '''
            manually fix the fingers of a node
        '''

        finger_table = node.finger_table
        for i in range(self.m):
            start: int = (node.id + pow(2, i)) % pow(2, self.m)
            finger_table['start'][i] = start
            finger_table['interval'][i] = (start, (node.id + pow(2, i+1)) % pow(2, self.m))

        # get the identifier successor id map
        identifier_successor_id_map = {}
        sorted_node_ids = sorted(self.node_bank.keys())
        print('my sorted node ids are: ', sorted_node_ids)
        for index, identifier in enumerate(sorted_node_ids):
            idx = index
            while not self.node_bank[sorted_node_ids[idx]].is_active:
                idx = (idx + 1) % len(sorted_node_ids)
            identifier_successor_id_map[identifier] = sorted_node_ids[idx]
                
        # initialize the successor nodes in the finger tables of the given node
        print(f'identifier_successor_id_map is: {identifier_successor_id_map}')
        for i in range(self.m):
            start = finger_table['start'][i]
            print(f'start is: {start}')
            finger_table['successor'][i] = identifier_successor_id_map[start]

        
    def join_network(self, node: Node): # find the immediate successor of node
        print(f'Joining node {node.id} to the network...')
        try:
            if node.id not in self.node_bank:
                self.node_bank[node.id] = node
            else:
                node = self.node_bank[node.id]
            
            node.predecessor = None
            
            # Get all other active nodes excluding the joining node
            active_nodes = [n for n in self.node_bank.values() if n.is_active and n.id != node.id]
            n_prime = random.choice(active_nodes)

            successor_node = self.find_successor(n_prime, node.id)
          
            node.set_active_status(True)
            node.successor = successor_node.id
            print(f'predecessor of {node.id} is: {node.predecessor}')
            print(f"Node {node.id} has joined the network.\n")
        except Exception as e:
            print(f"Error joining node {node.id} to network: {str(e)}")
            pass

    def leave_network(self, node: Node):
        node.set_active_status(False)
        
        # Clear the node's information
        node.predecessor = None
        node.successor = None
        node.finger_table = {}
        
        if self.verbose:
            print(f"Node {node.id} has left the network.\n")
     

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
  

# Example Usage and Tests
if __name__ == "__main__":
    m = 3
    nodes_to_activate = [0, 1, 3]   
    network = ChordNetwork(m=m, nodes_to_activate=nodes_to_activate, verbose=True)
    network.display_network()
        
        
    # print('--------------------------------')
    # print('Test find_predecessor function')
    # print('--------------------------------')
    predecessor_map ={
        0:3,
        1:0,
        3:1,
        2:1,    
        4:3,
        5:3,
        6:3,
        7:3
    }
    for identifier, actual_predecessor_id in predecessor_map.items():
        active_nodes = [n for n in network.node_bank.values() if n.is_active]
        for start_node in active_nodes:
            print(f'finding predecessor of identifier {identifier} from node {start_node.id} ')
            predecessor = network.find_predecessor(start_node, identifier)
            print(f'predecessor of {identifier} is: {predecessor.id}')
            print('----------------------------------------------------------------')
            assert predecessor.id == actual_predecessor_id
           

    # finding the successor of an identifier
    successor_map ={
        0:0,
        1:1,
        2:3,    
        3:3,
        4:0,
        5:0,
        6:0,
        7:0
    }
    print('testing successor function')
    for key, actual_successor_id in successor_map.items():
        active_nodes = [n for n in network.node_bank.values() if n.is_active]
        active_nodes = active_nodes[:1]
        # print('active nodes are: ', [n.id for n in active_nodes])
        for start_node in active_nodes:
            # print('----------------------------------------start node is: ', start_node.id)
            successor = network.find_successor(start_node, key)
            # print(f'-------------actual successor is: {actual_successor_id}')
            assert successor.id == actual_successor_id


   

    # print('--------------------------------')
    # print('Test joining a new node to the network')
    # print('--------------------------------')
    new_node = Node(id=6, active_status=False)
    network.join_network(new_node)
    print('After joining new node, network state is:')
    network.display_network()

    # assert that the new node is in the network
    # assert network.node_bank[new_node.id].is_active
    # network.display_network()

    # print('stabilizing the network by stabilizing all nodes...')
    # active_nodes = [n for n in network.node_bank.values() if n.is_active]

    # Best order to stabilize after a new node is to stabilize the new node, 
    # then stabilize the new node's successor, then stabilize the new node's predecessor
    # Then stabilize the new node;s sucessor's predecessor.
    # In this case, we stabilize 6, then 0, then 3

    # # stabilize node 6
    # print('before stabilizing node 6, 0, and 3, network state is:')
    # # network.display_network()
    # network.stabilize(network.node_bank[6])
    # network.stabilize(network.node_bank[0])
    # network.stabilize(network.node_bank[3]) 
    # print('after stabilizing node 6, 0, and 3, network state is:')
    # network.display_network()
    
    # Randomly stabilizing nodes in the network for 5000 iterations
    print('Randomly stabilizing nodes in the network for 5000 iterations...')
    active_nodes = [n for n in network.node_bank.values() if n.is_active]
    for i in range(5000):
        random_choice_of_active_node = random.choice(active_nodes)
        network.stabilize(random_choice_of_active_node)

    network.display_network()

    # assert that the network is stable
    print('asserting that the network is stable...')
    active_nodes = [n for n in network.node_bank.values() if n.is_active]
    for node in active_nodes:
        if node.id == 0:
            assert node.predecessor == 6
            assert node.successor ==  1

        if node.id == 1:
            assert node.predecessor == 0
            assert node.successor == 3

        if node.id == 3:
            assert node.predecessor == 1
            assert node.successor == 6

        if node.id == 6:
            assert node.predecessor == 3
            assert node.successor == 0

    print('Network is stable!')
    print('--------------------------------')


    print('Testing manually fixing fingers...')
    for node in network.node_bank.values():
        network.manually_fix_fingers(node)

    print('Asserrting that all fingers are fixed...')   
    active_nodes = [n for n in network.node_bank.values() if n.is_active]
    for node in active_nodes:
        if node.id == 0:
            actual_start = [1, 2, 4]
            actual_interval = [(1, 2), (2, 4), (4, 0)]
            actual_successor = [1, 3, 6]
            for i in range(m):
                assert node.finger_table['start'][i] == actual_start[i]
                assert node.finger_table['interval'][i] == actual_interval[i]
                assert node.finger_table['successor'][i] == actual_successor[i]
        if node.id == 1:
            actual_start = [2, 3, 5]
            actual_interval = [(2, 3), (3, 5), (5,1)]
            actual_successor = [3, 3, 6]
            for i in range(m):
                assert node.finger_table['start'][i] == actual_start[i]
                assert node.finger_table['interval'][i] == actual_interval[i]
                assert node.finger_table['successor'][i] == actual_successor[i]
            
        if node.id == 3:
            actual_start = [4, 5, 7]
            actual_interval = [(4, 5), (5, 7), (7, 3)]
            actual_successor = [6, 6, 0]
            for i in range(m):
                assert node.finger_table['start'][i] == actual_start[i]
                assert node.finger_table['interval'][i] == actual_interval[i]
                assert node.finger_table['successor'][i] == actual_successor[i]
            
        if node.id == 6:
            actual_start = [7, 0, 2]
            actual_interval = [(7, 0), (0, 2), (2, 6)]
            actual_successor = [0, 0, 3]
            for i in range(m):
                assert node.finger_table['start'][i] == actual_start[i]
                assert node.finger_table['interval'][i] == actual_interval[i]
                assert node.finger_table['successor'][i] == actual_successor[i]
    print('All fingers are fixed!')             
    network.display_network()  


    print('--------------------------------')
    print('Test find_predecessor function with the new network state')
    print('--------------------------------')
    print('predecessor of identifiers:')
    predecessor_map ={
        0:6,
        1:0,
        2:1,    
        3:1,
        4:3,
        5:3,
        6:3,
        7:6
    }  
    
    for key, actual_predecessor_id in predecessor_map.items():
        active_nodes = [n for n in network.node_bank.values() if n.is_active]
        for start_node in active_nodes:
            predecessor = network.find_predecessor(start_node, key)
            assert predecessor.id == actual_predecessor_id
    print('All predecessors are correct!')  

    print('--------------------------------')
    print('Test find_successor function with the new network state')
    print('--------------------------------')
    print('successor of identifiers:')
    successor_map ={
        0:0,
        1:1,
        2:3,    
        3:3,
        4:6,
        5:6,
        6:6,
        7:0
    }
    for key, actual_successor_id in successor_map.items():
        active_nodes = [n for n in network.node_bank.values() if n.is_active]
        active_nodes = active_nodes[:1]
        # print('active nodes are: ', [n.id for n in active_nodes])
        for start_node in active_nodes:
            successor = network.find_successor(start_node, key)
            print(f'successor of key {key} is: {successor.id}')
            assert successor.id == actual_successor_id
    print('All successors are correct!')


    # fixing fingers test
    # print('--------------------------------')
    # print('fix fingers test')
    # print('--------------------------------')

    # for i in range(10000):
    #     random_choice_of_active_node = random.choice(active_nodes)
    #     network.fix_fingers(random_choice_of_active_node)
    # network.fix_fingers(network.node_bank[0])
    # network.fix_fingers(network.node_bank[1])
    # network.fix_fingers(network.node_bank[3])
    # network.fix_fingers(network.node_bank[6])
    # network.display_network()

