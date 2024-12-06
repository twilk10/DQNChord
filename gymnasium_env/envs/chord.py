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
                    
                    if self._is_in_closed_interval(n, start, next_start):
                        node.finger_table['successor'][i] = n
                        break
                    
            # Set the successor of the node to the first entry in the finger table
            node.successor = node.id


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
        # print(f'finding successor of id: {id} starting from node {node.id}...')
        # if id == node.id:
        #     # print(f'\t\t{id} is equal to {node.id}')
        #     return self.node_bank[node.successor]
        
        n_prime = self.find_predecessor(node, id)
        # print(f'+++++ received predecessor of {id} as: {predecessor.id}')
      
        
        return self.node_bank[n_prime.finger_table['successor'][0]]

    def find_predecessor(self, node: Node, id: int) -> Node: # ask node to find predecessor of id
        '''
        works by finding the immediate predecessor node
        of the desired identifier; the successor of that node must be the
        successor of the identifier
        '''
        # print(f'\tfinding predecessor of id: {id} from node {node.id}...')
        n_prime = node
        # print(f'successor of {id} is: {successor_node.id}')

        # test_interval = (n_prime.id, n_prime.successor)
        # print(f'\ttest interval is: {test_interval}')   
        if self._is_in_left_closed_right_open_interval(id, n_prime.id, n_prime.successor, verbose = False):
                # print(f'FINAL FUCKINGLY OUTER----------------------------------------------------------------')
                return self.node_bank[n_prime.predecessor]
        

        interval = (n_prime.id, n_prime.finger_table['successor'][0])
        # print(f'\tinitial interval of node {n_prime.id} is: {interval}')

       
        while not self._is_in_left_open_right_closed_interval(id, interval[0], interval[1]):
            # if id == n_prime.id:
            #     # print(f'\t\t{id} is equal to {n_prime.id}')
            #     return self.node_bank[n_prime.predecessor]
            
            interval = (n_prime.id, n_prime.finger_table['successor'][0])
            if self._is_in_closed_interval(id, interval[0], interval[1], verbose = False):
                # print(f'FINAL FUCKINGLY INNER----------------------------------------------------------------')
                return n_prime

            # print('\tin predecessor finding loop')
            returned_node = self.closest_preceding_node(n_prime, id)
            # print(f'\tclosest preceding node (returned node) of id: {id} is: {returned_node.id}, from node {n_prime.id}')

            # check if the returned node is the successor of the identifier
            n_prime = returned_node
            interval = (returned_node.id, returned_node.successor)

            
            # print(f'\ttest interval is: {test_interval}')   
            # if self._is_in_left_closed_right_open_interval(id, test_interval[0], test_interval[1], verbose = False):
            #     print(f'\t\t{id} is in interval ({test_interval[0]}, {test_interval[1]})')
            #     print(f'----------------------------------------------------------------RAHFJHDFKJHDKJFDKJDFK')
            #     return returned_node
          
            
            # n_prime = returned_node
            # interval = (n_prime.id, n_prime.finger_table['successor'][0])
            
            # print(f'\tinterval of node {n_prime.id} is: {interval}')
            # print(f'\tis id {id} in interval {interval}: {self._is_in_left_open_right_closed_interval(id, interval[0], interval[1], verbose = False)}')
            # print('----------------------------------------------------------------')
            
         
        # print(f'\n \t--------predecessor of {id} is: {n_prime.id}')
        return n_prime

    def closest_preceding_node(self, node: Node, id: int) -> Node:# ask node to find closest preceding node of id
      
        for i in reversed(range(self.m)):
            # interval = node.finger_table['interval'][i]
            # # print(f'\t\tInterval at finger table entry {i} is: {interval}')
            # if(node.finger_table['successor'][i] == node.id):
            #     continue    
            x = node.finger_table['successor'][i]
            if self._is_in_open_interval(x, node.id, id,verbose = False):
                # print(f'\t\t {id} is in interval {interval}')
                # print(f'\t\tclosest preceding node of id: {id} is: {node.id}')
                # return_node_id = node.finger_table['successor'][i]  
                # print(f'\t\tInterval for this mess is : ({node.id}, {id})')
                # print(f'\t\treturning node: {return_node_id}, FRUSTRATINGLY')
                return self.node_bank[x]
            
                  
        # print(f'closest preceding node of {id} is: {node.id}')

        # print('could not find closest preceding node in finger table, searching successor list')
        max_closest_preceding_node_id = node.id
        for i in reversed(range(self.m)):
            if(node.finger_table['successor'][i] == node.id):
                continue    
            # choose the largest of the successors whose value is less than id
            if id > node.finger_table['successor'][i]:
                max_closest_preceding_node_id = max(max_closest_preceding_node_id, node.finger_table['successor'][i])
            

        # print(f'max closest preceding node id foundis: {max_closest_preceding_node_id} --------IMPORTANT') 
        return self.node_bank[max_closest_preceding_node_id]



    def _is_in_closed_interval(self, x: int, start: int, end: int, verbose = True) -> bool:
        # Handle circular interval
        #  left <= x <= right
        # print('\t\tin closed interval')
        if start < end:
            # if verbose:
                # print(f'start is less than end')
            
            # if verbose:
                # print(f'{x} is in closed interval ({start}, {end})')
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
        # print('\t\tin open interval')
        if left < right:
            # print(f'\t\tleft is less than right')
            # print(f'\t\t{x} is in interval ({left}, {right})')
            return left < x < right
        elif left > right:
            # print(f'\t\tleft is greater than right')
            if x >= 0 and x < right:
                # print(f'\t\t{x} is in interval ({left}, {right})')
                return True
            if x > left and x <= pow(2, self.m):
                # print(f'\t\t{x} is in interval ({left}, {right})')
                return True
        else:
            # print(f'\t\tleft is equal to right')
            # print('\t\treturning False')    
            return False
        
        # print(f'\t\t{x} is not in interval ({left}, {right})  end')
        return False
    

    def stabilize(self, node: Node):  # Updates predecessor and successor pointers of a node
        print(f'stabilizing node {node.id}...')
        
        
        try:
            # ask the node's successor for its predecessor
            successor_node = self.node_bank[node.finger_table['successor'][0]]
            x = successor_node.predecessor
            print(f'x is: {x}')

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
        for i in range(len(finger_table['start'])):
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

    def join_network(self, node: Node): # find the immediate successor of node
        print(f'Joining node {node.id} to the network...')
        if node.id not in self.node_bank:
            # print('Node is not in node bank, adding it...')
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
        # print('Node that was added to the network is: ', node)
        
        print(f"Node {node.id} has joined the network.\n")

        # if self.verbose:
        #     print(f"Node {node.id} has joined the network.\n")
        #     print(f'Node is: {node}')

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
  

# Example Usage
if __name__ == "__main__":
    m = 3
    keys = [1,2,6]
    nodes_to_activate = [0, 1, 3]   
    network = ChordNetwork(m=m, keys=keys, nodes_to_activate=nodes_to_activate, verbose=True)
    # network.display_network()

    # # check if successor of key 1 is 1
    # successor1 = network.find_successor(network.node_bank[3], 1)
    # assert successor1.id == 1

    # check if successor of key 2 is 3
    # must start from an active node


    # print('--------------------------------')
    # print('Test closest_preceding_node function')
    # print('--------------------------------')
  
    # for identifier in range(2**m):
    #     active_nodes = [n for n in network.node_bank.values() if n.is_active]
    #     active_nodes = active_nodes[:1]
    #     # print('active nodes are: ', [n.id for n in active_nodes])
    #     for start_node in active_nodes:
    #         print(f'finding closest preceding node for identifier {identifier} at node {start_node.id}')
    #         closest_preceding_node = network.closest_preceding_node(start_node, identifier)
    #         print(f'closest preceding node is: {closest_preceding_node.id}') 
    #         print('----------------------------------------------------------------')   
            # assert closest_preceding_node.id == actual_closest_preceding_node_id
            
        
        
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
            predecessor = network.find_predecessor(start_node, identifier)
            # print(f'predecessor of {identifier} is: {predecessor.id}')
            # print('----------------------------------------------------------------')
            assert predecessor.id == actual_predecessor_id
           











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
    for key, actual_successor_id in successor_map.items():
        active_nodes = [n for n in network.node_bank.values() if n.is_active]
        active_nodes = active_nodes[:1]
        # print('active nodes are: ', [n.id for n in active_nodes])
        for start_node in active_nodes:
            # print('----------------------------------------start node is: ', start_node.id)
            successor = network.find_successor(start_node, key)
            # print(f'-------------actual successor is: {actual_successor_id}')
            assert successor.id == actual_successor_id


   

    print('--------------------------------')
    print('Test joining a new node to the network')
    print('--------------------------------')
    new_node = Node(id=6, active_status=False)
    network.join_network(new_node)
    # print('After joining new node, network state is:')
    # network.display_network()
    # print('after joining new node, network state is:')
    # assert that the new node is in the network
    assert network.node_bank[new_node.id].is_active
    # network.display_network()

    # print('stabilizing the network by stabilizing all nodes...')
    active_nodes = [n for n in network.node_bank.values() if n.is_active]

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
    for i in range(5000):
        random_choice_of_active_node = random.choice(active_nodes)
        network.stabilize(random_choice_of_active_node)

    network.display_network()

    # assert that the network is stable
    # print('asserting that the network is stable...')
    # active_nodes = [n for n in network.node_bank.values() if n.is_active]
    # for node in active_nodes:
    #     if node.id == 0:
    #         assert node.predecessor == 6
    #         assert node.successor ==  1

    #     if node.id == 1:
    #         assert node.predecessor == 0
    #         assert node.successor == 3

    #     if node.id == 3:
    #         assert node.predecessor == 1
    #         assert node.successor == 6

    #     if node.id == 6:
    #         assert node.predecessor == 3
    #         assert node.successor == 0

    # network.display_network()  



    # print('--------------------------------')
    # print('Test find_successor function with the new network state')
    # print('--------------------------------')
    # map ={
    #     0:1,
    #     1:3,
    #     2:3,    
    #     3:6,
    #     4:6,
    #     5:6,
    #     6:0,
    #     7:0
    # }
    # for key, actual_successor_id in map.items():
    #     active_nodes = [n for n in network.node_bank.values() if n.is_active]
    #     active_nodes = active_nodes[:1]
    #     # print('active nodes are: ', [n.id for n in active_nodes])
    #     for start_node in active_nodes:
    #         print('----------------------------------------start node is: ', start_node.id)
    #         successor = network.find_successor(start_node, key)
    #         assert successor.id == actual_successor_id



    # fixing fingers test
    print('--------------------------------')
    print('fix fingers test')
    print('--------------------------------')

    # for i in range(10000):
    #     random_choice_of_active_node = random.choice(active_nodes)
    #     network.fix_fingers(random_choice_of_active_node)
    network.fix_fingers(network.node_bank[0])
    network.display_network()

