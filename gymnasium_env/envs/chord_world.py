import gymnasium as gym 
import numpy as np
from typing import Dict, List
import random
from enum import Enum
from gymnasium.spaces import flatten_space, flatten


import random
from typing import List, Dict, Optional
import random

random.seed(42)


class Action(Enum):
    STABILIZE = 0
    DO_NOTHING = 1

class ChordWorldEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self, max_steps = 100, seed = 42):
        super(ChordWorldEnv, self).__init__()

        self.seed(seed)

        self.max_steps = max_steps
        self.current_step = 0

        # Initialize the network  
        self.verbose = False
        self.network = ChordNetwork(n_nodes_to_activate=20, r=2, bank_size=40, verbose=self.verbose)
        self.previous_network_state = None
        self.network_state = None  
         
        # Define max network size and r based on the network
        self.min_network_size = self.network.r+2
        self.max_network_size = self.network.bank_size  # This should be 40
        self.r = self.network.r  # This should be 2
        
        
        self.action_space = gym.spaces.Discrete(2)  # Actions 0 to 2

        self.original_observation_space = gym.spaces.Dict({
            'lookup_success_rate': gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            'stability_score': gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            'active_nodes': gym.spaces.MultiBinary(self.network.bank_size),
            'finger_tables': gym.spaces.Box(
                low=0, high=self.max_network_size - 1,
                shape=(self.max_network_size, self.r +1 ),
                dtype=np.int32
                ),
            })
        
        self.observation_space = flatten_space(self.original_observation_space)

        # Lookup stats
        self.total_lookup_attempts = 0
        self.failed_lookup_attempts = 0
        self.successful_lookup_attempts = 0
        self.lookup_success_rate = 0.0

        self.is_successful_lookup = False

        # Stabilization score
        self.previous_stability_score = 0.0
        self.stability_score = 0.0
        
        self.state = None
        # self.reset()

    def seed(self, seed = None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        random.seed(seed)
        np.random.seed(seed)
        return [seed]
    
    def _get_obs(self):
        ''' 
            Get the observation
        '''
        # if self.total_lookup_attempts > 0:
        #     self.lookup_success_rate = self.successful_lookup_attempts / self.total_lookup_attempts
        # else:
        #     self.lookup_success_rate = 0.0

         # Initialize observations
        active_nodes = np.zeros(self.max_network_size, dtype=np.int8)
        finger_tables = np.full((self.max_network_size, self.r + 1), -1, dtype=np.int32)  # Fill with -1 for inactive nodes


       # Populate active_nodes and finger_tables
        for node_id, node in self.network.node_bank.items():
            if node.is_active:
                active_nodes[node_id] = 1
                successors = node.finger_table.get('successors', [])
                predecessors = node.finger_table.get('predecessors', [])
                finger_entries = successors + predecessors
                # # Pad finger_entries 
                finger_entries += [-1] * (self.r + 1 - len(finger_entries))
                finger_tables[node_id] = np.array(finger_entries)
                
        # self.lookup_success_rate = self._calculate_lookup_success_rate()
        
        observation = {
            'lookup_success_rate': np.array([self.lookup_success_rate], dtype=np.float32),
            'stability_score': np.array([self.stability_score], dtype=np.float32),
            'active_nodes': active_nodes,
            'finger_tables': finger_tables,
        }

        flat_observaton = flatten(self.original_observation_space, observation)
        return flat_observaton

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        
        self.current_step = 0

        # Initialize the network 
        # # Initialize the network    
        self.network = ChordNetwork(n_nodes_to_activate=20, r=2, bank_size=40, verbose=self.verbose)
        self.previous_network_state = None
        self.network_state = None  
         
        # Define max network size and r based on the network
        self.min_network_size = self.network.r+2
        self.max_network_size = self.network.bank_size  # This should be 20
        self.r = self.network.r  # This should be 2
    
        self.previous_stability_score = 0.0
        self.stability_score = 0.0

        # Lookup stats
        self.total_lookup_attempts = 0
        self.failed_lookup_attempts = 0
        self.successful_lookup_attempts = 0
        self.lookup_success_rate = 0.0

        self.is_successful_lookup = False

        # Stabilization score
        self.stability_score = self._calculate_stability_score()
        
        self.state = None
        observation = self._get_obs()

        return observation, {}

    def step(self, action):
        self.current_step += 1

        # Record the stability score before the environment update
        stability_before_update = self._calculate_stability_score()

        # update the environment
        self._update_environment()

        # Calculate stability score after the environment update
        stability_after_update = self._calculate_stability_score()

        # Agent takes an action
        self._take_action(action)
        
        # Calculate stability score after the agent's action
        stability_after_action = self._calculate_stability_score()  
        self.stability_score = stability_after_action

        # Compute reward based on the action and changes in stability
        reward = self._compute_reward(action, stability_before_update, stability_after_update, stability_after_action)

        # Get new observation
        self.state = self._get_obs()

        # Check if the done
        done = self._check_done()

        terminated = done
        truncated = False

        return self.state, reward, terminated, truncated, {}

    def _initialize_network(self):
        ''' 
            State of the network should be set 
        '''
        self.network_state = self.network.get_network_state()
        return self.network_state

    def _take_action(self, action):
        ''' 
            Take an action based on the action space
        '''
        if action == Action.STABILIZE.value:
            self._stabilize()

    def _stabilize(self):
        ''' 
            Stabilization from Chord
            Return stability score. Value between 0 and 1
        '''
        self.network.stabilize()
        

    # def _initiate_lookup(self):
    #     ''' 
    #         Determine if the lookup is successful based on the network state
    #         return True if successful, False otherwise
    #         update look up stats to help determine the lookup success rate
    #     '''
    #     # key should be the id of a node inside the network

    #     key = random.randint(1, self.network.bank_size)
    #     result = self.network.lookup(key)
    #     self.total_lookup_attempts += 1
    #     if result is not None:
    #         self.successful_lookup_attempts += 1
    #         self.is_successful_lookup = True
    #     else:
    #         self.failed_lookup_attempts += 1
    #         self.is_successful_lookup = False

    def _update_environment(self):
        ''' 
            Simulate network dynamics
            Nodes may join or leave, affecting the agent
            50% chances for nodes to join or leave
        '''
        current_network_size = len([n for n in self.network.node_bank.values() if n.is_active])
        while current_network_size < self.min_network_size:
            self.network.join_x_random_nodes(x=1)
            current_network_size = len([n for n in self.network.node_bank.values() if n.is_active])

        if random.random() < 0.33:
            self.network.join_x_random_nodes(x=1)
        elif random.random() < 0.66:
            self.network.drop_x_random_nodes(x=1)

    def _calculate_lookup_success_rate(self):
        ''' 
            Calculate the lookup success rate
        '''
        active_node_ids = [node.id for node in self.network.node_bank.values() if node.is_active]
        total_lookup_attempts = len(active_node_ids)
        successful_lookup_attempts = 0
        seen_nodes = set()
        for key in active_node_ids:
            # key = random.randint(1, self.network.bank_size)
            result = self.network.lookup(key)
            if result is not None:
                successful_lookup_attempts += 1
            seen_nodes.add(key)
        return successful_lookup_attempts / total_lookup_attempts
    
    def _calculate_stability_score(self):
        total_entries = 0
        correct_entries = 0
        active_nodes = [node for node in self.network.node_bank.values() if node.is_active]
        for node in active_nodes:
            ideal_finger_table = self.network.compute_ideal_finger_table(node)
            actual_finger_table = node.finger_table# print(f'Ideal finger table: {ideal_finger_table}')
            # print(f'Actual finger table: {actual_finger_table}')
            # Ensure both lists are of the same length
            max_len = max(len(ideal_finger_table['successors']), len(actual_finger_table['successors']))
            # Pad shorter list with -1
            ideal_finger_table['successors'] += [-1] * (max_len - len(ideal_finger_table['successors']))
            actual_finger_table['successors'] += [-1] * (max_len - len(actual_finger_table['successors']))
                
                # total_entries += max_len
            # Compare entries one by one
            for ideal_entry, actual_entry in zip(ideal_finger_table['successors'], actual_finger_table['successors']):
                total_entries += 1
                if ideal_entry == actual_entry:
                        correct_entries += 1
        if total_entries > 0:
            # print(f'-----------------Stability score: {correct_entries / total_entries}')
            return correct_entries / total_entries
        else:
            return 0.0
        
    def _compute_reward(self, action, stability_before, stability_after_env, stability_after_action):
        ''' 
            Compute reward based on the action taken and the network state
        '''
        reward = 0

        # Determine if the network became less stable after the environment update
        network_became_unstable = stability_after_env < stability_before

        # Determine if the agent's action improved stability
        stability_improved = stability_after_action > stability_after_env

        # Agent's decision and its impact
        if action == Action.STABILIZE.value:
            if stability_improved:
                reward = 1  # Positive reward for improving stability
            else:
                reward = -1  # Penalty for failing to improve stability
                # we need to actually stabilize the network
                # self.network.stabilize()

        elif action == Action.DO_NOTHING.value:
            if network_became_unstable:
                reward = -1  # Penalty for not addressing instability
                # self.network.stabilize()
            else:
                reward = 1  # Positive reward for maintaining stability

        return reward   
    
    def _check_done(self):
        ''' 
            Define conditions for ending the episode
        '''
        if self.current_step >= self.max_steps:
            return True
        return False  # Continuous task

    def close(self):
        ''' 
            Optional: clean up resources
        '''
        pass


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
        n_prime = self.find_predecessor(node, id)
        
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
        if start < end:
            return start <= x <= end
        elif start > end:
            if x >= 0 and x <= end:
                return True
            if x >= start and x < pow(2, self.m):
                return True
            
            return False
        else:
            if x >= start and x <= end:
                return True 
        return False
        
    def _is_in_left_open_right_closed_interval(self, x: int, left: int, right: int, verbose = False) -> bool:
        # left < x <= right  
        # Handle circular interval
        if left < right:
            return left < x <= right
        elif left > right:
            if x >= 0 and x <= right:
                return True
            elif x > left and x <= pow(2, self.m):
                return True
            else:
                return False
        else:
            if x == right:
                return True 
            else:
                return False
        return False
    
    def _is_in_left_closed_right_open_interval(self, x: int, left: int, right: int, verbose = False) -> bool:
        # left <= x < right  
        # Handle circular interval
        if left < right:
            if left <= x < right:
                return True
            else:
                return False
        elif left > right:
            if x >= 0 and x < right:
                return True
            if x >= left and x <= pow(2, self.m):
                return True
        else:
            if x == left:
                return True 
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

            interval = (node.id, successor_node.id)
            if x is not None and self._is_in_open_interval(x, interval[0], interval[1]):    
                node.successor = x

            self.notify(self.node_bank[node.successor], node)
        except Exception as e:
            print(f"Error stabilizing node {node.id}: {str(e)}")
            


    def notify(self, node: Node, n_prime: Node): # ask node to update its predecessor to n_prime
        predecessor = node.predecessor
        if predecessor is None or self._is_in_open_interval(n_prime.id, predecessor, node.id):
            node.predecessor = n_prime.id
        else:
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
        for i in range(1, len(finger_table['start'])):
            print(f'fixing successor at index {i}') 
            print(f'finding successor for identifier {finger_table["start"][i]}')
            successor = self.find_successor(node, finger_table['start'][i])
            print(f'found successor: {successor.id}')
            print('----------------------------------------------------------------')
            finger_table['successor'][i] = successor.id

    def join_network(self, node: Node): # find the immediate successor of node
        print(f'Joining node {node.id} to the network...')
        try:
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

            node.predecessor = self.find_predecessor(n_prime, node.id).id
            
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
            self.leave_network(node_to_drop)

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
    #         successor = network.find_successor(start_node, key)
    #         print(f'successor of key {key} is: {successor.id}')
    #         assert successor.id == actual_successor_id



    # fixing fingers test
    # print('--------------------------------')
    # print('fix fingers test')
    # print('--------------------------------')

    for i in range(10000):
        random_choice_of_active_node = random.choice(active_nodes)
        network.fix_fingers(random_choice_of_active_node)
    network.fix_fingers(network.node_bank[0])
    network.fix_fingers(network.node_bank[1])
    network.fix_fingers(network.node_bank[3])
    network.fix_fingers(network.node_bank[6])
    network.display_network()

