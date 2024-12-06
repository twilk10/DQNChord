import gymnasium as gym 
import numpy as np
from typing import Dict, List
import random
from enum import Enum
from gymnasium.spaces import flatten_space, flatten

import hashlib

def hash_identifier(identifier: str, m: int) -> int:
    """
    Hashes a string identifier using SHA-1 and maps it to the identifier space [0, 2^m - 1].

    Args:
        identifier (str): The unique identifier for the node (e.g., IP address).
        m (int): Number of bits in the identifier space.

    Returns:
        int: The hashed identifier within [0, 2^m - 1].
    """
    hash_bytes = hashlib.sha1(identifier.encode('utf-8')).digest()
    hash_int = int.from_bytes(hash_bytes, byteorder='big')
    return hash_int % (2 ** m)


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
    def __init__(self, id, active_status, predecessor=None, successor=None, finger_table=None):
        self.is_active = active_status
        self.id = id
        self.predecessor = predecessor
        self.successor = successor
        self.finger_table: List[int] = finger_table if finger_table is not None else []
        self.keys: List[int] = []  # Initialize keys list

    def set_active_status(self, new_status):
        self.is_active = new_status

    def __str__(self):
        return (f"\t Node Id: {self.id}\n"
                f"\t Active Status: {self.is_active}\n"
                f"\t Predecessor: {self.predecessor}\n"
                f"\t Successor: {self.successor}\n"
                f"\t Finger Table: {self.finger_table}\n"
                f"\t Keys: {self.keys}\n")


class ChordNetwork:
    def __init__(self,m=4, verbose = False):
        self.m = m
        self.max_network_size = 2 ** self.m
        self.n_nodes_to_activate = self.max_network_size // 2
        self.verbose = verbose
        self.node_bank: Dict[int, Node] = {}

        self.initialize_node_bank()
        self.activate_n_nodes()
        
        if self.verbose:
            print('Initialization Done!')
    
    def initialize_node_bank(self):
        if self.verbose:
            print(f'Initializing node bank with identifier space size {self.max_network_size}...')
        
        for i in range(self.max_network_size):
            if i not in self.node_bank:
                node = Node(id=i, active_status=False)
                self.node_bank[node.id] = node
    
    def activate_n_nodes(self):
        if self.verbose:
            print(f'Activating {self.n_nodes_to_activate} Nodes...')

        # Activate the first 'n' nodes from the node bank
        for i in range(self.n_nodes_to_activate):
            node = self.node_bank[i]
            node.is_active = True

        for i in range(self.n_nodes_to_activate):
            node = self.node_bank[i]
            self.join_network(node)
          
        for i in range(self.n_nodes_to_activate):
            node = self.node_bank[i]
            node.finger_table = [None] * self.m

            # Call fix_fingers to populate finger table
            if self.verbose:
                print(f"Node {node.id}: Fixing fingers...")
            self.fix_fingers(node)
            # Stabilize the node to update predecessor and successor   
    
    def find_successor(self, node:Node, id):# ask Node n to find the successor of id
        print(f'finding successor of {id}')
        predecessor = self.find_predecessor(node, id)
        print(f'predecessor of {id} is: {predecessor.id}')
        successor_id = predecessor.successor
        print(f'successor of {id} is: {successor_id}')
        return self.node_bank[successor_id]
    
    def find_predecessor(self, node:Node, id:int):
        print(f'finding predecessor of {id}')
        n_prime = node
        while not self._is_in_interval(id, n_prime.id, n_prime.successor):
            n_prime = self.closest_preceding_node(n_prime, id)
        print(f'predecessor of {id} is: {n_prime.id}')
        return n_prime
    
    def closest_preceding_node(self, node: Node, id: int):
        print(f'finding closest preceding node of {id}')
        for i in reversed(range(self.m)):  # Use self.m instead of self.r
            finger_id = node.finger_table[i]
            if finger_id is None:
                continue  # Skip if finger is not set
            finger_node = self.node_bank.get(finger_id)
            if finger_node and self._is_in_interval(finger_node.id, node.id, id):
                return finger_node
        return node
    
    def _is_in_interval(self, x:int, start:int, end:int):
        # handle circular interval
        if start < end:
            return start < x < end
        else:
            return x > start or x < end
        
    def stabilize(self, node: Node): # updates predecessor and successor pointers
        successor_node = self.node_bank[node.successor]
        x_id = successor_node.predecessor

        if x_id and self._is_in_interval(x_id, node.id, successor_node.id):
            node.successor = x_id
           
        self.notify(node, successor_node)


    def notify(self, node: Node, successor_node: Node):
        if successor_node.predecessor is None or self._is_in_interval(node.id, successor_node.predecessor, successor_node.id):
            successor_node.predecessor = node.id

   
    def fix_fingers(self, node: Node): # update the finger table of a node
        # Need to look into this better
        for i in range(self.m):
            # start_i is the index of the finger in the finger table
            start_i = (node.id + pow(2, i)) % pow(2, self.m)
            # find the successor of start_i
            successor = self.find_successor(node, start_i)
            # update the finger table
            node.finger_table[i] = successor.id
            if i == 0:
                node.successor = successor.id
            if self.verbose:
                print(f"Node {node.id}: Fixed finger {i} to Node {successor.id}.")
                self.display_network()

   
    def join_network(self, node: Node):
        node.set_active_status(True)
        node.predecessor = None
        # Assuming at least one other active node exists
        print(f'joining node: {node.id}')
        n_prime = random.choice([n for n in self.node_bank.values() if n.is_active and n.id != node.id])
        print(f'chose random node n_prime that will find the successor of {node.id}: {n_prime.id}')
        node.successor = self.find_successor(n_prime, node.id).id
        print(f'found successor of {node.id}: {node.successor}')
        # Initialize finger table
        # self.fix_fingers(node)
        # # Stabilize the node
        # self.stabilize(node)
        # if self.verbose:
        #     print(f"Node {node.id} has joined the network.\n")


    def leave_network_gracefully(self, node: Node): 
        node.set_active_status(False)
        # Clear the node's finger table
        node.predecessor = None
        node.successor = None
        node.finger_table = []
        if self.verbose:
            print(f"Node {node.id} has left the network Gracefully.\n")

    def node_failure(self, node: Node):
        node.set_active_status(False)
        node.predecessor = None
        node.successor = None
        node.finger_table = []
        if self.verbose:
            print(f"Node {node.id} has failed.\n")
    
    def join_x_random_nodes(self, x: int):
        # join x random inactive nodes to the network
        for i in range(x):
            inactive_nodes = [n for n in self.node_bank.values() if not n.is_active]
            if len(inactive_nodes) <= 0:
                continue
            node_to_join = random.choice(inactive_nodes)
            self.join_network(node_to_join)

    def drop_x_random_nodes(self, x: int):
        # drop x random active from the network
        for i in range(x):
            active_nodes = [n for n in self.node_bank.values() if n.is_active]

            if len(active_nodes) <= x:
                raise ValueError(f"Not enough active nodes to drop {x} nodes.")
            
            node_to_drop = random.choice(active_nodes)
            # We cannot drop the Node with id 0
            while node_to_drop.id == 0:
                node_to_drop = random.choice(active_nodes)
            self.leave_network(node_to_drop)

    def get_network_state(self):
        network_state = {}
        for node in self.node_bank.values():
            if node.is_active:
                network_state[node.id] = node.finger_table
        return network_state

    def display_network(self):
        print(" Network state:")
        for node in self.node_bank.values():
            if node.is_active:
                print(node)


if __name__ == "__main__":
    network = ChordNetwork(m= 4, verbose=True)
    # network.display_network()
    new_node = Node(id=6, active_status=False)
    network.join_network(new_node)
    network.display_network()