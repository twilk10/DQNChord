import gymnasium as gym 
import numpy as np
from typing import Dict, List
import random
from enum import Enum
from gymnasium.spaces import flatten_space, flatten

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
    def __init__(self, id, active_status, predecessor = None, successor = None, finger_table = []):
        self.is_active = active_status
        self.id = id
        self.predecessor = predecessor
        self.successor = successor
        self.finger_table: List[int] = finger_table
    
    def set_active_status(self, new_status):
        self.is_active = new_status
    def __str__(self):
        return (f"\t Node Id: {self.id}\n"
                f"\t Active Status: {self.is_active}\n"
                f"\t Finger Table: {self.finger_table}\n")

class ChordNetwork:
    def __init__(self,n_nodes_to_activate, r, bank_size, verbose = False):
        self.n_nodes_to_activate = n_nodes_to_activate
        self.bank_size = bank_size
        self.verbose = verbose
        self.r = r # number of successor nodes a node can have
        self.node_bank: Dict[int, Node] = self.initialize_node_bank(bank_size)
        self.activate_n_nodes(n_nodes_to_activate, r)

        # print('Node bank:')
        # for node in self.node_bank.values():
        #     print(node)
        if self.verbose:
            print('Initialization Done!')
    
    def initialize_node_bank(self, bank_size):
        if self.verbose:
            print(f'Initializing node bank of size {bank_size}...')
        bank = {}
        for i in range(bank_size):
            if i not in bank:
                node = Node(id=i, active_status=False)
                bank[node.id] = node
        return bank
    
    def activate_n_nodes(self, n_nodes_to_activate, r):
        if self.verbose:
            print(f'Activating {n_nodes_to_activate} Nodes...')

        # Activate the first 'n' nodes from the node bank
        for i in range(n_nodes_to_activate):
            node = self.node_bank[i]
            node.is_active = True

        for i in range(n_nodes_to_activate):
            node = self.node_bank[i]
            self.assign_initial_successors_and_predecessors(node, r, initial_run= True)
           

    def assign_initial_successors_and_predecessors(self, node: Node, r: int, initial_run = False):
        # Get the list of active node IDs in ascending order
        active_nodes_ids = sorted([n.id for n in self.node_bank.values() if n.is_active])
        total_number_of_active_nodes = len(active_nodes_ids)

        # Validate input
        if not initial_run and node.id not in active_nodes_ids:
            raise ValueError(f"Node {node.id} is not active or missing from the active nodes.")
        if not initial_run and r > total_number_of_active_nodes:
            print('Active nodes:', active_nodes_ids)
            raise ValueError(f"Not enough active nodes to assign {r} successors/predecessors.")
        
        # Get the index of the current node in the sorted active nodes list
        node_index = active_nodes_ids.index(node.id)
        
        # Assign successors (based on Chord's finger table definition)
        node.finger_table = []
        for i in range(r):
            successor_index = (node_index + pow(2, i)) % total_number_of_active_nodes
            successor_node_id = active_nodes_ids[successor_index]
            node.finger_table.append(successor_node_id)
        
        # Assign predecessor (only one predecessor for this use case)
        predecessor_index = (node_index - 1) % total_number_of_active_nodes
        predecessor_node_id = active_nodes_ids[predecessor_index]
        node.predecessor = predecessor_node_id
    
    def find_successor(self, node:Node, id):# ask Node n to find the successor of id
        predecessor = self.find_predecessor(node, id)
        return predecessor.successor
       
    def find_predecessor(self, node:Node, id:int):
        n_prime = node
        while id not in n_prime.finger_table['successors']:
            n_prime = self.closest_preceding_node(n_prime, id)
        return n_prime
    
    def closest_preceding_node(self, n: Node, id:int):
        sorted_successors = sorted(n.finger_table['successors'])
        for successor in sorted_successors:
            if successor > id:
                return successor
        return sorted_successors[-1]
    
    def stabilize(self, current_node: Node): # updates predecessor and successor pointers
        n_s = current_node.successor

        # ask n_s for its predecessor
        n_s_predecessor: Node = self.node_bank[n_s.id].predecessor # n_s returns n as its predecessor

        # check if n_s_predeccessor should be a successor of the current node
        if n_s_predecessor.id != current_node.id:
            # update the current node's successor to the n_s_predecessor that was returned 
            current_node.successor = n_s_predecessor.id
            current_node.finger_table.append(n_s_predecessor.id) # add to the current node's finger table, The order of this table will be updated by fix_fingers()   
        
            # notifies n_s_predecessor that its predecessor has changed to current node
            self.notify(current_node, n_s_predecessor) 


    def notify(self,n_prime: Node,  node: Node ): # Notifies the node that n_prime is now its predecessor
        node.predecessor = n_prime.id
        if self.verbose:
            print(f"Node {node.id} has been notified that Node {n_prime.id} is now its predecessor.\n") 
   
    def fix_fingers(self, node: Node):
        # Need to look into this better
        for finger_index in range(len(node.finger_table)):
            finger = node.finger_table[finger_index]
            successor = self.find_successor(node, finger)
            node.finger_table[finger_index] = successor
   
    def join_network(self, node: Node):
        node.set_active_status(True)
        node.predecessor = None
        n_prime = random.choice([n for n in self.node_bank.values() if n.is_active])
        successor = self.find_successor(n_prime, node.id)
        node.successor = successor

        if self.verbose:
            print(f"Node {node.id} has joined the network.\n")

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

    def compute_ideal_finger_table(self, node: Node):
        active_nodes_ids = sorted([n.id for n in self.node_bank.values() if n.is_active])
        total_number_of_active_nodes = len(active_nodes_ids)
        
        node_index = active_nodes_ids.index(node.id)
        ideal_finger_table = []
        
        # Compute ideal successors
        for i in range(self.r):
            successor_index = (node_index + pow(2, i)) % total_number_of_active_nodes
            successor_node_id = active_nodes_ids[successor_index]
            ideal_finger_table.append(successor_node_id)
        
        predecessor_index = (node_index - 1) % total_number_of_active_nodes
        predecessor_node_id = active_nodes_ids[predecessor_index]
        
        
        return ideal_finger_table, predecessor_node_id


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
