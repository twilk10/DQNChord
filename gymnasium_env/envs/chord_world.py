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
    FIX_FINGERS = 1
    DO_NOTHING = 2

class ChordWorldEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self, 
                 max_steps = 100, 
                 seed = 42, 
                 stabalize_interval = 10, 
                 fix_fingers_interval = 15, 
                 agent_id = 0, 
                 churn_join_prob = 0.2, 
                 churn_drop_prob = 0.2):
        super(ChordWorldEnv, self).__init__()

        self.seed(seed)
        self.max_steps = max_steps
        self.current_step = 0

        self.stabalize_interval = stabalize_interval
        self.fix_fingers_interval = fix_fingers_interval
        self.agent_id = agent_id
        self.churn_join_prob = churn_join_prob
        self.churn_drop_prob = churn_drop_prob
        self.verbose = False

        self.network = ChordNetwork(m=4, verbose=self.verbose, seed=seed)

        if not self.network.node_bank[self.agent_id].is_active:
            self.network.join_network(self.network.node_bank[self.agent_id])

        self.action_space = gym.spaces.Discrete(3)

        max_node_id = self.network.max_network_size - 1
        self.observation_space = gym.spaces.Box(
            low=np.array([0,0,0.0], dtype=np.float32),
            high=np.array([max_node_id,max_node_id,1.0], dtype=np.float32),
            dtype=np.float32
        )

        self.state = self._get_obs()

        # self.previous_network_state = None
        # self.network_state = None  
         
        # # Define max network size and r based on the network
        # self.min_network_size = 2
        # self.max_network_size = self.network.max_network_size
        # self.r = self.network.r  # This should be 2
        
        

        # self.original_observation_space = gym.spaces.Dict({
        #     'stability_score': gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
        #     'active_nodes': gym.spaces.MultiBinary(self.network.bank_size),
        #     'finger_tables': gym.spaces.Box(
        #         low=0, high=self.max_network_size - 1,
        #         shape=(self.max_network_size, self.r +1 ),
        #         dtype=np.int32
        #         ),
        #     })
        
        # self.observation_space = flatten_space(self.original_observation_space)

        # # Lookup stats
        # self.total_lookup_attempts = 0
        # self.failed_lookup_attempts = 0
        # self.successful_lookup_attempts = 0
        # # self.lookup_success_rate = 0.0

        # self.is_successful_lookup = False

        # # Stabilization score
        # self.previous_stability_score = 0.0
        # self.stability_score = 0.0
        

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
        # active_nodes = np.zeros(self.max_network_size, dtype=np.int8)
        # finger_tables = np.full((self.max_network_size, self.r + 1), -1, dtype=np.int32)  # Fill with -1 for inactive nodes
         # Agent-only local observation
        agent_node = self.network.node_bank[self.agent_id]

        # Local stability indicator: how correct is the agent’s immediate successor?
        local_stability = self._local_stability_indicator(agent_node)

        pred = agent_node.predecessor if agent_node.predecessor is not None else 0
        succ = agent_node.successor if agent_node.successor is not None else 0

        obs = np.array([pred, succ, local_stability], dtype=np.float32)
        return obs


    #    # Populate active_nodes and finger_tables
    #     for node_id, node in self.network.node_bank.items():
    #         if node.is_active:
    #             active_nodes[node_id] = 1
    #             successors = node.finger_table.get('successors', [])
    #             predecessors = node.finger_table.get('predecessors', [])
    #             finger_entries = successors + predecessors
    #             # # Pad finger_entries 
    #             finger_entries += [-1] * (self.r + 1 - len(finger_entries))
    #             finger_tables[node_id] = np.array(finger_entries)
                
    #     # self.lookup_success_rate = self._calculate_lookup_success_rate()
        
    #     observation = {
    #         'lookup_success_rate': np.array([self.lookup_success_rate], dtype=np.float32),
    #         'stability_score': np.array([self.stability_score], dtype=np.float32),
    #         'active_nodes': active_nodes,
    #         'finger_tables': finger_tables,
    #     }

    #     flat_observaton = flatten(self.original_observation_space, observation)
    #     return flat_observaton

    def _local_stability_indicator(self, node):
        # Simple heuristic: if agent’s successor’s predecessor is agent, local stability = 1.0 else <1.0
        if node.successor is None:
            return 0.0
        succ_node = self.network.node_bank[node.successor]
        if succ_node.predecessor == node.id:
            return 1.0
        else:
            return 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.network = ChordNetwork(m=4, verbose=self.verbose, seed=seed)
        if not self.network.node_bank[self.agent_id].is_active:
            self.network.join_network(self.network.node_bank[self.agent_id])
        self.state = self._get_obs()
        return self.state, {}

    def step(self, action):
        self.current_step += 1

        # Record local stability before environment updates
        stability_before = self._local_stability_indicator(self.network.node_bank[self.agent_id])

        # Update environment: churn and non-agent nodes stabilize/fix_fingers at intervals
        self._update_environment()

        # Agent takes action
        if action == Action.STABILIZE.value:
            self.network.stabilize(self.network.node_bank[self.agent_id])
        elif action == Action.FIX_FINGERS.value:
            self.network.fix_fingers(self.network.node_bank[self.agent_id])
        elif action == Action.DO_NOTHING.value:
            pass

        # Compute reward based on local stability changes and minimizing wasted actions
        stability_after = self._local_stability_indicator(self.network.node_bank[self.agent_id])
        reward = self._compute_reward(action, stability_before, stability_after)

        # New observation
        self.state = self._get_obs()

        # Check if done
        done = (self.current_step >= self.max_steps)
        terminated = done
        truncated = False

        return self.state, reward, terminated, truncated, {}
        # self.current_step += 1

        # # Record the stability score before the environment update
        # stability_before_update = self._calculate_stability_score()

        # # update the environment
        # self._update_environment()

        # # Calculate stability score after the environment update
        # stability_after_update = self._calculate_stability_score()

        # # Agent takes an action
        # self._take_action(action)
        
        # # Calculate stability score after the agent's action
        # stability_after_action = self._calculate_stability_score()  
        # self.stability_score = stability_after_action

        # # Compute reward based on the action and changes in stability
        # reward = self._compute_reward(action, stability_before_update, stability_after_update, stability_after_action)

        # # Get new observation
        # self.state = self._get_obs()

        # # Check if the done
        # done = self._check_done()

        # terminated = done
        # truncated = False

        # return self.state, reward, terminated, truncated, {}

    def _compute_reward(self, action, stability_before, stability_after):
        ''' 
            Compute reward based on the action taken and the network state
        '''
        reward = 0.0
        stability_improved = stability_after > stability_before
        stability_declined = stability_after < stability_before
        stability_same = (stability_after == stability_before)

        # Example reward scheme:
        if action == Action.STABILIZE.value:
            if stability_improved:
                reward += 1.0
            else:
                reward -= 0.5
        elif action == Action.FIX_FINGERS.value:
            if stability_improved:
                reward += 1.0
            else:
                reward -= 0.5
        elif action == Action.DO_NOTHING.value:
            if stability_declined:
                reward -= 1.0
            else:
                # If stability stayed the same or improved without any action:
                reward += 1.0

        return reward

        # # Determine if the network became less stable after the environment update
        # network_became_unstable = stability_after_env < stability_before

        # # Determine if the agent's action improved stability
        # stability_improved = stability_after_action > stability_after_env

        # # Agent's decision and its impact
        # if action == Action.STABILIZE.value:
        #     if stability_improved:
        #         reward = 1  # Positive reward for improving stability
        #     else:
        #         reward = -1  # Penalty for failing to improve stability
        #         # we need to actually stabilize the network
        #         # self.network.stabilize()

        # elif action == Action.DO_NOTHING.value:
        #     if network_became_unstable:
        #         reward = -1  # Penalty for not addressing instability
        #         # self.network.stabilize()
        #     else:
        #         reward = 1  # Positive reward for maintaining stability

        # return reward   

    def _update_environment(self):
        # ''' 
        #     Simulate network dynamics
        #     Nodes may join or leave, affecting the agent
        #     50% chances for nodes to join or leave
        # '''
        # current_network_size = len([n for n in self.network.node_bank.values() if n.is_active])
        # while current_network_size < self.min_network_size:
        #     self.network.join_x_random_nodes(x=1)
        #     current_network_size = len([n for n in self.network.node_bank.values() if n.is_active])

        # if random.random() < 0.33:
        #     self.network.join_x_random_nodes(x=1)
        # elif random.random() < 0.66:
        #     self.network.drop_x_random_nodes(x=1)
        # This simulates a second passing
        # Churn simulation

        r = random.random()
        if r < self.churn_join_prob:
            self.network.join_x_random_nodes(x=1)
        elif r < self.churn_join_prob + self.churn_drop_prob:
            # drop nodes only if enough active nodes available
            if len([n for n in self.network.node_bank.values() if n.is_active]) > 1:
                self.network.drop_x_random_nodes(x=1)

        # Other nodes stabilize/fix fingers at intervals
        for node_id, node in self.network.node_bank.items():
            if node_id == self.agent_id:
                continue  # Agent handled separately
            if node.is_active:
                # Non-agent nodes stabilize every stabilize_interval steps
                if self.current_step % self.stabalize_interval == 0:
                    self.network.stabilize(node)
                # Non-agent nodes fix_fingers every fix_fingers_interval steps
                if self.current_step % self.fix_fingers_interval == 0:
                    self.network.fix_fingers(node)


    # def _initialize_network(self):
    #     ''' 
    #         State of the network should be set 
    #     '''
    #     self.network_state = self.network.get_network_state()
    #     return self.network_state

    # def _take_action(self, action):
    #     ''' 
    #         Take an action based on the action space
    #     '''
    #     if action == Action.STABILIZE.value:
    #         self._stabilize()

    # def _stabilize(self):
    #     ''' 
    #         Stabilization from Chord
    #         Return stability score. Value between 0 and 1
    #     '''
    #     self.network.stabilize()
        

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

    # def _calculate_lookup_success_rate(self):
    #     ''' 
    #         Calculate the lookup success rate
    #     '''
    #     active_node_ids = [node.id for node in self.network.node_bank.values() if node.is_active]
    #     total_lookup_attempts = len(active_node_ids)
    #     successful_lookup_attempts = 0
    #     seen_nodes = set()
    #     for key in active_node_ids:
    #         # key = random.randint(1, self.network.bank_size)
    #         result = self.network.lookup(key)
    #         if result is not None:
    #             successful_lookup_attempts += 1
    #         seen_nodes.add(key)
    #     return successful_lookup_attempts / total_lookup_attempts
    
    # def _calculate_stability_score(self):
    #     total_entries = 0
    #     correct_entries = 0
    #     active_nodes = [node for node in self.network.node_bank.values() if node.is_active]
    #     for node in active_nodes:
    #         ideal_finger_table = self.network.compute_ideal_finger_table(node)
    #         actual_finger_table = node.finger_table# print(f'Ideal finger table: {ideal_finger_table}')
    #         # print(f'Actual finger table: {actual_finger_table}')
    #         # Ensure both lists are of the same length
    #         max_len = max(len(ideal_finger_table['successors']), len(actual_finger_table['successors']))
    #         # Pad shorter list with -1
    #         ideal_finger_table['successors'] += [-1] * (max_len - len(ideal_finger_table['successors']))
    #         actual_finger_table['successors'] += [-1] * (max_len - len(actual_finger_table['successors']))
                
    #             # total_entries += max_len
    #         # Compare entries one by one
    #         for ideal_entry, actual_entry in zip(ideal_finger_table['successors'], actual_finger_table['successors']):
    #             total_entries += 1
    #             if ideal_entry == actual_entry:
    #                     correct_entries += 1
    #     if total_entries > 0:
    #         # print(f'-----------------Stability score: {correct_entries / total_entries}')
    #         return correct_entries / total_entries
    #     else:
    #         return 0.0
        
    # def _check_done(self):
    #     ''' 
    #         Define conditions for ending the episode
    #     '''
    #     if self.current_step >= self.max_steps:
    #         return True
    #     return False  # Continuous task

    def close(self):
        ''' 
            Optional: clean up resources
        '''
        pass



'''''

Chord Network Classes
I have created two classes for the Chord Network: Node and ChordNetwork.

The Node class represents a node in the Chord network, including its ID, active status, predecessor, successor, and finger table.
The ChordNetwork class represents the entire Chord network, including the node bank, nodes to activate, and various methods for 
initializing the network, activating nodes, and stabilizing the network.    

There are also some helper functions that are used to test the correctness of the finger tables. This is contained in the main function.
The tests are commented out in the code.

'''
class Node:
    def __init__(self, 
                 id: int, 
                 active_status: bool, 
                 predecessor: Optional[int] = None, 
                 successor: Optional[int] = None, 
                 finger_table: Optional[List[int]] = None, 
                 m: int = 0):
        self.is_active = active_status
        self.id = id
        self.predecessor = predecessor
        self.successor = successor
        if finger_table is None:
            self.finger_table : Dict[str, List[int]] = {
                'start': [0]* m,
                'interval': [0]*m,
                'successor': [0]*m
            }
        else:
            self.finger_table = finger_table

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
        # print(f'identifier_successor_id_map is: {identifier_successor_id_map}')
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
        # print(f'+++++ received predecessor of {id} as: {n_prime.id}')
      
        
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
        print(f'INTERVAL[0]: {interval[0]}, INTERVAL[1]: {interval[1]}, n_prime_id: {n_prime.id}, n_prime_succ: {n_prime.finger_table['successor'][0]}')
        while not self._is_in_left_open_right_closed_interval(id, interval[0], interval[1]):
           
            # check if the id is in nprime's interval   
            interval = (n_prime.id, n_prime.finger_table['successor'][0])
            print(f'INTERVAL[0]: {interval[0]}, INTERVAL[1]: {interval[1]}, n_prime_id: {n_prime.id}, n_prime_succ: {n_prime.finger_table['successor'][0]}')
            if self._is_in_closed_interval(id, interval[0], interval[1], verbose = False):
                if id == n_prime.id:
                    return self.node_bank[n_prime.predecessor]
                return n_prime
            
            returned_node = self.closest_preceding_node(n_prime, id)
            print(f'n_prim: {n_prime}, id: {id}')
            n_prime = returned_node
            interval = (returned_node.id, returned_node.successor)
            print(f'returned_node: {returned_node},interval: {interval}')

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
        print(f'LEFT: {left}, RIGHT: {right}')
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
        # print(f'stabilizing node {node.id}...')
        
        
        try:
            # ask the node's successor for its predecessor
            successor_node = self.node_bank[node.finger_table['successor'][0]]
            if not successor_node.is_active:
                print(f'successor node {successor_node.id} is not active')
                for i in range(self.m):
                    successor_id = node.finger_table['successor'][i]
                    successor_node = self.node_bank[successor_id]
                    if successor_node.is_active:
                        break   
                
                successor_node.predecessor = node.id
                node.successor = successor_node.id
                node.finger_table['successor'][0] = successor_node.id   
            else:  
                x = successor_node.predecessor
                # print(f'x is: {x}') 

                interval = (node.id, successor_node.id)
                print(f'interval is: {interval}')   
                if x is not None and self._is_in_open_interval(x, interval[0], interval[1]):
                    print('in if statement')
                    node.successor = x
                    node.finger_table['successor'][0] = x

            # print(f' {node.successor} notifies {node.id} that it is {node.id}\'s successor')
            self.notify(self.node_bank[node.successor], node)
            
        except Exception as e:
            print(f"Error stabilizing node {node.id}: {str(e)}")
            


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
        # print(f'fixing fingers for node {node.id}...')
      
        if not node.finger_table:
            node.finger_table = {
                'start': [0]* self.m,
                'interval': [0]*self.m,
                'successor': [0]*self.m
            }
            for i in range(self.m):
                start = (node.id + pow(2, i)) % pow(2, self.m)
                node.finger_table['start'][i] = start
                node.finger_table['interval'][i] = (start, (node.id + pow(2, i+1)) % pow(2, self.m))

        
        # fixing the successor nodes randomly to account for periods of instability 
        finger_table = node.finger_table
        i = random.randint(1, len(finger_table['start']) - 1)
        successor = self.find_successor(node, finger_table['start'][i])
        print(f'successor: {successor}')
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
        for index, identifier in enumerate(sorted_node_ids):
            idx = index
            while not self.node_bank[sorted_node_ids[idx]].is_active:
                idx = (idx + 1) % len(sorted_node_ids)
            identifier_successor_id_map[identifier] = sorted_node_ids[idx]
                
        # initialize the successor nodes in the finger tables of the given node
        print(f'identifier_successor_id_map is: {identifier_successor_id_map}')
        for i in range(self.m):
            start = finger_table['start'][i]
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
            # choose a node to use to help init the joining node's finger table
            n_prime = random.choice(active_nodes)
            print(f'n_prime that is chosen is: {n_prime.id}')  

            self.init_finger_table(node, n_prime) # n_prime is the node that is helping to init the finger table of the joining node
            node.set_active_status(True)
            
        except Exception as e:
            print(f"Error joining node {node.id} to network: {str(e)}")
    
    def init_finger_table(self, joining_node: Node, n_prime: Node):
        joining_node.finger_table = {
            'start': [0]* self.m,
            'interval': [0]*self.m,
            'successor': [0]*self.m
        }

        for i in range(self.m):
           start = (joining_node.id + pow(2, i)) % pow(2, self.m)
           joining_node.finger_table['start'][i] = start
           joining_node.finger_table['interval'][i] = (start, (joining_node.id + pow(2, i+1)) % pow(2, self.m)) 
        

        joining_node_successor = self.find_successor(n_prime, joining_node.finger_table['start'][0])
        joining_node.finger_table['successor'][0] = joining_node_successor.id
        joining_node.successor = joining_node_successor.id
        joining_node.finger_table['successor'][0] = joining_node_successor.id   
        joining_node.predecessor = joining_node_successor.predecessor

        finger_table = joining_node.finger_table
        for i in range(1,self.m):
            successor = self.find_successor(n_prime, finger_table['start'][i])
            finger_table['successor'][i] = successor.id    


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
  

'''
Tests:
Can be used to test the correctness of the finger tables.
Used to test the stabilization and fix_fingers functions when a node joins or leaves the network.
'''
if __name__ == "__main__":
    # Creating a chord network based on the example in the paper: https://pdos.csail.mit.edu/papers/chord:sigcomm01/chord_sigcomm.pdf
    m = 3
    nodes_to_activate = [0, 1, 3]   
    network = ChordNetwork(m=m, nodes_to_activate=nodes_to_activate, verbose=True)
    network.display_network()
        
        
    print('--------------------------------')
    print('TEST FINDING THE PREDECESSOR OF AN IDENTIFIER')
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
    print('--------------------------------')
    print('TEST FINDING THE SUCCESSOR OF AN IDENTIFIER')
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
    print('TEST JOINING A NEW NODE TO THE NETWORK')
    new_node = Node(id=6, active_status=False)
    network.join_network(new_node)
    print('After joining new node, network state is:')
    network.display_network()
    
    # Randomly stabilizing nodes in the network for 5000 iterations
    print('Randomly stabilizing nodes in the network for 5000 iterations...')
    active_nodes = [n for n in network.node_bank.values() if n.is_active]
    for i in range(5000):
        random_choice_of_active_node = random.choice(active_nodes)
        network.stabilize(random_choice_of_active_node)
    
    network.display_network()

    # assert that the network is stable
    print('asserting that the network is stable (Pointers only since the finger tables are not fixed by the stabilizing function)...')
    active_nodes = [n for n in network.node_bank.values() if n.is_active]
    for node in active_nodes:
        if node.id == 0:
            assert node.predecessor == 6
            assert node.successor ==  1
            assert node.finger_table['successor'][0] == 1       

        if node.id == 1:
            assert node.predecessor == 0
            assert node.successor == 3
            assert node.finger_table['successor'][0] == 3

        if node.id == 3:
            assert node.predecessor == 1
            assert node.successor == 6
            assert node.finger_table['successor'][0] == 6

        if node.id == 6:
            assert node.predecessor == 3
            assert node.successor == 0
            assert node.finger_table['successor'][0] == 0

    network.display_network()
    print('All pointers updated, Network is stable!')

    # fixing fingers test
    print('--------------------------------')
    print('FIXING FINGERS TEST')

    for i in range(10000):
        random_choice_of_active_node = random.choice(active_nodes)
        network.fix_fingers(random_choice_of_active_node)

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
    print('TEST FINDING THE PREDECESSOR OF AN IDENTIFIER WITH THE NEW NETWORK STATE')
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
    print('find_predecessor works fine in the new network state!')  

    print('--------------------------------')
    print('TEST FINDING THE SUCCESSOR OF AN IDENTIFIER WITH THE NEW NETWORK STATE')
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
            # print(f'successor of key {key} is: {successor.id}')
            assert successor.id == actual_successor_id
    print('All successors are correct in the new network state!')

  
    print('--------------------------------')
    print('TEST WHEN A NODE LEAVES THE NETWORK...')
    id_of_node_to_leave = 1
    active_nodes = [n for n in network.node_bank.values() if n.is_active]   
    initial_size_of_network = len(active_nodes)    
    network.leave_network(network.node_bank[id_of_node_to_leave])
    assert network.node_bank[id_of_node_to_leave].is_active == False
    active_nodes = [n for n in network.node_bank.values() if n.is_active]       
    assert len(active_nodes) == initial_size_of_network - 1
    # network.display_network()

    print('Stabilizing the network after a node leaves...')
    network.stabilize(network.node_bank[0])
    # active_nodes = [n for n in network.node_bank.values() if n.is_active]
    # for i in range(5000):
    #     random_choice_of_active_node = random.choice(active_nodes)
    #     network.stabilize(random_choice_of_active_node)

    network.display_network() 
    # for node in active_nodes:
    #     if node.id == 0:
    #         assert node.predecessor == 6
    #         assert node.successor ==  0
    #         assert node.finger_table['successor'][0] == 0   

    #     if node.id == 3:
    #         assert node.predecessor == 1
    #         assert node.successor == 6
    #         assert node.finger_table['successor'][0] == 6

    #     if node.id == 6:
    #         assert node.predecessor == 3
    #         assert node.successor == 0
    #         assert node.finger_table['successor'][0] == 0

    # network.display_network() 


