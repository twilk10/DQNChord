import os
import sys
import gymnasium as gym 
import numpy as np
from typing import List, Dict, Optional
import random
from enum import Enum

# network = ChordNetwork(size=10, r=2, bank_size=20)
# network.display_network()

class Action(Enum):
    STABALIZE = 0,
    LOOKUP = 1,

class ChordWorldEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self):
        super(ChordWorldEnv, self).__init__()
        
        self.action_space = gym.spaces.Discrete(2)  # Actions 0 to 2

        self.observation_space = gym.spaces.Dict({
            'lookup_success_rate': gym.spaces.Box(low=0.0, high=1.0, shape=()),
        })

        # Initialize the network    
        self.network = ChordNetwork()

        self.lookup_success_rate = 1.0

        self.state = None
        self.reset()

    def _get_obs(self):
        return {
            'lookup_success_rate': self.lookup_success_rate,
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # self.state = self._initialize_state()
        self.lookup_success_rate = 1.0
        # reset network func or:
        self.network_state = self._initialize_network()
        observation = self._get_obs()

        return observation

    def step(self, action):
        # agent will take an action
        stability_score, is_successful_lookup = self._take_action(action)

        # update the netwrk
        self._update_environment()

        # compute reward
        reward = self._compute_reward(action, stability_score, is_successful_lookup)

        # Get new observation
        self.state = self._get_obs()

        # Check if the done
        done = self._check_done()

        terminated = done
        truncated = False

        return self.state, reward, terminated, truncated, {}

    def _initialize_network(self):
        # State of the network should be set randomly
        pass

    def _take_action(self, action):
        if action == Action.STABALIZE.value:
            stability_score = self._stabilize()
        elif action == Action.LOOKUP.value:
            is_successful_lookup = self._initiate_lookup()

        return stability_score, is_successful_lookup

    def _stabilize(self):
        ''' 
            stabalization from chord
            # return stability score. Value between 0 and 1
        '''
        self.network.stabilize()
        actual_network_state = self.network.get_network_state()

        # TODO: The dqn agent needs to update the network state
        expected_network_state = self.network_state

        # compare the actual network state with the expected network state after the agent has updated it
        count = 0
        for node_id in actual_network_state:
            # check if the finger table of the node is the same in both states
            actual_finger_table = actual_network_state[node_id]
            expected_finger_table = expected_network_state[node_id]
            if actual_finger_table == expected_finger_table:
                count += 1
        return count / len(actual_network_state)
       

    def _initiate_lookup(self):
        # Determine if the lookup is successful based on the network state
        # return True if successful, False otherwise
        key = random.randint(0, 100)
        result = self.network.lookup(key)
        if result is not None:
            return True
        return False

    def _update_environment(self):
        # Simulate network dynamics
        # Nodes may join or leave, affecting the agent
        # 50% chances for nodes to join or leave
        if random.random() < 0.5:
            self.network.join_x_random_nodes(1)
        else:
            self.network.drop_x_random_nodes(1)
        

    def _compute_reward(self, action, stability_score, is_successful_lookup):
        if action == Action.LOOKUP.value:
            reward = 1 if is_successful_lookup else -1
        elif action == Action.STABALIZE.value:
            reward = 1 if stability_score > 0.8 else -1
        else:
            reward = 0
        return reward
    
    def _check_done(self):
        # Define conditions for ending the episode
        return False  # Continuous task

    def close(self):
        # Optional: clean up resources
        pass


class Node:
    def __init__(self, id, active_status):
        self.is_active = active_status
        self.id = id
        self.is_agent = True if self.id == 0 else False
        self.finger_table = {
            'predecessors': [],
            'successors': []
        }
        self.data={}
        
    def set_active_status(self, new_status):
        self.is_active = new_status
    def __str__(self):
        return (f"\t Node Id: {self.id}\n"
                f"\t Active Status: {self.is_active}\n"
                f"\t Finger Table: {self.finger_table}\n")

class ChordNetwork:
    def __init__(self,size, r, bank_size, verbose = False):
        self. verbose = verbose
        self.r = r # number of successor nodes a node can have
        self.node_bank: Dict[int, Node] = self.initialize_node_bank(bank_size)
        self.initialize_graph(size, r)
        if self.verbose:
            print('Initialization Done!')
    
    def initialize_node_bank(self, bank_size):
        if self.verbose:
            print('Initializing node bank...')
        bank = {}
        for i in range(bank_size):
            if i not in bank:
                node = Node(id=i, active_status=False)
                bank[node.id] = node
        return bank
    
    def initialize_graph(self, size, r):
        if self.verbose:
            print('initializing graph...')

        # Activate the first 'size' nodes from the node bank
        for i in range(size):
            node = self.node_bank[i]
            node.is_active = True

        for i in range(size):
            node = self.node_bank[i]
            self.assign_successors_and_predecessors(node, r, initial_run= True)
           

    def assign_successors_and_predecessors(self, node: Node, r: int, initial_run = False):
        # Get the list of active node IDs in ascending order
        active_nodes_ids = sorted([n.id for n in self.node_bank.values() if n.is_active])
        total_number_of_active_nodes = len(active_nodes_ids)

        # Validate input
        if not initial_run and node.id not in active_nodes_ids:
            raise ValueError(f"Node {node.id} is not active or missing from the active nodes.")
        if not initial_run and r > total_number_of_active_nodes:
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

    def drop_x_random_nodes(self, x: int):
        # drop x random active from the network
        active_nodes = [n for n in self.node_bank.values() if n.is_active]
        for i in range(x):
            node_to_drop = random.choice(active_nodes)
            self.leave_network(node_to_drop)
            active_nodes.remove(node_to_drop)

    def join_x_random_nodes(self, x: int):
        # join x random inactive nodes to the network
        inactive_nodes = [n for n in self.node_bank.values() if not n.is_active]
        for i in range(x):
            node_to_join = random.choice(inactive_nodes)
            self.join_network(node_to_join)
            inactive_nodes.remove(node_to_join)

    def lookup(self, key: int):
        if self.verbose:
            print(f"Starting lookup for Node ID {key} at Node {0}")

        if key == 0:
            return 0

        node_0_finger_table = self.node_bank[0].finger_table
        if key in node_0_finger_table['successors']:
            return node_0_finger_table['successors'][key]

        network_size = len([n for n in self.node_bank.values() if n.is_active])
        max_hops = network_size - 1
        # find successor that is closest to the key in the succesor list
        sorted_successors = sorted(node_0_finger_table['successors'])
        if key > sorted_successors[-1]:
            return self._lookup_helper(key, sorted_successors[-1], max_hops - 1)
        elif key > sorted_successors[0] and key < sorted_successors[-1]:  # within range
            closest_preceding_node = node_0_finger_table['successors'][0]
            for successor in node_0_finger_table['successors']:
                    if successor > key:
                        break
                    if abs(successor - key) < abs(closest_preceding_node - key):
                        closest_preceding_node = successor
            return self._lookup_helper(key, closest_preceding_node, max_hops - 1)
        return None
    
    def _lookup_helper(self, key: int, node_id: int, max_hops: int):
        if node_id is None or max_hops < 0:
            return None

        finger_table = self.node_bank[node_id].finger_table
        if key in finger_table['successors']:
            return (node_id)

        sorted_successors = sorted(finger_table['successors'])
        if key > sorted_successors[-1]:
            return self._lookup_helper(key, sorted_successors[-1], max_hops - 1)
        elif key > sorted_successors[0] and key < sorted_successors[-1]:  # within range
            closest_preceding_node = sorted_successors[0]
            for successor in sorted_successors:
                if key > successor and abs(successor - key) < abs(closest_preceding_node - key):
                    closest_preceding_node = successor
            return self._lookup_helper(key, closest_preceding_node, max_hops - 1)
        return None

    def join_network(self, node: Node):
        node.set_active_status(True)
        self.assign_successors_and_predecessors(node, self.r)
        self.stabilize()
        if self.verbose:
            print(f"Node {node.id} has joined the network.\n")

    def leave_network(self, node: Node): 
        node.set_active_status(False)
        # Clear the node's finger table
        node.finger_table = {'predecessors': [], 'successors': []}
            
        self.stabilize()
        if self.verbose:
            print(f"Node {node.id} has left the network.\n")

    def stabilize(self):
        # updates finger tables of active nodes only
        active_nodes = [node for node in self.node_bank.values() if node.is_active]
        for node in active_nodes:
            self.assign_successors_and_predecessors(node, self.r)

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
