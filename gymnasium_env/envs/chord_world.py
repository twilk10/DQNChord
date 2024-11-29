import os
import sys
import gymnasium as gym 
import numpy as np
from typing import List, Dict, Optional
import random

# network = ChordNetwork(size=10, r=2, bank_size=20)
# network.display_network()


class ChordWorldEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self):
        super(ChordWorldEnv, self).__init__()

        self.register_env() # register the environment
        
        self.action_space = gym.spaces.Discrete(2)  # Actions 0 to 4

        self.observation_space = gym.spaces.Dict({
            'node_id': gym.spaces.Discrete(256),
            'finger_table': gym.spaces.Box(low=0, high=255, shape=(2,)),
            'successor': gym.spaces.Discrete(256),
            'predecessor': gym.spaces.Discrete(256),
            'lookup_success_rate': gym.spaces.Box(low=0.0, high=1.0, shape=()),
        })

        self.state = None
        self.reset()

    def register_env(self):
        gym.register(
            id='ChordNodeEnv-v0',
            entry_point='dqn:ChordNodeEnv',
        )

    def reset(self):
        # Initialize the agent's state and the network
        self.state = self._initialize_state()
        self.network_state = self._initialize_network()
        return self.state

    def step(self, action):
        self._update_environment()
        self._take_action(action)
        reward = self._compute_reward(action)
        done = self._check_done()
        return self.state, reward, done, {}

    def _initialize_state(self):
        # Initialize the agent's state

        # this is dummy code:
        # node_id = np.random.randint(0, 256)
        # return {
        #     'node_id': node_id,
        #     'finger_table': np.full(, -1),  
        #     'successor': (node_id + 1) % 256,  # Simplistic initial successor
        #     'predecessor': (node_id - 1) % 256,  # Simplistic initial predecessor
        #     'lookup_success_rate': 1.0,
        #     'finger_table_accuracy': 1.0,
        #     # Other state variables
        # }
        pass

    def _initialize_network(self):
        # init network from chord
        pass

    def _take_action(self, action):
        if action == 1:
            self._stabilize()
        elif action == 2:
            self._initiate_lookup()
        elif action == 3:
            pass

    def _stabilize(self):
        # stabalization from chord
        pass

    def _initiate_lookup(self):
        # Simulate a lookup operation
        # Determine if the lookup is successful based on the network state
        pass

    def _update_environment(self):
        # Simulate network dynamics
        # Nodes may join or leave, affecting the agent
        pass

    def _compute_reward(self, action):
        # Compute the reward based on the current state and action
        reward = 0
        # Calculate reward components
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

    def display_network(self):
        print(" Network state:")
        for node in self.node_bank.values():
            if node.is_active:
                print(node)


