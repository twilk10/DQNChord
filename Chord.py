import math
from threading import Thread, Lock
import time
from typing import List, Dict, Optional
from Node import Node
import random
from enum import Enum


class ChurnRateTimer(Enum):
    LOW = 20
    MEDIUM = 50
    HIGH = 100

class ChordNetwork:
    def __init__(self,size, r, bank_size):
        self.r = r # number of successor nodes a node can have
        self.node_bank: Dict[int, Node] = self.initialize_node_bank(bank_size)
        self.initialize_graph(size, r)
        print('Initialization Done!')
        
    
    def initialize_node_bank(self, bank_size):
        print('Initializing node bank...')
        bank = {}
        for i in range(bank_size):
            if i not in bank:
                node = Node(id=i, active_status=False)
                bank[node.id] = node
        return bank
    
    def initialize_graph(self, size, r):
        print('initializing graph...')
        # Activate the first 'size' nodes from the node bank
        for i in range(size):
            node = self.node_bank[i]
            node.is_active = True

        for i in range(size):
            print('assigning predecessors and successors for node', node.id)
            node = self.node_bank[i]
            self.assign_successors_and_predecessors(node, r, initial_run= True)
            print(f'finger table for node{node.id:} \n {node.finger_table} ')

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
        active_nodes = sorted([n for n in self.node_bank.values() if n.is_active])
        for i in range(x):
            node_to_drop = random.choice(active_nodes)
            self.leave_network(node_to_drop)

    def lookup(self, key: int):
        print(f"Starting lookup for Node ID {key} at Node {0}")
        node_0_finger_table = self.node_bank[0].finger_table
        print('node 1 finger table is:', node_0_finger_table)
        if key in node_0_finger_table['successors']:
            return node_0_finger_table['successors'][key]

        network_size = len([n for n in self.node_bank.values() if n.is_active])
        max_hops =  2 * math.log10(network_size)
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
            closest_preceding_node = node_0_finger_table['successors'][0]
            for successor in node_0_finger_table['successors']:
                    if successor > key:
                        break
                    if abs(successor - key) < abs(closest_preceding_node - key):
                        closest_preceding_node = successor
            return self._lookup_helper(key, closest_preceding_node, max_hops - 1)
            closest_successor = min(finger_table['successors'], key=lambda x: abs(x - key))
            return self._lookup_helper(key, closest_successor, max_hops - 1) 
        return None

    def join_network(self, node: Node):
        node.set_active_status(True)
        node.reset_timer()
        self.assign_successors_and_predecessors(node, self.r)
        self.stabilize()
        print(f"Node {node.id} has joined the network.\n")

    def leave_network(self, node: Node): 
        node.set_active_status(False)
        node.reset_timer()
        # Clear the node's finger table
        node.finger_table = {'predecessors': [], 'successors': []}
            
        self.stabilize()
        print(f"Node {node.id} has left the network.\n")

    def stabilize_network(self):
        # updates finger tables of active nodes only
        active_nodes = [node for node in self.node_bank.values() if node.is_active]
        for node in active_nodes:
            self.assign_successors_and_predecessors(node, self.r)

    def display_network(self):
        print(" Network state:")
        for node in self.node_bank.values():
            if node.is_active:
                print(node)
    
    def random_churn_rate(self):
        rand = random.random()
        if rand < 0.333:
            return ChurnRateTimer.LOW.value
        elif rand < 0.666:
            return ChurnRateTimer.MEDIUM.value
        return ChurnRateTimer.HIGH.value



