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
    def __init__(self,size, r, bank_size, start_timer_thread = True):
        self.r = r # number of successor nodes a node can have
        self.node_bank: Dict[int, Node] = self.initialize_node_bank(bank_size)
        self.initialize_graph(size, r)

        self.lock = Lock()
        self.timer_thread = None
        if start_timer_thread:
            print('in this if statement')
            self.timer_thread = Thread(target=self.update_node_timers)
            self.timer_thread.daemon = True
            self.timer_thread.start()
        print('done')
        
    
    def initialize_node_bank(self, bank_size):
        print('Initializing node bank...')
        bank = {}
        for i in range(bank_size):
            if i not in bank:
                timer = self.random_churn_rate()
                status = False

                # set timer to a large number for Agent Node 0 and status to True
                if i == 0:
                    timer = float('inf')
                    status = True

                node = Node(id=i, ttl=timer, active_status=status)
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
        
        # # Ensure the special node (Node 0) is set up
        # self.assign_successors_and_predecessors(self.node_bank[0], r)
        # print(f'finger table for node{self.node_bank[0].id:} \n {self.node_bank[0].finger_table} ')

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

    def update_node_timers(self):
        while True:
            for node in self.node_bank.values():
                node.ttl -= 1
                if node.ttl <= 0:
                    if node.is_active:
                        print(f"Node {node.id} is attempting to leave the network...")
                        self.leave_network(node)
                    else:
                        print(f"Node {node.id} is attempting to join the network...")
                        self.join_network(node)

            # Have Node 0 act
            # agent_node = self.node_bank[0]
            # agent_node.act(self)
            time.sleep(1)

    def join_network(self, node: Node):
        with self.lock:
            node.set_active_status(True)
            node.reset_timer()
            self.assign_successors_and_predecessors(node, self.r)

            self.update_all_finger_tables()
            print(f"Node {node.id} has joined the network.\n")
            # visualize_chord_network(self) 

    def leave_network(self, node: Node): 
        with self.lock:
            
            node.set_active_status(False)
            node.reset_timer()

            # Clear the node's finger table
            node.finger_table = {'predecessors': [], 'successors': []}
            
            self.update_all_finger_tables()
            print(f"Node {node.id} has left the network.\n")

    def update_all_finger_tables(self):
        # updates finger tables of active nodes only
        active_nodes = [node for node in self.node_bank.values() if node.is_active]
        for node in active_nodes:
            self.assign_successors_and_predecessors(node, self.r)


    def display_network(self):
        print(" Network state:")
        for node in self.node_bank.values():
            if node.is_active:
                print(node)

    def get_visualization_data(self):
        with self.lock:
            nodes = []
            edges = []
            # active_nodes = [node for node in self.node_bank.values() if node.is_active]
            active_nodes = sorted(
                [node for node in self.node_bank.values() if node.is_active],
                key=lambda node: node.id
            )

            # Dynamically scale the radius based on the number of nodes
            base_radius = 300  # Base radius for a small number of nodes
            scaling_factor = 20  # Increase radius by 20 pixels per additional node
            radius = base_radius + scaling_factor * max(0, len(active_nodes) - 10)

            for i, node in enumerate(active_nodes):
                angle = (2 * math.pi / len(active_nodes)) * i
                x = radius * math.cos(angle)
                y = radius * math.sin(angle)
                node_data = {
                        'id': node.id,
                        'label': str(node.id),
                        'title': f"Finger Table:\n{node.finger_table}",
                        'color': 'red' if node.is_agent else 'green',
                        'x': x,  # Set the x-coordinate
                        'y': y   # Set the y-coordinate
                    }
                # Highlight agent node differently
                if node.is_agent:
                    node_data['color'] = 'red'
                else:
                     node_data['color'] = 'green'
                nodes.append(node_data)

                # Add edges to successors
                # print('assigning edges for node:', node.id)
                # print(f'fingertable for node{node.id}: \n {node.finger_table}')
                curr_edges = []
                for successor_id in node.finger_table['successors']:
                    if successor_id in self.node_bank and self.node_bank[successor_id].is_active:
                        curr_edges.append({
                            'from': node.id,
                            'to': successor_id,
                            'arrows': 'to'
                        })
                        edges.append({
                            'from': node.id,
                            'to': successor_id,
                            'arrows': 'to'
                        })
                # print('done gathering edges:', curr_edges)
            return nodes, edges

    def random_churn_rate(self):
        rand = random.random()
        if rand < 0.333:
            return ChurnRateTimer.LOW.value
        elif rand < 0.666:
            return ChurnRateTimer.MEDIUM.value
        return ChurnRateTimer.HIGH.value



