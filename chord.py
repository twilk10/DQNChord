from threading import Thread
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
    def __init__(self, size, r, bank_size):
        self.size = size
        self.r = r # number of successor nodes a node can have
        self.node_bank: Dict[int, Node] = self.initialize_node_bank(bank_size)
        self.graph: List[Node] = self.initialize_graph(size, r)
        
        self.timer_thread = Thread(target=self.update_timers)
        self.timer_thread.start()
    
    def initialize_node_bank(self, bank_size):
        print('Initializing node bank...')
        bank = {}
        for i in range(bank_size):
            if i not in bank:
                timer = self.random_churn_rate()
                node = Node(id=i, ttl=timer, active_status=False)
                bank[node.id] = node
        return bank

    def initialize_graph(self, size, r):
        print('Initializing graph...')
        graph = []
        for i in range(size):
            node = self.node_bank[i]
            node.set_active_status(True)
            self.assign_initial_successors_and_predecessors(node, size, r)
            graph.append(node)
        return graph 
      
    def assign_initial_successors_and_predecessors(self, node: Node, size, r):
        # Successor assignment
        for j in range(1, r+1):
            successor_id = (node.id + j) % size
            node.finger_table['successors'].append(successor_id)

        # Predecessor assignment
        predecessor_id = (node.id - 1) % size
        node.finger_table['predecessors'].append(predecessor_id)

    def update_timers(self):
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
            time.sleep(1)

    def join_network(self, node:Node):
        if node in self.graph:
            print(f"Node {node.id} is already in the network. Cannot add to network")
            return
        
        node.set_active_status(True)
        node.reset_timer()
        self.add_node(node)

        self.update_all_finger_tables()
        print(f"Node {node.id} has joined the network. \n")

    def leave_network(self, node: Node):
        if node not in self.graph:
            print(f"\t Node {node.id} is not in the network. Cannot remove from network")
            return
        
        # Remove node and reassign successors/predecessors
        node.set_active_status(False)
        node.reset_timer()
        self.remove_node(node)
        
        self.update_all_finger_tables()
        print(f"Node {node.id} has left the network. \n")

    def update_all_finger_tables(self):
        print('updating finger tables ...')
        for idx, node in enumerate(self.graph):
            # update predecessor
            predecessor_node = self.graph[idx - 1] # we can use negative numbers to wrap around an array in python
            node.finger_table['predecessors']= [predecessor_node.id]

            # update successors
            # print('idx is:', idx)
            new_successors_list = []
            for i in range(idx + 1, (idx + self.r + 1)):
                modulo_idx = i % len(self.graph) # modulo index to ensure wrap around
                # print(f'\t i is :{i},  modulo_idx is { modulo_idx}')
                new_successors_list.append(self.graph[modulo_idx].id)
            node.finger_table['successors']= new_successors_list

        # Display network to check that all tables were updated accordingly
        graph_list = []
        for node in self.graph:
            graph_list.append(str(node.id))
        print('After updating finger table, graph list is:', '->'.join(graph_list))
        # self.display_network()

    def display_network(self):
        print(" Network state:")
        for node in self.graph:
            print(node)

    def add_node(self, new_node):
        # find position to add node
        # Implemented with linear search. Might want to change to binary in future
        idx = 0 
        while idx < len(self.graph) and self.graph[idx].id < new_node.id:
            idx+= 1 

        self.graph.insert(idx, new_node)

    def remove_node(self, node):
        self.graph = [n for n in self.graph if n.id != node.id]

    def random_churn_rate(self):
        rand = random.random()
        if rand < 0.333:
            return ChurnRateTimer.LOW.value
        elif rand < 0.666:
            return ChurnRateTimer.MEDIUM.value
        return ChurnRateTimer.HIGH.value
        



def test_chord_protocol():
    # Create a Chord network with 5 nodes and 2 successors for each node
    # Node bank size of 10 nodes
    network = ChordNetwork(size=5, r=2, bank_size=10)
    
    # Display initial state of the network
    print("Initial Network State:")
    network.display_network()

test_chord_protocol()
