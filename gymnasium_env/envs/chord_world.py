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
                 max_steps = 300, 
                 seed = 42, 
                 stabalize_interval = 5, 
                 fix_fingers_interval = 10, 
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




    def seed(self, seed = None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        random.seed(seed)
        np.random.seed(seed)
        return [seed]
    
    def _get_obs(self):
        ''' 
            Get the observation
        '''
        agent_node = self.network.node_bank[self.agent_id]

        # Local stability indicator: how correct is the agent’s immediate successor?
        local_stability = self._local_stability_indicator(agent_node)

        pred = agent_node.predecessor if agent_node.predecessor is not None else 0
        succ = agent_node.successor if agent_node.successor is not None else 0

        obs = np.array([pred, succ, local_stability], dtype=np.float32)
        return obs

    def _local_stability_indicator(self, node):
        """
        Improved heuristic:
        - +0.5 if agent’s successor’s predecessor is agent
        - +0.5 if agent’s predecessor’s successor is agent
        This gives a range [0.0, 1.0].
        """
        stability_score = 0.0

        # Check successor relationship
        if node.successor is not None:
            succ_node = self.network.node_bank[node.successor]
            if succ_node.predecessor == node.id:
                stability_score += 0.5

        # Check predecessor relationship
        if node.predecessor is not None:
            pred_node = self.network.node_bank[node.predecessor]
            if pred_node.successor == node.id:
                stability_score += 0.5

        return stability_score

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


    def _update_environment(self):
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
        for i in range(self.network.m * 3):
             active_nodes = [node for node in self.network.node_bank.values() if node.is_active]
             node = random.choice(active_nodes)
             if node.id == self.agent_id:
                    continue  # Agent handled separately
             if node.is_active:
                    # Non-agent nodes stabilize every stabilize_interval steps
                    if self.current_step % self.stabalize_interval == 0:
                        self.network.stabilize(node)
                    # Non-agent nodes fix_fingers every fix_fingers_interval steps
                    # if self.current_step % self.fix_fingers_interval == 0:
                    #     self.network.fix_fingers(node)

        for node_id, node in self.network.node_bank.items():
                if node_id == self.agent_id:
                    continue  # Agent handled separately
                if node.is_active:
                    # # Non-agent nodes stabilize every stabilize_interval steps
                    # if self.current_step % self.stabalize_interval == 0:
                    #     self.network.stabilize(node)
                    # Non-agent nodes fix_fingers every fix_fingers_interval steps
                    if self.current_step % self.fix_fingers_interval == 0:
                        self.network.fix_fingers(node)


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
                 k: int = 3,
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

        self.successor_list: List[int] = [successor] + [0] * (k - 1)    

    def set_active_status(self, new_status: bool):
        self.is_active = new_status

    def __str__(self):
        return (f"\t Node Id: {self.id}\n"
                f"\t Active Status: {self.is_active}\n"
                f"\t Predecessor: {self.predecessor}\n"
                f"\t Successor: {self.successor}\n"
                f"\t Finger Table: {self.finger_table}\n"
                f"\t Successor List: {self.successor_list}\n")

# ChordNetwork Class
class ChordNetwork:
    def __init__(self, m=4, nodes_to_activate: List[int] = [], verbose=False, seed=42):
        self.m = m
        self.k = 3
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
        # print('my sorted node ids are: ', sorted_node_ids)
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
                # print(f'start is: {start}')
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

        # set the successor list
        successor_list = [0] * self.k
        for idx, node_id in enumerate(active_nodes_ids):
            current_node = self.node_bank[node_id]
            print(f'f here is the node\n {current_node}')
            for i in range(self.k):
                next_successor_index = (idx + pow(2, i)) % total_number_of_active_nodes
                next_successor_node_id = active_nodes_ids[next_successor_index]
                if next_successor_node_id == current_node.id:
                    continue
                # print('current node id is: ', current_node)
                print(f'current node {current_node.id} successor list is: {current_node.successor_list}')
                
                successor_list[i] = next_successor_node_id 
            current_node.successor_list = successor_list
           
   

    def find_successor(self, node: Node, id: int) -> Node: # ask node to find successor of id
        n_prime = self.find_predecessor(node, id)
        # print(f'+++++ received predecessor of {id} as: {n_prime.id}')
      
        if n_prime is None:
            return None
        
        return self.node_bank[n_prime.finger_table['successor'][0]]

    # def find_predecessor(self, node: Node, id: int) -> Node: # ask node to find predecessor of id
    #     '''
    #     works by finding the immediate predecessor node
    #     of the desired identifier; the successor of that node must be the
    #     successor of the identifier
    #     '''
    #     n_prime = node
    
    #     if self._is_in_left_closed_right_open_interval(id, n_prime.id, n_prime.successor, verbose = False):
    #             if id == n_prime.id:
    #                 return self.node_bank[n_prime.predecessor]
    #             return n_prime
        

    #     interval = (n_prime.id, n_prime.finger_table['successor'][0])
    #     # print(f'INTERVAL[0]: {interval[0]}, INTERVAL[1]: {interval[1]}, n_prime_id: {n_prime.id}, n_prime_succ: {n_prime.finger_table['successor'][0]}')
    #     while not self._is_in_left_open_right_closed_interval(id, interval[0], interval[1]):
           
    #         # check if the id is in nprime's interval   
    #         interval = (n_prime.id, n_prime.finger_table['successor'][0])
    #         # print(f'INTERVAL[0]: {interval[0]}, INTERVAL[1]: {interval[1]}, n_prime_id: {n_prime.id}, n_prime_succ: {n_prime.finger_table['successor'][0]}')
    #         if self._is_in_closed_interval(id, interval[0], interval[1], verbose = False):
    #             if id == n_prime.id:
    #                 return self.node_bank[n_prime.predecessor]
    #             return n_prime
            
    #         returned_node = self.closest_preceding_node(n_prime, id)
    #         # print(f'n_prim: {n_prime}, id: {id}')
    #         n_prime = returned_node
    #         interval = (returned_node.id, returned_node.successor)
    #         #   print(f'returned_node: {returned_node},interval: {interval}')

    #     return n_prime
    
    def find_predecessor(self, node: Node, id: int) -> Node:
        try:
            # print(f'+++++ finding predecessor of {id}, using node: {node.id}')
            n_prime = node

            counter = 0
            while not self._is_in_left_open_right_closed_interval(id, n_prime.id, n_prime.successor, verbose=False):
                returned_node = self.closest_preceding_node(n_prime, id)
                if returned_node.id == n_prime.id:
                    # Prevent infinite loop if no closer node is found
                    break
                n_prime = returned_node

                # print('closest preceding node of ', id, '  is: ', n_prime)
                # print(f'node which initialized the search is: {node.id}, it has status: {node.is_active}')
                # self.display_network()
                # if self._is_in_left_closed_right_open_interval(id, n_prime.id, n_prime.successor, verbose=False):
                #     print(f'+++++ found (IN LOOP) predecessor of {id} as\n: {n_prime}')  


                #     if self.node_bank[id].is_active or id == n_prime.id:
                #         return self.node_bank[n_prime.predecessor]
                #     else:
                #         return n_prime
                counter += 1
                if counter > self.m + 1:
                    raise Exception('infinite loop')
                        
                    
            # print(f'+++++ found predecessor of {id} as\n: {n_prime}')  
            return self.node_bank[n_prime.id]
        except Exception as e:
            print(f"Error finding predecessor for id {id}: {str(e)}")
            return None

    def closest_preceding_node(self, node: Node, id: int) -> Node:
        for i in reversed(range(self.m)):
            x = node.finger_table['successor'][i]
            if self._is_in_open_interval(x, node.id, id, verbose=False) and self.node_bank[x].is_active:
                if self.node_bank[x].is_active:
                    return self.node_bank[x]
                else:
                     # If x is inactive, refer to the successor list
                    for backup_successor_id in node.successor_list:
                        if self._is_in_open_interval(backup_successor_id, node.id, id, verbose=False):
                            if self.node_bank[backup_successor_id].is_active:
                                # print(f'-------------SUCCESSOR LIST BACKUP FOR X: {backup_successor_id}')
                                return self.node_bank[backup_successor_id]
                    # self.display_network()
                    # print(f'id being searched for is: {id}, using Node: {node.id}')
                    # raise Exception('closest preceding node is not active')
                
        # print(f'+++++ no preceding active node found for {id}, returning node itself: {node.id}')
        return node  # If no preceding active node is found, return the node itself

    # def closest_preceding_node(self, node: Node, id: int) -> Node:# ask node to find closest preceding node of id
    #     # print('self.m is: ', self.m)
    #     for i in reversed(range(self.m)):
    #         # print('iiiiiiii is: ', i)
        
    #         x = node.finger_table['successor'][i]
    #         if self._is_in_open_interval(x, node.id, id,verbose = False):
    #             print(f'-------------INTERVAL IS: {node.id, id},,X IS: {x}')
    #             if self.node_bank[x].is_active:
    #                 return self.node_bank[x]
    #             else:
    #                 # max_closest_preceding_node_id = node.id
    #                 # for i in reversed(range(self.m)):
    #                 #     if(node.finger_table['successor'][i] == node.id) or (node.finger_table['successor'][i] == x):
    #                 #         continue    
    #                 #     # choose the largest of the successors whose value is less than id
    #                 #     if id > node.finger_table['successor'][i]:
    #                 #         max_closest_preceding_node_id = max(max_closest_preceding_node_id, node.finger_table['successor'][i])

    #                 # print(f'------------NEW ELSE-----------------------id being searched for is: {id}')
    #                 # print(f'------------NEW ELSE-----------------------max_closest_preceding_node_id is: {max_closest_preceding_node_id}') 
    #                 # return self.node_bank[max_closest_preceding_node_id]
                    
    #                 self.display_network()
    #                 # succ = self.find_successor(node, id)
    #                 raise Exception('here')
    #                 print(f'-------------------------GOT SUCCESSOR AS:\n {succ}')
    #                 return succ

    #     max_closest_preceding_node_id = node.id
    #     for i in reversed(range(self.m)):
            
    #         if(node.finger_table['successor'][i] == node.id):
    #             continue    
    #         # choose the largest of the successors whose value is less than id
    #         if id > node.finger_table['successor'][i]:
    #             max_closest_preceding_node_id = max(max_closest_preceding_node_id, node.finger_table['successor'][i])

    #     print(f'-----------------------------------max_closest_preceding_node_id is: {max_closest_preceding_node_id}') 
    #     return self.node_bank[max_closest_preceding_node_id]



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
        # print(f'LEFT: {left}, RIGHT: {right}')
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
    

    # def stabilize(self, node: Node):  # Updates predecessor and successor pointers of a node
    #     print(f'stabilizing node {node.id}...')
        
        
    #     try:
    #         # ask the node's successor for its predecessor
    #         successor_node = self.node_bank[node.successor_list[0]]
    #         if not successor_node.is_active:
    #             print(f'successor node {successor_node.id} is not active')
    #             for i in range(self.m):
    #                 successor_id = node.successor_list[i]
    #                 successor_node = self.node_bank[successor_id]
    #                 if successor_node.is_active:
    #                     break   
                
    #             successor_node.predecessor = node.id
    #             node.successor = successor_node.id
    #             node.finger_table['successor'][0] = successor_node.id   
    #         else:  
    #             print(f'successor node {successor_node.id} is  active')
    #             x = successor_node.predecessor
    #             # print(f'x is: {x}') 

    #             interval = (node.id, successor_node.id)
    #             print(f'interval is: {interval}')   
    #             if x is not None and self._is_in_open_interval(x, interval[0], interval[1]):
    #                 print('in if statement')
    #                 node.successor = x
    #                 node.finger_table['successor'][0] = x

    #         # print(f' {node.successor} notifies {node.id} that it is {node.id}\'s successor')
    #         self.notify(self.node_bank[node.successor], node)
            
    #     except Exception as e:
    #         print(f"Error stabilizing node {node.id}: {str(e)}")
            
    def stabilize(self, node: Node):
        # print(f'Stabilizing node {node.id}...')
        
        try:
            # Primary successor
            primary_successor_id = node.successor
            primary_successor_node = self.node_bank.get(primary_successor_id)
            
            if not primary_successor_node or not primary_successor_node.is_active:
                # print(f'Primary successor {primary_successor_id} is not active.')
                # Find the next active successor from the successor list
                for i in range(len(node.successor_list)):
                    backup_successor_id = node.successor_list[i]
                    backup_successor_node = self.node_bank.get(backup_successor_id)
                    if backup_successor_node and backup_successor_node.is_active:
                        node.successor = backup_successor_id
                        node.finger_table['successor'][0] = backup_successor_id
                        # print(f'Updated primary successor to backup successor {backup_successor_id}.')
                        self.notify(backup_successor_node, node)
                        break
                else:
                    print('No active successors found in the successor list.')
            else:
                # Ask the primary successor for its predecessor
                x = primary_successor_node.predecessor
                if x is not None and self._is_in_open_interval(x, node.id, primary_successor_id):
                    # Update primary successor to x
                    # print(f'Found closer successor {x} for node {node.id}. Updating successor.')
                    node.successor = x
                    node.finger_table['successor'][0] = x
                    self.notify(self.node_bank[x], node)
                
                # Update successor list
                self.update_successor_list(node)
                
        except Exception as e:
            print(f"Error stabilizing node {node.id}: {str(e)}")
            # if node.is_active:  
            #     print(f'Manually stabilizing node {node.id}...')
            #     # self.manually_assign_successors_and_predecessors(node)
            # else:
            #     print('Node is not active, skipping stabilization.')
                    


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

        try:
            self.update_successor_list(node)
        except Exception as e:
            print(f"Error updating successor list for node {node.id}: {str(e)}")
            # self.manually_assign_successors_and_predecessors(node)

    def update_successor_list(self, node: Node):
        active_node_ids = sorted([n.id for n in self.node_bank.values() if n.is_active])
        if node.id in active_node_ids:
            index = active_node_ids.index(node.id)

            for i in range(1, self.k + 1):
                successor_id = active_node_ids[(index + i) % len(active_node_ids)]
                node.successor_list[i - 1] = successor_id
                print(f"Node {node.id} successor list updated to: {node.successor_list}")
        else:
            # Node is inactive; clear its successor list
            node.successor_list = [0] * self.k
            print(f"Node {node.id} is inactive. Successor list has been cleared.")

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
        # i = random.randint(1, len(finger_table['start']) - 1)
        for i in range(self.m):
            start = finger_table['start'][i]

            successor = self.find_successor(node, start)
            if successor is None:
                self.manually_fix_fingers(node)
                break
            # print(f'successor: {successor}')
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
        # print(f'identifier_successor_id_map is: {identifier_successor_id_map}')
        for i in range(self.m):
            start = finger_table['start'][i]
            finger_table['successor'][i] = identifier_successor_id_map[start]

        
    def join_network(self, node: Node): # find the immediate successor of node
        # print(f'Joining node {node.id} to the network...')
        try:
            if node.id not in self.node_bank:
                self.node_bank[node.id] = node
            else:
                node = self.node_bank[node.id]
            
            node.predecessor = None
            
            # Get all other active nodes excluding the joining node
            active_nodes = [n for n in self.node_bank.values() if n.is_active and n.id != node.id]
            # choose a node to use to help init the joining node's finger table

            if not active_nodes:
                # If no other active nodes, node points to itself
                node.successor = node.id
                node.finger_table['successor'][0] = node.id
                node.predecessor = node.id
                node.successor_list = [node.id] * self.k
                # print(f"Node {node.id} is the only active node in the network.")
                return
            
            n_prime = random.choice(active_nodes)
            # print(f'n_prime that is chosen is: {n_prime.id}')  

            node.set_active_status(True)

            self.init_finger_table(node, n_prime) # n_prime is the node that is helping to init the finger table of the joining node
            
            # Populate the successor list for the joining node
            self.populate_successor_list(node)
            
            # Update predecessor and successor nodes' successor lists
            successor_node = self.node_bank[node.successor]
            self.notify(successor_node, node)
            
        except Exception as e:
            print(f"Error joining node {node.id} to network: {str(e)}")
            node.set_active_status(False)
    
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
        # joining_node.finger_table['successor'][0] = joining_node_successor.id   
        joining_node.predecessor = joining_node_successor.predecessor

        finger_table = joining_node.finger_table
        for i in range(1,self.m):
            successor = self.find_successor(n_prime, finger_table['start'][i])
            finger_table['successor'][i] = successor.id    

        # Populate the successor list
        self.populate_successor_list(joining_node)

    def populate_successor_list(self, node: Node):
        sorted_node_ids = sorted([n.id for n in self.node_bank.values() if n.is_active])
        index = sorted_node_ids.index(node.id)
        # Populate the successor list with the next k active nodes
        for i in range(1, self.k + 1):
            successor_id = sorted_node_ids[(index + i) % len(sorted_node_ids)]
            node.successor_list[i - 1] = successor_id

    def leave_network(self, node: Node):
        node.set_active_status(False)
        
        # Clear the node's information
        node.predecessor = None
        node.successor = None
        node.finger_table = {}
        
        # print(f"Node {node.id} has left the network, it status is now: {node.is_active}.\n")
     

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
    # network.display_network()
        
        
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


