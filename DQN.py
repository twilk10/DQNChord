import os
import sys
import gymnasium as gym 
import numpy as np
from Chord import ChordNetwork
   # Add the parent directory to sys.path
   # 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
network = ChordNetwork(size=10, r=2, bank_size=20)
network.display_network()


class ChordWorldEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self):
        super(ChordWorldEnv, self).__init__()

        self.register_env() # register the environment
        
        self.action_space = gym.spaces.Discrete(2)  # Actions 0 to 4

        self.observation_space = gym.spaces.Dict({
            'node_id': gym.spaces.Discrete(256),
            'finger_table': gym.spaces.Box(low=0, high=255, shape=(m,)),
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
