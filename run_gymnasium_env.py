import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import torch.nn.functional as F
import matplotlib.pyplot as plt
import gc
import gymnasium_env


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

gc.collect()
torch.cuda.empty_cache()

seed = 1234 
np.random.seed(seed)
np.random.default_rng(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.states = deque(maxlen=capacity)
        self.actions = deque(maxlen=capacity)
        self.next_states = deque(maxlen=capacity)
        self.rewards = deque(maxlen=capacity)
        self.dones = deque(maxlen=capacity)
        
    def store(self, state, action, next_state, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.next_states.append(next_state)
        self.rewards.append(reward)
        self.dones.append(done)
        
    def sample(self, batch_size):
        indices = np.random.choice(len(self), size=batch_size, replace=False)

        states = torch.stack([torch.as_tensor(self.states[i], dtype=torch.float32, device=device) for i in indices]).to(device)
        actions = torch.as_tensor([self.actions[i] for i in indices], dtype=torch.long, device=device)
        next_states = torch.stack([torch.as_tensor(self.next_states[i], dtype=torch.float32, device=device) for i in indices]).to(device)
        rewards = torch.as_tensor([self.rewards[i] for i in indices], dtype=torch.float32, device=device)
        dones = torch.as_tensor([self.dones[i] for i in indices], dtype=torch.bool, device=device)

        return states, actions, next_states, rewards, dones
    
    def __len__(self):
        return len(self.dones)
    
class DQN_Network(nn.Module):
    def __init__(self, num_actions, input_dim):
        super(DQN_Network, self).__init__()

        # self.FC = nn.Sequential(
        #         nn.Linear(input_dim, 128),
        #         nn.ReLU(inplace=True),
        #         nn.Linear(128, 64),
        #         nn.ReLU(inplace=True),
        #         nn.Linear(64, num_actions)
        #     )

        self.FC = nn.Sequential(
                nn.Linear(input_dim, 12),
                nn.ReLU(inplace=True),
                nn.Linear(12, 8),
                nn.ReLU(inplace=True),
                nn.Linear(8, num_actions)
            )
        
        for layer in [self.FC]:
            for module in layer:
                if isinstance(module, nn.Linear):
                    nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
        
    def forward(self, x):
        Q = self.FC(x)    
        return Q

class DQN_Agent:
    def __init__(self, action_space, observation_space, epsilon_max, epsilon_min,
                 epsilon_decay, clip_grad_norm, learning_rate, discount, memory_capacity):
        
        self.loss_history = []
        self.running_loss = 0
        self.learned_counts = 0
                     
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.discount = discount

        self.action_space = action_space
        self.action_space.seed(seed) 

        input_dim = observation_space.shape[0]
        self.main_network = DQN_Network(num_actions=self.action_space.n, input_dim=input_dim).to(device)
        self.target_network = DQN_Network(num_actions=self.action_space.n, input_dim=input_dim).to(device).eval()

        self.replay_memory = ReplayMemory(memory_capacity)
        
        self.target_network.load_state_dict(self.main_network.state_dict())

        self.clip_grad_norm = clip_grad_norm 
        self.critertion = nn.MSELoss()
        # adamn optmization function
        self.optimizer = optim.Adam(self.main_network.parameters(), lr=learning_rate)
                
    def select_action(self, state):
        if np.random.random() < self.epsilon_max:
            return self.action_space.sample()
        
        with torch.no_grad():
            Q_values = self.main_network(state)
            action = torch.argmax(Q_values).item()
                        
            return action
   
    def learn(self, batch_size, done):
        states, actions, next_states, rewards, dones = self.replay_memory.sample(batch_size)
                    
        actions = actions.unsqueeze(1)
        rewards = rewards.unsqueeze(1)
        dones = dones.unsqueeze(1)       
        
        predicted_q = self.main_network(states) 
        predicted_q = predicted_q.gather(dim=1, index=actions) 

        with torch.no_grad():            
            next_target_q_value = self.target_network(next_states).max(dim=1, keepdim=True)[0] 
            
        next_target_q_value[dones] = 0 
        y_js = rewards + (self.discount * next_target_q_value) 
        loss = self.critertion(predicted_q, y_js) 
        
        self.running_loss += loss.item()
        self.learned_counts += 1

        if done:
            episode_loss = self.running_loss / self.learned_counts 
            self.loss_history.append(episode_loss)
            self.running_loss = 0
            self.learned_counts = 0
            
        self.optimizer.zero_grad()
        loss.backward() 
        torch.nn.utils.clip_grad_norm_(self.main_network.parameters(), self.clip_grad_norm)
        self.optimizer.step() 
 
    def hard_update(self):
        self.target_network.load_state_dict(self.main_network.state_dict())
    
    def update_epsilon(self):
        self.epsilon_max = max(self.epsilon_min, self.epsilon_max * self.epsilon_decay)

    def save(self, path):
        torch.save(self.main_network.state_dict(), path)

class Model_TrainTest:
    def __init__(self, hyperparams):
        self.clip_grad_norm = hyperparams["clip_grad_norm"]
        self.learning_rate = hyperparams["learning_rate"]
        self.discount_factor = hyperparams["discount_factor"]
        self.batch_size = hyperparams["batch_size"]
        self.update_frequency = hyperparams["update_frequency"]
        self.max_episodes = hyperparams["max_episodes"]
        self.max_steps = hyperparams["max_steps"]
        
        self.epsilon_max = hyperparams["epsilon_max"]
        self.epsilon_min = hyperparams["epsilon_min"]
        self.epsilon_decay = hyperparams["epsilon_decay"]
        
        self.memory_capacity = hyperparams["memory_capacity"]
        
        self.action_space = hyperparams["action_space"]
        self.observation_space = hyperparams["observation_space"]
                        
        self.agent = DQN_Agent(
                                action_space = self.action_space,
                                observation_space = self.observation_space,
                                epsilon_max = self.epsilon_max, 
                                epsilon_min = self.epsilon_min, 
                                epsilon_decay = self.epsilon_decay,
                                clip_grad_norm = self.clip_grad_norm,
                                learning_rate = self.learning_rate,
                                discount = self.discount_factor,
                                memory_capacity = self.memory_capacity
                            )
        
    def plot_training(self):
        sma = np.convolve(self.reward_history, np.ones(50)/50, mode='valid')
        
        #reward plot
        plt.figure()
        plt.title("Rewards")
        plt.plot(self.reward_history, label='Raw Reward', color='#F6CE3B', alpha=1)
        plt.plot(sma, label='SMA 50', color='#385DAA')
        plt.xlabel("Episode")
        plt.ylabel("Rewards")
        plt.legend()
        
        plt.tight_layout()
        plt.grid(True)
        plt.show()
        plt.clf()
        plt.close() 
        
        # loss plot
        plt.figure()
        plt.title("Loss")
        plt.plot(self.agent.loss_history, label='Loss', color='#CB291A', alpha=1)
        plt.xlabel("Episode")
        plt.ylabel("Loss")
        
        plt.tight_layout()
        plt.grid(True)
        plt.show()        
    

def train(agent, env): 
    total_steps = 0
    agent.reward_history = []
    
    for episode in range(1, agent.max_episodes+1):
        state, _ = env.reset(seed=seed)
        state = torch.FloatTensor(state).to(device)
        done = False
        truncation = False
        step_size = 0
        episode_reward = 0
                                            
        while not done and not truncation:
            action = agent.agent.select_action(state)
            next_state, reward, done, truncation, _ = env.step(action)
            next_state = torch.FloatTensor(next_state).to(device)
            
            agent.agent.replay_memory.store(state, action, next_state, reward, done) 
            
            if len(agent.agent.replay_memory) > agent.batch_size and sum(agent.reward_history) > 0:
                agent.agent.learn(agent.batch_size, (done or truncation))
            
                if total_steps % agent.update_frequency == 0:
                    agent.agent.hard_update()
            
            state = next_state
            episode_reward += reward
            step_size +=1
                        
        agent.reward_history.append(episode_reward) 
        total_steps += step_size
                                                                        
        agent.agent.update_epsilon()
                        
        result = (f"Episode: {episode}, "
                    f"Total Steps: {total_steps}, "
                    f"Ep Step: {step_size}, "
                    f"Raw Reward: {episode_reward:.2f}, "
                    f"Epsilon: {agent.agent.epsilon_max:.2f}")
        print(result)
    agent.plot_training()

def test(agent, env, max_episodes, path):  
    from gymnasium.spaces.utils import unflatten

    agent.agent.main_network.load_state_dict(torch.load(path))
    agent.agent.main_network.eval()
    
    success_count = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    original_obs_space = env.unwrapped.original_observation_space  # Access the original observation space from the unwrapped environment

    for episode in range(1, max_episodes + 1):         
        state, _ = env.reset(seed=seed)
        state_tensor = torch.FloatTensor(state).to(device)
        done = False
        truncation = False
        step_size = 0
        episode_reward = 0
                                                                
        while not done and not truncation:
            action = agent.agent.select_action(state_tensor)
            next_state, reward, done, truncation, _ = env.step(action)
            next_state_tensor = torch.FloatTensor(next_state).to(device)
    
            # Update state for the next iteration
            state_tensor = next_state_tensor
            episode_reward += reward
            step_size += 1

        # After the episode ends, access lookup_success_rate
        state_np = state_tensor.cpu().numpy()
        obs_dict = unflatten(original_obs_space, state_np)
        lookup_success_rate = obs_dict['lookup_success_rate'][0]

        if lookup_success_rate >= 0.5:
            succeed = True
            success_count += 1
        else:
            succeed = False

        # Print episode result
        result = (f"Episode: {episode}, "
                  f"Steps: {step_size}, "
                  f"Reward: {episode_reward:.2f}, "
                  f"Lookup Success Rate: {lookup_success_rate:.2f}, "
                  f"Succeed: {succeed}")
        print(result)
    
    # Calculate and print overall success rate
    success_rate = success_count / max_episodes * 100
    print(f"\nOverall Success Rate: {success_rate:.2f}%")


def main():
    print(device)
    env = gym.make('gymnasium_env/ChordWorldEnv-v0')
    # env_20 = gym.make('FrozenLake-v1', desc=custom_map_20, is_slippery=False, max_episode_steps=100)

    RL_hyperparams = {
        "clip_grad_norm": 3,
        "learning_rate": 6e-4,
        "discount_factor": 0.93,
        "batch_size": 32,
        "update_frequency": 10,
        "max_episodes": 3000,
        "max_steps": 200,
        "epsilon_max": 0.999, 
        "epsilon_min": 0.01,
        "epsilon_decay": 0.999,
        "memory_capacity": 4_000,
        "action_space": env.action_space,
        "observation_space": env.observation_space,
    }

    DRL_Chord = Model_TrainTest(RL_hyperparams)
    train(DRL_Chord, env)
    DRL_Chord.agent.save("Chord_model.pt")

    test(DRL_Chord, env, max_episodes = 200, path = "Chord_model.pt")

if __name__ == '__main__':
    main()