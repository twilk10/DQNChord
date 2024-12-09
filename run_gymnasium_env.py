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
from gymnasium.spaces.utils import unflatten

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Garbage collection and CUDA memory cleanup
gc.collect()
torch.cuda.empty_cache()

# Set seeds for reproducibility
seed = 1234
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
        
    def store(self, state, action, next_state, reward, done):
        self.memory.append((state, action, next_state, reward, done))
        
    def sample(self, batch_size):
        batch = np.random.choice(len(self), size=batch_size, replace=False)
        states, actions, next_states, rewards, dones = zip(*[self.memory[i] for i in batch])

        states = torch.stack(states).to(device)
        actions = torch.tensor(actions, dtype=torch.long, device=device).unsqueeze(1)
        next_states = torch.stack(next_states).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1)
        dones = torch.tensor(dones, dtype=torch.bool, device=device).unsqueeze(1)

        return states, actions, next_states, rewards, dones
    
    def __len__(self):
        return len(self.memory)
    
class DQN_Network(nn.Module):
    def __init__(self, num_actions, input_dim):
        super(DQN_Network, self).__init__()

        self.FC = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_actions)
        )
        
        # Initialize weights
        for module in self.FC:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
        
    def forward(self, x):
        Q = self.FC(x)
        return Q

class DQN_Agent:
    def __init__(self, action_space, observation_space, epsilon_max, epsilon_min,
                 decay_episodes, clip_grad_norm, learning_rate, discount, memory_capacity):
        
        self.loss_history = []
        self.step_loss_history = []
        self.running_loss = 0
        self.learned_counts = 0
                     
        self.epsilon = epsilon_max
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.decay_episodes = decay_episodes
        self.epsilon_decay = (epsilon_max - epsilon_min) / decay_episodes
        self.discount = discount

        self.action_space = action_space
        self.action_space.seed(seed)

        input_dim = observation_space.shape[0]
        self.main_network = DQN_Network(num_actions=self.action_space.n, input_dim=input_dim).to(device)
        self.target_network = DQN_Network(num_actions=self.action_space.n, input_dim=input_dim).to(device)
        self.target_network.load_state_dict(self.main_network.state_dict())
        self.target_network.eval()

        self.replay_memory = ReplayMemory(memory_capacity)
        
        self.clip_grad_norm = clip_grad_norm
        self.criterion = nn.SmoothL1Loss()
        self.optimizer = optim.Adam(self.main_network.parameters(), lr=learning_rate)
                
    def select_action(self, state):
        if np.random.random() < self.epsilon:
            return self.action_space.sample()
        else:
            with torch.no_grad():
                Q_values = self.main_network(state)
                action = torch.argmax(Q_values).item()
                return action
   
    def learn(self, batch_size):
        states, actions, next_states, rewards, dones = self.replay_memory.sample(batch_size)
                    
        predicted_q = self.main_network(states).gather(1, actions)
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0].unsqueeze(1)
            next_q_values[dones] = 0.0
            target_q = rewards + self.discount * next_q_values

        loss = self.criterion(predicted_q, target_q)
        
        self.running_loss += loss.item()
        self.learned_counts += 1

        self.step_loss_history.append(loss.item())

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.main_network.parameters(), self.clip_grad_norm)
        self.optimizer.step()
 
    def hard_update(self):
        self.target_network.load_state_dict(self.main_network.state_dict())
    
    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay)

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
        self.decay_episodes = hyperparams["decay_episodes"]
        
        self.memory_capacity = hyperparams["memory_capacity"]
        
        self.action_space = hyperparams["action_space"]
        self.observation_space = hyperparams["observation_space"]
                        
        self.agent = DQN_Agent(
                                action_space = self.action_space,
                                observation_space = self.observation_space,
                                epsilon_max = self.epsilon_max, 
                                epsilon_min = self.epsilon_min, 
                                decay_episodes = self.decay_episodes,
                                clip_grad_norm = self.clip_grad_norm,
                                learning_rate = self.learning_rate,
                                discount = self.discount_factor,
                                memory_capacity = self.memory_capacity
                            )
        self.reward_history = []
        self.loss_history = []

    def plot_training(self):
        sma_rewards = np.convolve(self.reward_history, np.ones(50)/50, mode='valid')
        
        # Reward plot
        plt.figure()
        plt.title("Rewards")
        plt.plot(self.reward_history, label='Raw Reward', color='#F6CE3B', alpha=0.6)
        plt.plot(range(len(sma_rewards)), sma_rewards, label='SMA 50', color='#385DAA')
        plt.xlabel("Episode")
        plt.ylabel("Rewards")
        plt.legend()
        plt.grid(True)
        plt.show()
        
        # Loss plot
        sma_loss = np.convolve(self.loss_history, np.ones(50)/50, mode='valid')
        plt.figure()
        plt.title("Loss")
        plt.plot(self.loss_history, label='Raw Loss', color='#CB291A', alpha=0.6)
        plt.plot(range(len(sma_loss)), sma_loss, label='SMA 50', color='#2E8B57')
        plt.xlabel("Episode")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.show()

def train(agent, env): 
    total_steps = 0
    agent.reward_history = []
    agent.loss_history = []
    
    for episode in range(1, agent.max_episodes + 1):
        state, _ = env.reset(seed=seed)
        state = torch.FloatTensor(state).to(device)
        done = False
        truncation = False
        step_size = 0
        episode_reward = 0
                                            
        while not done and not truncation:
            # Normalize state
            state_norm = (state - state.mean()) / (state.std() + 1e-8)
            action = agent.agent.select_action(state_norm)
            next_state, reward, done, truncation, _ = env.step(action)
            next_state = torch.FloatTensor(next_state).to(device)
            next_state_norm = (next_state - next_state.mean()) / (next_state.std() + 1e-8)
            
            agent.agent.replay_memory.store(state_norm, action, next_state_norm, reward, done)
            
            if len(agent.agent.replay_memory) > agent.batch_size:
                agent.agent.learn(agent.batch_size)
                if total_steps % agent.update_frequency == 0:
                    agent.agent.hard_update()
            
            state = next_state
            episode_reward += reward
            step_size += 1
            total_steps += 1
                                
        agent.reward_history.append(episode_reward)
        if agent.agent.learned_counts > 0:
            avg_loss = agent.agent.running_loss / agent.agent.learned_counts
            agent.loss_history.append(avg_loss)
            agent.agent.running_loss = 0
            agent.agent.learned_counts = 0
        else:
            agent.loss_history.append(0)
        
        agent.agent.update_epsilon()
                        
        result = (f"Episode: {episode}, "
                  f"Total Steps: {total_steps}, "
                  f"Episode Steps: {step_size}, "
                  f"Episode Reward: {episode_reward:.2f}, "
                  f"Epsilon: {agent.agent.epsilon:.4f}")
        print(result)
    agent.plot_training()

    plt.figure()
    plt.title("Training Step Loss")
    plt.plot(agent.agent.step_loss_history, label='Per-step Loss', color='blue', alpha=0.6)
    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

def test(agent, env, max_episodes, path):
    agent.agent.main_network.load_state_dict(torch.load(path))
    agent.agent.main_network.eval()

    stability_success_count = 0
    stability_threshold = 0.8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for episode in range(1, max_episodes + 1):
        state, _ = env.reset(seed=seed)
        state_tensor = torch.FloatTensor(state).to(device)

        done = False
        truncation = False
        step_size = 0
        episode_reward = 0
        episode_stabilities = []

        while not done and not truncation:
            state_norm = (state_tensor - state_tensor.mean()) / (state_tensor.std() + 1e-8)
            action = agent.agent.select_action(state_norm)
            next_state, reward, done, truncation, _ = env.step(action)
            next_state_tensor = torch.FloatTensor(next_state).to(device)

            local_stability = next_state[2]
            episode_stabilities.append(local_stability)

            state_tensor = next_state_tensor
            episode_reward += reward
            step_size += 1

        # Compute average stability for the agent
        average_stability = sum(episode_stabilities) / len(episode_stabilities) if episode_stabilities else 0.0
        is_success = average_stability >= stability_threshold
        if is_success:
            stability_success_count += 1

        # Compute all nodes' stabilities
        all_stabilities = {}
        for node_id, node in env.network.node_bank.items():
            if node.is_active:
                node_stability = env._local_stability_indicator(node)
                all_stabilities[node_id] = node_stability

        average_stability_all = (sum(all_stabilities.values()) / len(all_stabilities)) if all_stabilities else 0.0

        # Print episode result with additional metrics
        result = (
            f"Episode: {episode}, "
            f"Steps: {step_size}, "
            f"Reward: {episode_reward:.2f}, "
            f"Agent Avg Stability: {average_stability:.2f}, "
            f"All Nodes Avg Stability: {average_stability_all:.2f}, "
            f"Agent Stabilize: {env.agent_stabilize_count}, "
            f"Non-Agent Stabilize: {env.non_agent_stabilize_count}, "
            f"Agent Fix Fingers: {env.agent_fix_fingers_count}, "
            f"Non-Agent Fix Fingers: {env.non_agent_fix_fingers_count}, "
            f"Success: {'Yes' if is_success else 'No'}"
        )
        print(result)

    success_rate = (stability_success_count / max_episodes) * 100
    print(f"\nSuccess Rate Based on Average Stability: {success_rate:.2f}%")


def main():
    print(f"Using device: {device}")
    env = gym.make('gymnasium_env/ChordWorldEnv-v0')

    RL_hyperparams = {
        "clip_grad_norm": 1.0,
        "learning_rate": 5e-5,
        "discount_factor": 0.995,
        "batch_size": 64,
        "update_frequency": 1000,
        "max_episodes": 3000,
        "max_steps": 200,
        "epsilon_max": 0.999, 
        "epsilon_min": 0.01,
        "decay_episodes": 2000,
        "memory_capacity": 10000,
        "action_space": env.action_space,
        "observation_space": env.observation_space,
    }

    DRL_Chord = Model_TrainTest(RL_hyperparams)
    train(DRL_Chord, env)
    DRL_Chord.agent.save("Chord_model.pt")

    test(DRL_Chord, env, max_episodes=40, path="Chord_model.pt")

if __name__ == '__main__':
    main()