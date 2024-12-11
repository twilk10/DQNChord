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

# Set seeds for reproducibility
seed = 42 
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

        self.FC = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, num_actions)
            )

        # self.FC = nn.Sequential(
        #         nn.Linear(input_dim, 12),
        #         nn.ReLU(inplace=True),
        #         nn.Linear(12, 8),
        #         nn.ReLU(inplace=True),
        #         nn.Linear(8, num_actions)
        #     )
        
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
        self.step_loss_history = []
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
        
        self.clip_grad_norm = clip_grad_norm
        self.criterion = nn.SmoothL1Loss()
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

        self.step_loss_history.append(loss.item())

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

    plt.figure()
    plt.title("Training Step Loss")
    plt.plot(agent.agent.step_loss_history, label='Per-step Loss', color='blue', alpha=0.6)
    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

import torch
import matplotlib.pyplot as plt
import seaborn as sns

def test(agent, env, max_episodes, path, seed=42):
    agent.agent.main_network.load_state_dict(torch.load(path))
    agent.agent.main_network.eval()

    stability_success_count = 0
    stability_threshold = 0.8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Initialize lists to store metrics
    episodes = []
    agent_avg_stabilities = []
    all_nodes_avg_stabilities = []
    agent_stabilize_counts = []
    non_agent_stabilize_counts = []
    agent_fix_fingers_counts = []
    non_agent_fix_fingers_counts = []
    success_flags = []

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
            next_state, reward, done, truncation, info = env.step(action)
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

        # Append metrics to lists
        episodes.append(episode)
        agent_avg_stabilities.append(average_stability)
        all_nodes_avg_stabilities.append(info.get("average_stability_all", 0.0))
        agent_stabilize_counts.append(info.get("agent_stabilize_count", 0))
        non_agent_stabilize_counts.append(info.get("non_agent_stabilize_count", 0))
        agent_fix_fingers_counts.append(info.get("agent_fix_fingers_count", 0))
        non_agent_fix_fingers_counts.append(info.get("non_agent_fix_fingers_count", 0))
        success_flags.append(is_success)

        # Print episode result with additional metrics
        result = (
            f"Episode: {episode}, "
            f"Steps: {step_size}, "
            f"Reward: {episode_reward:.2f}, "
            f"Agent Avg Stability: {average_stability:.2f}, "
            f"All Nodes Avg Stability: {info.get('average_stability_all', 0.0):.2f}, "
            f"Agent Stabilize: {info.get('agent_stabilize_count', 0)}, "
            f"Non-Agent Stabilize: {info.get('non_agent_stabilize_count', 0)}, "
            f"Agent Fix Fingers: {info.get('agent_fix_fingers_count', 0)}, "
            f"Non-Agent Fix Fingers: {info.get('non_agent_fix_fingers_count', 0)}, "
            f"total_drop_join: {info.get('total_drop_join', 0)}, "
            f"Success: {'Yes' if is_success else 'No'}"
        )
        print(result)

    success_rate = (stability_success_count / max_episodes) * 100
    print(f"\nSuccess Rate Based on Average Stability: {success_rate:.2f}%")

    # Plotting
    sns.set(style="whitegrid")
    plt.figure(figsize=(15, 20))

    # Plot Agent Avg Stability vs All Nodes Avg Stability
    plt.subplot(3, 2, 1)
    plt.plot(episodes, agent_avg_stabilities, label='Agent Avg Stability', color='blue')
    plt.plot(episodes, all_nodes_avg_stabilities, label='All Nodes Avg Stability', color='green')
    plt.xlabel('Episode')
    plt.ylabel('Average Stability')
    plt.title('Average Stability per Episode')
    plt.legend()

    # Plot Agent Stabilize vs Non-Agent Stabilize
    plt.subplot(3, 2, 2)
    plt.plot(episodes, agent_stabilize_counts, label='Agent Stabilize Count', color='red')
    plt.plot(episodes, non_agent_stabilize_counts, label='Non-Agent Stabilize Count', color='orange')
    plt.xlabel('Episode')
    plt.ylabel('Stabilize Count')
    plt.title('Stabilize Counts per Episode')
    plt.legend()

    # Plot Agent Fix Fingers vs Non-Agent Fix Fingers
    plt.subplot(3, 2, 3)
    plt.plot(episodes, agent_fix_fingers_counts, label='Agent Fix Fingers Count', color='purple')
    plt.plot(episodes, non_agent_fix_fingers_counts, label='Non-Agent Fix Fingers Count', color='brown')
    plt.xlabel('Episode')
    plt.ylabel('Fix Fingers Count')
    plt.title('Fix Fingers Counts per Episode')
    plt.legend()

    # # Plot Success Rate
    # plt.subplot(3, 2, 4)
    # plt.plot(episodes, success_flags, label='Success', color='cyan')
    # plt.xlabel('Episode')
    # plt.ylabel('Success (1=Yes, 0=No)')
    # plt.title('Success per Episode')
    # plt.legend()

    # # Plot Cumulative Success Rate
    # cumulative_success = [sum(success_flags[:i+1]) / (i+1) * 100 for i in range(len(success_flags))]
    # plt.subplot(3, 2, 5)
    # plt.plot(episodes, cumulative_success, label='Cumulative Success Rate', color='magenta')
    # plt.xlabel('Episode')
    # plt.ylabel('Cumulative Success Rate (%)')
    # plt.title('Cumulative Success Rate over Episodes')
    # plt.legend()

    # Adjust layout and show plots
    plt.tight_layout()
    plt.show()

    # Optionally, save the plots to a file
    # plt.savefig('test_metrics_plots.png')



def main():
    print(device)
    env = gym.make('gymnasium_env/ChordWorldEnv-v0')
    # env_20 = gym.make('FrozenLake-v1', desc=custom_map_20, is_slippery=False, max_episode_steps=100)

    RL_hyperparams = {
        "clip_grad_norm": 1.0,
        "learning_rate": 1e-4,
        "discount_factor": 0.99,
        "batch_size": 64,
        "update_frequency": 1000,
        "max_episodes": 3000,
        "max_steps": 200,
        "epsilon_max": 0.999, 
        "epsilon_min": 0.01,
        "decay_episodes": 2500,
        "memory_capacity": 20000,
        "action_space": env.action_space,
        "observation_space": env.observation_space,
    }

    DRL_Chord = Model_TrainTest(RL_hyperparams)
    # train(DRL_Chord, env)
    # DRL_Chord.agent.save("Chord_model_churn_20.pt")

    # test(DRL_Chord, env, max_episodes=40, path="Chord_model_churn_20.pt")
    test(DRL_Chord, env, max_episodes=1000, path="Chord_model.pt")

if __name__ == '__main__':
    main()