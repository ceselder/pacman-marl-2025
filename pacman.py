import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from gymPacMan import gymPacMan_parallel_env
from collections import deque
import random

device = "cpu"
if torch.cuda.is_available():
    device = torch.device("cuda:0")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
print(f"Device: {device}")


if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')  # VROOM 'high' or 'medium' are good for A100 Tensor Cores. 'highest' is slower.


layout_name = 'tinyCapture.lay'
layout_path = os.path.join('layouts', layout_name)
env = gymPacMan_parallel_env(layout_file=layout_path,
                             display=False,
                             reward_forLegalAction=True,
                             defenceReward=False,
                             length=299,
                             enemieName = 'randomTeam',
                             self_play=False,
                             random_layout = False)
env.reset()

class AgentQNetwork(nn.Module):
    def __init__(self, obs_shape, action_dim, hidden_dim=128):
        super(AgentQNetwork, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(obs_shape[0], 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)

        conv_output_shape = obs_shape[1] * obs_shape[2] * 32 # assuming obs shape (C, H, W)

        # Flatten layer
        self.flatten = nn.Flatten()

        # Fully connected layers
        self.fc1 = nn.Linear(conv_output_shape, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs):
        # Pass through convolutional layers
        x = F.relu(self.conv1(obs))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Flatten the output
        x = self.flatten(x)

        x = F.relu(self.fc1(x))

        # Output Q-values
        q_values = self.fc2(x)
        return q_values

class SimpleQMixer(nn.Module):
    def __init__(self, n_agents, state_shape, embed_dim=32):
        super(SimpleQMixer, self).__init__()
        self.n_agents = n_agents
        self.embed_dim = embed_dim
        
        if isinstance(state_shape, (tuple, list)):
            self.state_dim = state_shape[0] * state_shape[1] * state_shape[2]
        else:
            self.state_dim = state_shape
        
        self.hyper_w1 = nn.Sequential(
            nn.Linear(self.state_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, n_agents * embed_dim)
        )
        self.hyper_b1 = nn.Linear(self.state_dim, embed_dim)
        self.hyper_w2 = nn.Sequential(
            nn.Linear(self.state_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.hyper_b2 = nn.Sequential(
            nn.Linear(self.state_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1)
        )

    def forward(self, agent_qs, states):
        bs = agent_qs.shape[0]
        if len(states.shape) > 2:
            state_flat = states[:, 0].reshape(bs, -1)
        else:
            state_flat = states
        
        agent_qs = agent_qs.view(bs, 1, self.n_agents)
        
        w1 = torch.abs(self.hyper_w1(state_flat))
        w1 = w1.view(bs, self.n_agents, self.embed_dim)
        b1 = self.hyper_b1(state_flat).view(bs, 1, self.embed_dim)
        w2 = torch.abs(self.hyper_w2(state_flat))
        w2 = w2.view(bs, self.embed_dim, 1)
        b2 = self.hyper_b2(state_flat).view(bs, 1, 1)
        
        hidden = torch.bmm(agent_qs, w1) + b1
        hidden = F.elu(hidden)
        q_tot = torch.bmm(hidden, w2) + b2
        
        return q_tot.view(bs, -1, 1)

def compute_td_loss(agent_q_networks, target_q_networks, mixer, target_mixer, 
                    batch, gamma=0.99):
    states, actions, rewards, next_states, dones = batch
    
    # Convert to tensors
    states = torch.tensor(states, dtype=torch.float32).to(device)
    actions = torch.tensor(actions, dtype=torch.long).to(device)
    rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
    next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
    dones = torch.tensor(dones, dtype=torch.float32).to(device)
    
    agent_q_values = []
    for agent_idx, q_net in enumerate(agent_q_networks):
        q_vals = q_net(states[:, agent_idx])
        q_taken = q_vals.gather(1, actions[:, agent_idx].unsqueeze(1))
        agent_q_values.append(q_taken)
    
    agent_q_values = torch.cat(agent_q_values, dim=1)
    q_tot = mixer(agent_q_values, states).squeeze(-1)
    
    with torch.no_grad():
        next_agent_q_values = []
        for agent_idx, (q_net, target_net) in enumerate(zip(agent_q_networks, target_q_networks)):
            next_q_vals = q_net(next_states[:, agent_idx])
            max_actions = next_q_vals.argmax(dim=1, keepdim=True)
            
            target_q_vals = target_net(next_states[:, agent_idx])
            next_q_taken = target_q_vals.gather(1, max_actions)
            next_agent_q_values.append(next_q_taken)
        
        next_agent_q_values = torch.cat(next_agent_q_values, dim=1)
        next_q_tot = target_mixer(next_agent_q_values, next_states).squeeze(-1)
        
        team_reward = rewards[:, 0, 0].unsqueeze(1)
        done_mask = dones[:, 0, 0].unsqueeze(1)
        target_q_tot = team_reward + gamma * next_q_tot * (1 - done_mask)
    
    loss = F.huber_loss(q_tot, target_q_tot)
    return loss

def soft_update_target_network(source_nets, target_nets, tau=0.01):
    for target, source in zip(target_nets, source_nets):
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(tau * source_param.data + (1 - tau) * target_param.data)

def soft_update_mixer(mixer, target_mixer, tau=0.01):
    for target_param, source_param in zip(target_mixer.parameters(), mixer.parameters()):
        target_param.data.copy_(tau * source_param.data + (1 - tau) * target_param.data)

def hard_update_mixer(mixer, target_mixer):
    target_mixer.load_state_dict(mixer.state_dict())
    
def update_target_network(agent_q_networks, target_q_networks):
    for target, source in zip(target_q_networks, agent_q_networks):
        target.load_state_dict(source.state_dict())

class ReplayBuffer:
    def __init__(self, buffer_size=50_000): 
        self.buffer = deque(maxlen=buffer_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        experiences = random.sample(self.buffer, batch_size)
        states = np.array([exp[0].cpu().numpy() for exp in experiences], dtype=np.float32)
        actions = np.array([exp[1] for exp in experiences], dtype=np.int64)
        rewards = np.array([exp[2] for exp in experiences])
        next_states = np.array([exp[3] for exp in experiences])
        dones = np.array([exp[4] for exp in experiences])
        return states, actions, rewards, next_states, dones

    def size(self):
        return len(self.buffer)

def epsilon_greedy_action(agent_q_network, state, epsilon, legal_actions):
    if random.random() < epsilon:
        action = random.choice(legal_actions)
    else:
        state = torch.unsqueeze(state.clone().detach(), 0).to(device)
        q_values = agent_q_network(state).cpu().detach().numpy()
        action = np.random.choice(np.flatnonzero(q_values == q_values.max()))
    return action

def get_exploration_bonus(obs, visit_counts, agent_index, beta=0.1):
    """
    Unified Exploration Logic.
    State Key = (Agent_ID, Position, Remaining_Target_Food)
    
    Why this works for Reward Shaping:
    1. Moving to new spots gives a bonus (Exploration).
    2. Eating food changes 'Remaining_Target_Food', effectively resetting 
       the visit count for the map. This gives a fresh burst of curiosity 
       immediately after scoring, preventing the agent from becoming lazy.
    """
    # --- 1. Find Self Position (Always Channel 1) ---
    pos = (0, 0)
    try:
        # Fast extraction of (y, x) from the one-hot layer
        ys, xs = np.nonzero(obs[1])
        if len(ys) > 0:
            pos = (int(ys[0]), int(xs[0]))
    except:
        pass

    # --- 2. Determine Target Food Count (Team Dependent) ---
    # Blue Agents (1, 3) hunt Red Food (Channel 7)
    # Red Agents (0, 2) hunt Blue Food (Channel 6)
    target_food_count = 0
    
    try:
        if agent_index in [1, 3]: 
            # Blue Team -> Count Channel 7 (Red Food)
            if 7 < obs.shape[0]: 
                target_food_count = int(np.sum(obs[7]))
        else: 
            # Red Team -> Count Channel 6 (Blue Food)
            if 6 < obs.shape[0]: 
                target_food_count = int(np.sum(obs[6]))
    except:
        pass

    # --- 3. Construct the State Key ---
    # We include agent_index so agents don't share their exploration memory.
    key = (agent_index, pos, target_food_count)

    # --- 4. Calculate & Update ---
    # Retrieve current count (default 0), increment, and save back
    current_count = visit_counts.get(key, 0) + 1
    visit_counts[key] = current_count
    
    # Return 1/sqrt(N) bonus
    return beta / np.sqrt(current_count)

    
def train_qmix(env, agent_q_networks, target_q_networks, mixer, target_mixer, 
               replay_buffer, n_episodes=500, 
               batch_size=512, 
               gamma=0.95, lr=0.001,
               exploration_beta=0.1, exploration_type='simple',
               updates_per_step=4):
    
    all_params = []
    for net in agent_q_networks:
        all_params.extend(net.parameters())
    all_params.extend(mixer.parameters())
    optimizer = optim.Adam(all_params, lr=lr)
    
    agent_q_networks = [torch.compile(net) for net in agent_q_networks] #blijkbaar sneller
    mixer = torch.compile(mixer) #blijkbaar sneller

    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.99
    legal_actions = [0, 1, 2, 3, 4]
    info = None
    agent_indexes = [1, 3]
    
    visit_counts = {} 
    episode_rewards = []
    episode_scores = []

    for episode in range(n_episodes):
        
        done = {agent_id: False for agent_id in agent_indexes}
        env.reset()
        episode_reward = 0
        score = 0
        
        while not all(done.values()):
            actions = [-1 for _ in env.agents]
            states = []
            
            # Helper to store obs for exploration calculation
            current_observations = {} 

            for i, agent_index in enumerate(agent_indexes):
                obs_agent = env.get_Observation(agent_index)
                
                # Only store if we are actually doing exploration to save CPU time
                if exploration_beta > 0:
                    current_observations[agent_index] = obs_agent 

                state = torch.as_tensor(obs_agent, dtype=torch.float32).to(device)
                states.append(state)
                
                current_legal = info["legal_actions"][agent_index] if info is not None else legal_actions
                action = epsilon_greedy_action(agent_q_networks[i], state, epsilon, current_legal)
                actions[agent_index] = action

            next_states, rewards, terminations, info = env.step(actions)
            score -= info["score_change"]
            done = {key: value for key, value in terminations.items() if key in agent_indexes}
            episode_reward += rewards[1] + rewards[3]

            augmented_rewards = {}
            for agent_index in agent_indexes:
                if exploration_beta > 0:
                    bonus = get_exploration_bonus(current_observations[agent_index], 
                                                visit_counts,
                                                agent_index, 
                                                beta=exploration_beta)
                    # Add to reward
                    augmented_rewards[agent_index] = rewards[agent_index] + bonus
                else:
                    augmented_rewards[agent_index] = rewards[agent_index]

            next_states_converted = []
            rewards_converted = []
            terminations_converted = []
            actions_converted = []

            for index in agent_indexes:
                next_states_converted.append(list(next_states.values())[index])
                rewards_converted.append(augmented_rewards[index]) 
                terminations_converted.append(terminations[index])
                actions_converted.append(actions[index])

            next_states_converted = torch.stack(next_states_converted)
            states_converted = torch.stack(states)
            rewards_converted = [rewards_converted]
            terminations_converted = [terminations_converted]
            
            replay_buffer.add((states_converted, actions_converted, rewards_converted, 
                              next_states_converted, terminations_converted))

            # Training steps
            if replay_buffer.size() >= batch_size:
                for _ in range(updates_per_step):
                    batch = replay_buffer.sample(batch_size)
                    loss = compute_td_loss(agent_q_networks, target_q_networks,
                                           mixer, target_mixer, batch, gamma=gamma)
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(all_params, max_norm=10.0)
                    optimizer.step()
                    soft_update_target_network(agent_q_networks, target_q_networks, tau=0.01)
                    soft_update_mixer(mixer, target_mixer, tau=0.01)

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        episode_rewards.append(episode_reward)
        episode_scores.append(score)
        
        if (episode + 1) % 5 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"Episode {episode + 1}/{n_episodes} | Avg Real Reward: {avg_reward:.2f} | Epsilon: {epsilon:.3f}")
    
    return episode_rewards, episode_scores



def plot_training_curves(rewards, scores, filename="training_results.png", window=20):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    def moving_avg(data, w):
        if len(data) < w: return data
        return np.convolve(data, np.ones(w)/w, mode='valid')
    
    # Plot Rewards
    axes[0].plot(rewards, alpha=0.3, color='blue')
    if len(rewards) >= window:
        axes[0].plot(range(window-1, len(rewards)), moving_avg(rewards, window), 
                     color='blue', linewidth=2, label=f'{window}-ep avg')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Episode Reward')
    axes[0].set_title('Training Rewards (Real)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot Scores
    axes[1].plot(scores, alpha=0.3, color='green')
    if len(scores) >= window:
        axes[1].plot(range(window-1, len(scores)), moving_avg(scores, window),
                     color='green', linewidth=2, label=f'{window}-ep avg')
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Game Score')
    axes[1].set_title('Game Scores')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save to file
    plt.savefig(filename)
    print(f"Plot saved successfully to: {os.path.abspath(filename)}")
    plt.close(fig) # Close memory to prevent leaks

n_agents = int(len(env.agents) / 2)
action_dim_individual_agent = 5  # North, South, East, West, Stop

obs_individual_agent = env.get_Observation(0)
obs_shape = obs_individual_agent.shape

# Create agent Q-networks
agent_q_networks = [AgentQNetwork(obs_shape=obs_shape, action_dim=action_dim_individual_agent).to(device) 
                    for _ in range(n_agents)]
target_q_networks = [AgentQNetwork(obs_shape=obs_shape, action_dim=action_dim_individual_agent).to(device) 
                     for _ in range(n_agents)]

# Initialize target networks
update_target_network(agent_q_networks, target_q_networks)

# Create mixing networks
mixer = SimpleQMixer(n_agents=n_agents, state_shape=obs_shape, embed_dim=32).to(device)
target_mixer = SimpleQMixer(n_agents=n_agents, state_shape=obs_shape, embed_dim=32).to(device)

# Initialize target mixer
hard_update_mixer(mixer, target_mixer)

# Create replay buffer
replay_buffer = ReplayBuffer(buffer_size=50_000)

# Train QMIX
rewards_exp, scores_exp = train_qmix(
    env, 
    agent_q_networks, 
    target_q_networks, 
    mixer, 
    target_mixer, 
    replay_buffer,
    n_episodes=300,        # Try fewer episodes, see if it converges faster
    batch_size=512,
    lr=0.001,
    gamma=0.99,
    exploration_beta=0.25,
    exploration_type='simple',
    updates_per_step=1
)




def save_models(agent_nets, mixer, filename="my_best_pacman.pt"):
    checkpoint = {
        'agent_nets': [net.state_dict() for net in agent_nets],
        'mixer': mixer.state_dict()
    }
    torch.save(checkpoint, filename)
    print(f"Model saved to {filename}")

# Save the final model
save_models(agent_q_networks, mixer, filename="final_model.pt")

plot_training_curves(rewards_exp, scores_exp, filename="A100_Exploration_Run.png")