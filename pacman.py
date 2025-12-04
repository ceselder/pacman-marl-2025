import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from gymPacMan import gymPacMan_parallel_env

device = "cpu"
if torch.cuda.is_available():
    device = torch.device("cuda:0")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
print(f"Device: {device}")

layout_name = 'tinyCapture.lay'                       # see 'layouts/' dir for other options
layout_path = os.path.join('layouts', layout_name)
env = gymPacMan_parallel_env(layout_file=layout_path, # see class def for options
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
    """
    Mixing network that combines individual agent Q-values into Q_tot.
    
    Key constraint: weights must be NON-NEGATIVE to ensure monotonicity.
    This guarantees: argmax_a Q_tot = (argmax_a1 Q1, argmax_a2 Q2, ...)
    So agents can act greedily on their own Q-values during execution.
    
    Architecture:
        1. Flatten global state
        2. Hypernetwork generates state-dependent weights
        3. Weights are passed through abs() for non-negativity
        4. Q_tot = sum(w_i * Q_i) + bias
    """
    def __init__(self, n_agents, state_shape, embed_dim=32):
        super(SimpleQMixer, self).__init__()
        
        self.n_agents = n_agents
        self.embed_dim = embed_dim
        
        # Calculate flattened state dimension
        # state_shape is (C, H, W) for a single agent's observation
        if isinstance(state_shape, (tuple, list)):
            self.state_dim = state_shape[0] * state_shape[1] * state_shape[2]
        else:
            self.state_dim = state_shape
        
        # Hypernetwork for first layer weights
        # Input: flattened state, Output: (n_agents * embed_dim) weights
        self.hyper_w1 = nn.Sequential(
            nn.Linear(self.state_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, n_agents * embed_dim)
        )
        
        # Hypernetwork for first layer bias (no non-negativity constraint)
        self.hyper_b1 = nn.Linear(self.state_dim, embed_dim)
        
        # Hypernetwork for second layer weights
        # Input: state, Output: embed_dim weights (to produce scalar Q_tot)
        self.hyper_w2 = nn.Sequential(
            nn.Linear(self.state_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # Hypernetwork for final bias (2-layer for expressivity)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(self.state_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1)
        )

    def forward(self, agent_qs, states):
        """
        Args:
            agent_qs: Individual agent Q-values, shape (batch, n_agents)
            states: Global state, shape (batch, n_agents, C, H, W)
                    We use first agent's obs as state proxy
        Returns:
            q_tot: Mixed Q-value, shape (batch, 1)
        """
        bs = agent_qs.shape[0]
        
        # Flatten state - use first agent's observation as global state proxy
        if len(states.shape) > 2:
            state_flat = states[:, 0].reshape(bs, -1)
        else:
            state_flat = states
        
        # Reshape agent Q-values for batched matrix multiply: (batch, 1, n_agents)
        agent_qs = agent_qs.view(bs, 1, self.n_agents)
        
        # Generate first layer weights and apply NON-NEGATIVITY via abs()
        w1 = torch.abs(self.hyper_w1(state_flat))
        w1 = w1.view(bs, self.n_agents, self.embed_dim)
        
        # First layer bias
        b1 = self.hyper_b1(state_flat).view(bs, 1, self.embed_dim)
        
        # Generate second layer weights (also non-negative)
        w2 = torch.abs(self.hyper_w2(state_flat))
        w2 = w2.view(bs, self.embed_dim, 1)
        
        # Final bias
        b2 = self.hyper_b2(state_flat).view(bs, 1, 1)
        
        # Forward pass: Q_tot = w2 * ELU(w1 * Q_agents + b1) + b2
        # (batch, 1, n_agents) @ (batch, n_agents, embed_dim) -> (batch, 1, embed_dim)
        hidden = torch.bmm(agent_qs, w1) + b1
        hidden = F.elu(hidden)
        
        # (batch, 1, embed_dim) @ (batch, embed_dim, 1) -> (batch, 1, 1)
        q_tot = torch.bmm(hidden, w2) + b2
        
        return q_tot.view(bs, -1, 1)


def compute_td_loss(agent_q_networks, target_q_networks, mixer, target_mixer, 
                    batch, gamma=0.99):
    """
    Computes the TD loss for QMix training.
    
    Key difference from IQL: we compute loss on Q_tot (mixed Q-values),
    not on individual agent Q-values separately.
    
    Args:
        agent_q_networks: List of Q-networks for each agent
        target_q_networks: List of target Q-networks for each agent
        mixer: QMix mixing network
        target_mixer: Target mixing network
        batch: (states, actions, rewards, next_states, dones)
        gamma: Discount factor
        
    Returns:
        loss: Single scalar loss for the team
    """
    states, actions, rewards, next_states, dones = batch
    
    # Convert to tensors
    states = torch.tensor(states, dtype=torch.float32).to(device)
    actions = torch.tensor(actions, dtype=torch.long).to(device)
    rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
    next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
    dones = torch.tensor(dones, dtype=torch.float32).to(device)
    
    # ========== Current Q-values ==========
    # Get Q(s,a) for each agent for the action actually taken
    agent_q_values = []
    for agent_idx, q_net in enumerate(agent_q_networks):
        q_vals = q_net(states[:, agent_idx])  # (batch, n_actions)
        q_taken = q_vals.gather(1, actions[:, agent_idx].unsqueeze(1))  # (batch, 1)
        agent_q_values.append(q_taken)
    
    # Stack into (batch, n_agents)
    agent_q_values = torch.cat(agent_q_values, dim=1)
    
    # Mix individual Q-values into Q_tot
    q_tot = mixer(agent_q_values, states)  # (batch, 1, 1)
    q_tot = q_tot.squeeze(-1)  # (batch, 1)
    
    # ========== Target Q-values (Double DQN style) ==========
    with torch.no_grad():
        next_agent_q_values = []
        for agent_idx, (q_net, target_net) in enumerate(zip(agent_q_networks, target_q_networks)):
            # Current network selects best action
            next_q_vals = q_net(next_states[:, agent_idx])
            max_actions = next_q_vals.argmax(dim=1, keepdim=True)
            
            # Target network evaluates that action
            target_q_vals = target_net(next_states[:, agent_idx])
            next_q_taken = target_q_vals.gather(1, max_actions)
            next_agent_q_values.append(next_q_taken)
        
        next_agent_q_values = torch.cat(next_agent_q_values, dim=1)  # (batch, n_agents)
        
        # Mix target Q-values
        next_q_tot = target_mixer(next_agent_q_values, next_states)  # (batch, 1, 1)
        next_q_tot = next_q_tot.squeeze(-1)  # (batch, 1)
        
        # Target = r + gamma * Q_tot_next * (1 - done)
        # Both agents get same team reward
        team_reward = rewards[:, 0, 0].unsqueeze(1)
        done_mask = dones[:, 0, 0].unsqueeze(1)
        
        target_q_tot = team_reward + gamma * next_q_tot * (1 - done_mask)
    
    # Huber loss for stability
    loss = F.huber_loss(q_tot, target_q_tot)
    
    return loss


from collections import deque
import random


def soft_update_target_network(source_nets, target_nets, tau=0.01):
    """Soft update for agent networks."""
    for target, source in zip(target_nets, source_nets):
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(tau * source_param.data + (1 - tau) * target_param.data)

def soft_update_mixer(mixer, target_mixer, tau=0.01):
    """Soft update for mixer network."""
    for target_param, source_param in zip(target_mixer.parameters(), mixer.parameters()):
        target_param.data.copy_(tau * source_param.data + (1 - tau) * target_param.data)

def hard_update_mixer(mixer, target_mixer):
    """Hard update for mixer network."""
    target_mixer.load_state_dict(mixer.state_dict())



class ReplayBuffer:
    def __init__(self, buffer_size=10_000):
        self.buffer = deque(maxlen=buffer_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        experiences = random.sample(self.buffer, batch_size)

        # Restructure the batch into separate arrays for states, actions, rewards, next_states, and dones
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
        # Explore: take a random action
        action = random.choice(legal_actions)
    else:
        state = torch.unsqueeze(state.clone().detach(), 0).to(device)
        q_values = agent_q_network(state).cpu().detach().numpy()
        action = np.random.choice(np.flatnonzero(q_values == q_values.max()))

    return action

def update_target_network(agent_q_networks, target_q_networks):
    for target, source in zip(target_q_networks, agent_q_networks):
        target.load_state_dict(source.state_dict())

def soft_update_target_network(agent_q_networks, target_q_networks, tau=0.01):
    for target, source in zip(target_q_networks, agent_q_networks):
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                tau * source_param.data + (1 - tau) * target_param.data
            )
def train_qmix(env, agent_q_networks, target_q_networks, mixer, target_mixer,
               replay_buffer, n_episodes=500, batch_size=32, gamma=0.95, lr=0.001):
    """
    Training loop for QMIX.
    
    Key changes from IQL:
    1. Single optimizer for both agent networks AND mixer
    2. Single loss computed on Q_tot (not separate per-agent losses)
    3. Must update target mixer alongside target agent networks
    """
    # Single optimizer for all parameters
    all_params = []
    for net in agent_q_networks:
        all_params.extend(net.parameters())
    all_params.extend(mixer.parameters())
    optimizer = optim.Adam(all_params, lr=lr)

    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.99
    legal_actions = [0, 1, 2, 3, 4]
    info = None
    agent_indexes = [1, 3]  # Blue team agent indices
    
    # For logging
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
            
            # Collect observations and select actions for each agent
            for i, agent_index in enumerate(agent_indexes):
                obs_agent = env.get_Observation(agent_index)
                state = torch.tensor(obs_agent, dtype=torch.float32).to(device)
                states.append(state)
                
                # Get legal actions if available
                current_legal = info["legal_actions"][agent_index] if info is not None else legal_actions
                action = epsilon_greedy_action(agent_q_networks[i], state, epsilon, current_legal)
                actions[agent_index] = action

            # Environment step
            next_states, rewards, terminations, info = env.step(actions)
            score -= info["score_change"]
            done = {key: value for key, value in terminations.items() if key in agent_indexes}
            episode_reward += rewards[1] + rewards[3]  # Sum of team rewards

            # Prepare experience for replay buffer
            next_states_converted = []
            rewards_converted = []
            terminations_converted = []
            actions_converted = []

            for index in agent_indexes:
                next_states_converted.append(list(next_states.values())[index])
                rewards_converted.append(rewards[index])
                terminations_converted.append(terminations[index])
                actions_converted.append(actions[index])

            next_states_converted = torch.stack(next_states_converted)
            states_converted = torch.stack(states)
            rewards_converted = [rewards_converted]
            terminations_converted = [terminations_converted]
            
            replay_buffer.add((states_converted, actions_converted, rewards_converted, 
                              next_states_converted, terminations_converted))

            # Training step
            if replay_buffer.size() >= batch_size:
                batch = replay_buffer.sample(batch_size)
                
                # Compute QMIX loss (single loss for the team)
                loss = compute_td_loss(agent_q_networks, target_q_networks,
                                       mixer, target_mixer, batch, gamma=gamma)
                
                optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(all_params, max_norm=10.0)
                
                optimizer.step()

                # Soft update both agent networks and mixer
                soft_update_target_network(agent_q_networks, target_q_networks, tau=0.01)
                soft_update_mixer(mixer, target_mixer, tau=0.01)

        # Decay exploration
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        
        # Logging
        episode_rewards.append(episode_reward)
        episode_scores.append(score)
        
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_score = np.mean(episode_scores[-10:])
            print(f"Episode {episode + 1}/{n_episodes} | "
                  f"Avg Reward: {avg_reward:.2f} | Avg Score: {avg_score:.2f} | "
                  f"Epsilon: {epsilon:.3f}")
    
    return episode_rewards, episode_scores
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
replay_buffer = ReplayBuffer(buffer_size=10_000)

# Train QMIX
episode_rewards, episode_scores = train_qmix(
    env, agent_q_networks, target_q_networks, 
    mixer, target_mixer, replay_buffer,
    n_episodes=500, batch_size=32, gamma=0.95, lr=0.001
)


# plot code
def plot_training_curves(rewards, scores, window=20):
    """Plot training progress."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Smooth curves with moving average
    def moving_avg(data, w):
        return np.convolve(data, np.ones(w)/w, mode='valid')
    
    # Rewards
    axes[0].plot(rewards, alpha=0.3, color='blue')
    if len(rewards) >= window:
        axes[0].plot(range(window-1, len(rewards)), moving_avg(rewards, window), 
                     color='blue', linewidth=2, label=f'{window}-ep avg')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Episode Reward')
    axes[0].set_title('Training Rewards')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Scores
    axes[1].plot(scores, alpha=0.3, color='green')
    if len(scores) >= window:
        axes[1].plot(range(window-1, len(scores)), moving_avg(scores, window),
                     color='green', linewidth=2, label=f'{window}-ep avg')
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Score')
    axes[1].set_title('Training Scores')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

