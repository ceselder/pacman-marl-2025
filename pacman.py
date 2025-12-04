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


def get_exploration_bonus(obs, visit_counts, beta=0.1, state_type='simple'):
    """
    Calculates exploration bonus and updates the visit_counts dictionary in-place.
    """
    # --- 1. Extract Position (State Key) ---
    # We assume Channel 4 is the agent (standard for this env). 
    # np.nonzero returns indices of non-zero elements.
    try:
        # obs is (Channels, Height, Width). obs[4] is the agent layer.
        ys, xs = np.nonzero(obs[4])
        pos = (ys[0], xs[0]) if len(ys) > 0 else (0, 0)
    except:
        pos = (0, 0) # Fallback

    # Define the key based on strategy
    if state_type == 'simple':
        key = pos
    elif state_type == 'food':
        # Channel 1 is usually food. Add food count to state.
        food_count = int(np.sum(obs[1]))
        key = (pos, food_count)
    else:
        key = pos

    # --- 2. Update Counts & Calculate Bonus ---
    # Get current count (default 0), add 1
    current_count = visit_counts.get(key, 0) + 1
    visit_counts[key] = current_count

    # Bonus formula: beta / sqrt(count)
    return beta / np.sqrt(current_count)

    
def train_qmix(env, agent_q_networks, target_q_networks, mixer, target_mixer, 
               replay_buffer, n_episodes=500, batch_size=32, gamma=0.95, lr=0.001,
               # code celeste for exploration
               exploration_beta=0.1, exploration_type='simple'):
    
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
    
    # code celeste for exploration
    # Initialize dictionary to track visits across the entire training run
    visit_counts = {} 
    
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
            
            # code celeste for exploration
            # We need to store the raw numpy observations to calculate bonuses later
            current_observations = {} 

            # Collect observations and select actions for each agent
            for i, agent_index in enumerate(agent_indexes):
                obs_agent = env.get_Observation(agent_index)
                
                # code celeste for exploration
                current_observations[agent_index] = obs_agent 

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
            
            # Log the REAL reward (without bonus) for the graph
            episode_reward += rewards[1] + rewards[3]

            # code celeste for exploration
            # --- Calculate Augmented Rewards (Real Reward + Exploration Bonus) ---
            augmented_rewards = {}
            for agent_index in agent_indexes:
                # Calculate bonus using the helper function
                bonus = get_exploration_bonus(current_observations[agent_index], 
                                              visit_counts, 
                                              beta=exploration_beta, 
                                              state_type=exploration_type)
                
                # The agent learns from (Reward + Bonus)
                augmented_rewards[agent_index] = rewards[agent_index] + bonus
            # -------------------------------------------------------------------

            # Prepare experience for replay buffer
            next_states_converted = []
            rewards_converted = []
            terminations_converted = []
            actions_converted = []

            for index in agent_indexes:
                next_states_converted.append(list(next_states.values())[index])
                
                # code celeste for exploration
                # Push the AUGMENTED reward to the buffer
                rewards_converted.append(augmented_rewards[index]) 
                
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
                
                # Compute QMIX loss
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
                  f"Avg Real Reward: {avg_reward:.2f} | Avg Score: {avg_score:.2f} | "
                  f"Epsilon: {epsilon:.3f}")
    
    return episode_rewards, episode_scores