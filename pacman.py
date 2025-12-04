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

# --- SETUP ---
device = "cpu"
if torch.cuda.is_available():
    device = torch.device("cuda:0")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
print(f"Device: {device}")

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')

layout_name = 'bloxCapture.lay'
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

# --- DUELING DQN ARCHITECTURE ---
class AgentQNetwork(nn.Module):
    def __init__(self, obs_shape, action_dim, hidden_dim=256):
        super(AgentQNetwork, self).__init__()
        self.conv1 = nn.Conv2d(obs_shape[0], 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        conv_output_shape = obs_shape[1] * obs_shape[2] * 64 
        self.flatten = nn.Flatten()
        self.fc_val = nn.Sequential(nn.Linear(conv_output_shape, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        self.fc_adv = nn.Sequential(nn.Linear(conv_output_shape, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, action_dim))

    def forward(self, obs):
        x = F.relu(self.conv1(obs))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.flatten(x)
        val = self.fc_val(x) 
        adv = self.fc_adv(x) 
        q_values = val + adv - adv.mean(dim=1, keepdim=True)
        return q_values

class SimpleQMixer(nn.Module):
    def __init__(self, n_agents, state_shape, embed_dim=32):
        super(SimpleQMixer, self).__init__()
        self.n_agents = n_agents
        if isinstance(state_shape, (tuple, list)):
            self.state_dim = state_shape[0] * state_shape[1] * state_shape[2]
        else:
            self.state_dim = state_shape
        self.hyper_w1 = nn.Sequential(nn.Linear(self.state_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, n_agents * embed_dim))
        self.hyper_b1 = nn.Linear(self.state_dim, embed_dim)
        self.hyper_w2 = nn.Sequential(nn.Linear(self.state_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, embed_dim))
        self.hyper_b2 = nn.Sequential(nn.Linear(self.state_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, 1))

    def forward(self, agent_qs, states):
        bs = agent_qs.shape[0]
        if len(states.shape) > 2: state_flat = states[:, 0].reshape(bs, -1)
        else: state_flat = states
        agent_qs = agent_qs.view(bs, 1, self.n_agents)
        w1 = torch.abs(self.hyper_w1(state_flat)).view(bs, self.n_agents, -1)
        b1 = self.hyper_b1(state_flat).view(bs, 1, -1)
        w2 = torch.abs(self.hyper_w2(state_flat)).view(bs, -1, 1)
        b2 = self.hyper_b2(state_flat).view(bs, 1, 1)
        hidden = F.elu(torch.bmm(agent_qs, w1) + b1)
        q_tot = torch.bmm(hidden, w2) + b2
        return q_tot.view(bs, -1, 1)

# --- PRIORITIZED REPLAY BUFFER INFRASTRUCTURE ---

class SumTree:
    """
    Binary heap for O(log N) sampling of priorities.
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.n_entries = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[dataIdx])

class PrioritizedReplayBuffer:
    def __init__(self, buffer_size=50_000, alpha=0.6):
        self.tree = SumTree(buffer_size)
        self.alpha = alpha  # How much prioritization is used (0 - no priority, 1 - full priority)
        self.epsilon = 0.01  # Small amount to avoid zero priority

    def add(self, experience):
        # New experiences get max priority so they are trained on at least once
        max_priority = np.max(self.tree.tree[-self.tree.capacity:])
        if max_priority == 0:
            max_priority = 1.0
        self.tree.add(max_priority, experience)

    def sample(self, batch_size, beta=0.4):
        experiences = []
        idxs = []
        priorities = []
        segment = self.tree.total() / batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            experiences.append(data)
            idxs.append(idx)

        # Calculate Importance Sampling Weights
        sampling_probabilities = np.array(priorities) / self.tree.total()
        is_weights = np.power(self.tree.n_entries * sampling_probabilities, -beta)
        is_weights /= is_weights.max() # Normalize

        # Unpack experience
        states = np.array([exp[0].cpu().numpy() for exp in experiences], dtype=np.float32)
        actions = np.array([exp[1] for exp in experiences], dtype=np.int64)
        rewards = np.array([exp[2] for exp in experiences])
        next_states = np.array([exp[3] for exp in experiences])
        dones = np.array([exp[4] for exp in experiences])

        return (states, actions, rewards, next_states, dones), idxs, is_weights

    def update_priorities(self, idxs, errors):
        for idx, error in zip(idxs, errors):
            p = (error + self.epsilon) ** self.alpha
            self.tree.update(idx, p)

    def size(self):
        return self.tree.n_entries

# --- HELPERS ---

def get_min_food_dist(obs, agent_index):
    try:
        ys, xs = np.nonzero(obs[1])
        if len(ys) == 0: return 0 
        my_pos = np.array([ys[0], xs[0]])
        target_ch = 7 if agent_index in [1, 3] else 6
        if target_ch >= obs.shape[0]: return 0
        f_ys, f_xs = np.nonzero(obs[target_ch])
        if len(f_ys) == 0: return 0 
        dists = np.sum(np.abs(np.stack([f_ys, f_xs], axis=1) - my_pos), axis=1)
        return np.min(dists)
    except: return 0

def get_exploration_bonus(obs, visit_counts, agent_index, beta=0.1):
    pos = (0, 0)
    try:
        ys, xs = np.nonzero(obs[1])
        if len(ys) > 0: pos = (int(ys[0]), int(xs[0]))
    except: pass
    team_id = agent_index % 2
    key = (team_id, pos)
    visit_counts[key] = visit_counts.get(key, 0) + 1
    return beta / np.sqrt(visit_counts[key])

def compute_td_loss(agent_q_networks, target_q_networks, mixer, target_mixer, batch, weights=None, gamma=0.99):
    states, actions, rewards, next_states, dones = batch
    
    states = torch.tensor(states, dtype=torch.float32).to(device)
    actions = torch.tensor(actions, dtype=torch.long).to(device)
    rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
    next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
    dones = torch.tensor(dones, dtype=torch.float32).to(device)
    if weights is not None:
        weights = torch.tensor(weights, dtype=torch.float32).to(device)
    
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
    
    # PER CHANGE: reduction='none' allows us to weight errors individually
    loss_elementwise = F.huber_loss(q_tot, target_q_tot, reduction='none')
    
    # Calculate TD Error for priority updates (Absolute error)
    td_errors = torch.abs(target_q_tot - q_tot).detach().cpu().numpy()
    
    # Apply Importance Sampling Weights
    if weights is not None:
        loss = (loss_elementwise * weights).mean()
    else:
        loss = loss_elementwise.mean()
        
    return loss, td_errors

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

def epsilon_greedy_action(agent_q_network, state, epsilon, legal_actions):
    if random.random() < epsilon:
        action = random.choice(legal_actions)
    else:
        state = torch.unsqueeze(state.clone().detach(), 0).to(device)
        q_values = agent_q_network(state).cpu().detach().numpy()
        action = np.random.choice(np.flatnonzero(q_values == q_values.max()))
    return action

# --- TRAINING LOOP ---

def train_qmix(env, agent_q_networks, target_q_networks, mixer, target_mixer, 
               replay_buffer, n_episodes=500, 
               batch_size=512, 
               gamma=0.95, lr=0.001,
               exploration_beta=0.1,
               updates_per_step=1,
               shaping_weight=0.05):
    
    all_params = list(agent_q_networks[0].parameters()) + list(mixer.parameters())
    optimizer = optim.Adam(all_params, lr=lr)
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_episodes, eta_min=1e-5)
    
    if hasattr(torch, 'compile'):
        try:
            agent_q_networks[0] = torch.compile(agent_q_networks[0])
            agent_q_networks[1] = agent_q_networks[0] 
            mixer = torch.compile(mixer)
        except: pass

    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.99
    legal_actions = [0, 1, 2, 3, 4]
    info = None
    agent_indexes = [1, 3]
    
    visit_counts = {} 
    episode_rewards = []
    episode_scores = []
    
    # PER Beta Annealing
    beta_start = 0.4
    beta_frames = n_episodes 
    beta_by_episode = lambda ep: min(1.0, beta_start + ep * (1.0 - beta_start) / beta_frames)

    for episode in range(n_episodes):
        done = {agent_id: False for agent_id in agent_indexes}
        env.reset()
        episode_reward = 0
        score = 0
        prev_dists = {}
        has_trained = False
        
        # Calculate PER beta for this episode
        per_beta = beta_by_episode(episode)

        while not all(done.values()):
            actions = [-1 for _ in env.agents]
            states = []
            current_observations = {} 

            for i, agent_index in enumerate(agent_indexes):
                obs_agent = env.get_Observation(agent_index)
                if exploration_beta > 0 or shaping_weight > 0:
                    current_observations[agent_index] = obs_agent 
                    prev_dists[agent_index] = get_min_food_dist(obs_agent, agent_index)

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
                total_extra = 0.0
                obs_curr = list(next_states.values())[agent_index]
                if exploration_beta > 0:
                    total_extra += get_exploration_bonus(current_observations[agent_index], visit_counts, agent_index, beta=exploration_beta)
                if shaping_weight > 0:
                    curr_dist = get_min_food_dist(obs_curr, agent_index)
                    shaping = (prev_dists[agent_index] - curr_dist) * shaping_weight
                    total_extra += shaping
                augmented_rewards[agent_index] = rewards[agent_index] + total_extra

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
            
            replay_buffer.add((states_converted, actions_converted, rewards_converted, next_states_converted, terminations_converted))

            if replay_buffer.size() >= batch_size:
                has_trained = True
                for _ in range(updates_per_step):
                    # PER Sampling
                    batch, idxs, weights = replay_buffer.sample(batch_size, beta=per_beta)
                    
                    # Compute Loss with IS weights
                    loss, td_errors = compute_td_loss(agent_q_networks, target_q_networks, mixer, target_mixer, batch, weights=weights, gamma=gamma)
                    
                    # Update Priorities
                    replay_buffer.update_priorities(idxs, td_errors)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
                    optimizer.step()
                    soft_update_target_network(agent_q_networks, target_q_networks, tau=0.01)
                    soft_update_mixer(mixer, target_mixer, tau=0.01)

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        if has_trained: scheduler.step()
        episode_rewards.append(episode_reward)
        episode_scores.append(score)
        
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"Ep {episode + 1}/{n_episodes} | Rew: {avg_reward:.2f} | Eps: {epsilon:.3f} | LR: {scheduler.get_last_lr()[0]:.5f}")
    
    return episode_rewards, episode_scores

def plot_training_curves(rewards, scores, filename="training_results.png", window=20):
    import matplotlib
    matplotlib.use('Agg')
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    def moving_avg(data, w):
        if len(data) < w: return data
        return np.convolve(data, np.ones(w)/w, mode='valid')
    axes[0].plot(rewards, alpha=0.3, color='blue')
    if len(rewards) >= window:
        axes[0].plot(range(window-1, len(rewards)), moving_avg(rewards, window), color='blue', linewidth=2, label=f'{window}-ep avg')
    axes[0].set_title('Training Rewards (Real)')
    axes[0].legend()
    axes[1].plot(scores, alpha=0.3, color='green')
    if len(scores) >= window:
        axes[1].plot(range(window-1, len(scores)), moving_avg(scores, window), color='green', linewidth=2, label=f'{window}-ep avg')
    axes[1].set_title('Game Scores')
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close(fig)

def save_models(agent_nets, mixer, filename="my_best_pacman.pt"):
    checkpoint = {'agent_nets': [net.state_dict() for net in agent_nets], 'mixer': mixer.state_dict()}
    torch.save(checkpoint, filename)
    print(f"Model saved to {filename}")

# --- MAIN EXECUTION ---
n_agents = int(len(env.agents) / 2)
action_dim_individual_agent = 5
obs_individual_agent = env.get_Observation(0)
obs_shape = obs_individual_agent.shape

shared_agent = AgentQNetwork(obs_shape=obs_shape, action_dim=action_dim_individual_agent).to(device)
shared_target = AgentQNetwork(obs_shape=obs_shape, action_dim=action_dim_individual_agent).to(device)
shared_target.load_state_dict(shared_agent.state_dict())
agent_q_networks = [shared_agent, shared_agent]
target_q_networks = [shared_target, shared_target]
mixer = SimpleQMixer(n_agents=n_agents, state_shape=obs_shape, embed_dim=32).to(device)
target_mixer = SimpleQMixer(n_agents=n_agents, state_shape=obs_shape, embed_dim=32).to(device)
hard_update_mixer(mixer, target_mixer)

# PER: Initialize PrioritizedReplayBuffer
replay_buffer = PrioritizedReplayBuffer(buffer_size=100_000, alpha=0.6)

rewards_exp, scores_exp = train_qmix(
    env, agent_q_networks, target_q_networks, mixer, target_mixer, replay_buffer,
    n_episodes=1500,
    batch_size=1024,
    lr=0.0005,
    gamma=0.999,
    updates_per_step=1,
    exploration_beta=0.08,
    shaping_weight=0.01,
)

save_models(agent_q_networks, mixer, filename="final_model.pt")
plot_training_curves(rewards_exp, scores_exp, filename="A100_Exploration_Run.png")