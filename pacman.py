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
import math

# --- SETUP ---
device = "cpu"
if torch.cuda.is_available():
    device = torch.device("cuda:0")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
print(f"Device: {device}")

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

layout_name = 'bloxCapture.lay'
layout_path = os.path.join('layouts', layout_name)
env = gymPacMan_parallel_env(layout_file=layout_path,
                             display=False,
                             reward_forLegalAction=True,
                             defenceReward=False,
                             length=299,
                             enemieName='randomTeam',
                             self_play=False,
                             random_layout=False)
env.reset()

# --- 1. RAINBOW COMPONENTS (Noisy Nets) ---

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        # We stick to the paper's init for Noisy Layers to ensure exploration works
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))

    def _scale_noise(self, size):
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, input):
        if self.training:
            return F.linear(input, 
                            self.weight_mu + self.weight_sigma * self.weight_epsilon,
                            self.bias_mu + self.bias_sigma * self.bias_epsilon)
        else:
            return F.linear(input, self.weight_mu, self.bias_mu)

# --- DUELING DQN ARCHITECTURE (Upgraded) ---

class AgentQNetwork(nn.Module):
    def __init__(self, obs_shape, action_dim, hidden_dim=512):
        super(AgentQNetwork, self).__init__()
        self.conv1 = nn.Conv2d(obs_shape[0], 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        
        conv_output_shape = obs_shape[1] * obs_shape[2] * 64 
        self.flatten = nn.Flatten()
        
        self.fc_val = nn.Sequential(
            NoisyLinear(conv_output_shape, hidden_dim), 
            nn.ReLU(), 
            NoisyLinear(hidden_dim, 1)
        )
        self.fc_adv = nn.Sequential(
            NoisyLinear(conv_output_shape, hidden_dim), 
            nn.ReLU(), 
            NoisyLinear(hidden_dim, action_dim)
        )
        
        # Apply Orthogonal Initialization to Conv Layers, apparently good idea
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            # Orthogonal init with gain for ReLU
            nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('relu'))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        # Note: We do NOT strictly init NoisyLinear here as it has its own logic

    def forward(self, obs):
        x = F.relu(self.conv1(obs))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.flatten(x)
        val = self.fc_val(x) 
        adv = self.fc_adv(x) 
        q_values = val + adv - adv.mean(dim=1, keepdim=True)
        return q_values

    def reset_noise(self):
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()

# --- QMIX MIXER ---

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

        # Apply Orthogonal Init to Mixer as well (Standard Linear layers)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('relu'))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

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

# --- 2. MULTI-STEP LEARNING BUFFER ---

class NStepBuffer:
    def __init__(self, n_step, gamma):
        self.n_step = n_step
        self.gamma = gamma
        self.buffer = deque(maxlen=n_step)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        if len(self.buffer) < self.n_step: return None
        
        curr_state, curr_action, _, _, _ = self.buffer[0]
        n_state, _, _, _, n_done = self.buffer[-1]
        
        R = 0
        for i in range(self.n_step):
            r = self.buffer[i][2][0] 
            R += (self.gamma ** i) * r
            if self.buffer[i][4][0]: # If done
                n_state = self.buffer[i][3]
                n_done = self.buffer[i][4]
                break
        return (curr_state, curr_action, [R]*2, n_state, n_done)

    def reset(self):
        self.buffer.clear()

# --- 3. CLEANER PRIORITIZED REPLAY (No SumTree) ---

class NaivePrioritizedBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.pos = 0
        self.epsilon = 1e-5
    
    def add(self, experience):
        max_p = self.priorities[:len(self.buffer)].max() if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.pos] = experience
        
        self.priorities[self.pos] = max_p
        self.pos = (self.pos + 1) % self.capacity
    
    def sample(self, batch_size, beta=0.4):
        N = len(self.buffer)
        if N == 0: return None
        
        probs = self.priorities[:N] ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(N, batch_size, p=probs)
        
        weights = (N * probs[indices]) ** (-beta)
        weights /= weights.max()
        
        experiences = [self.buffer[i] for i in indices]
        
        states = np.array([e[0].cpu().numpy() for e in experiences], dtype=np.float32)
        actions = np.array([e[1] for e in experiences], dtype=np.int64)
        rewards = np.array([e[2] for e in experiences])
        next_states = np.array([e[3].cpu().numpy() for e in experiences], dtype=np.float32)
        dones = np.array([e[4] for e in experiences])
        
        return (states, actions, rewards, next_states, dones), indices, weights
    
    def update_priorities(self, indices, errors):
        for idx, err in zip(indices, errors):
            self.priorities[idx] = (err + self.epsilon)
    
    def size(self):
        return len(self.buffer)

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

def compute_td_loss(agent_q_networks, target_q_networks, mixer, target_mixer, batch, weights=None, gamma=0.99, n_step=1):
    states, actions, rewards, next_states, dones = batch
    
    states = torch.tensor(states, dtype=torch.float32).to(device)
    actions = torch.tensor(actions, dtype=torch.long).to(device)
    rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
    next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
    dones = torch.tensor(dones, dtype=torch.float32).to(device)
    if weights is not None:
        weights = torch.tensor(weights, dtype=torch.float32).to(device)
    
    agent_q_values = []
    for i, q_net in enumerate(agent_q_networks):
        q_vals = q_net(states[:, i])
        q_taken = q_vals.gather(1, actions[:, i].unsqueeze(1))
        agent_q_values.append(q_taken)
    agent_q_values = torch.cat(agent_q_values, dim=1)
    q_tot = mixer(agent_q_values, states).squeeze(-1)
    
    # Double DQN Logic
    with torch.no_grad():
        next_agent_q_values = []
        for i, (q_net, target_net) in enumerate(zip(agent_q_networks, target_q_networks)):
            next_q_vals = q_net(next_states[:, i])
            max_actions = next_q_vals.argmax(dim=1, keepdim=True)
            target_q_vals = target_net(next_states[:, i])
            next_q_taken = target_q_vals.gather(1, max_actions)
            next_agent_q_values.append(next_q_taken)
            
        next_agent_q_values = torch.cat(next_agent_q_values, dim=1)
        next_q_tot = target_mixer(next_agent_q_values, next_states).squeeze(-1)
        
        team_reward = rewards[:, 0].unsqueeze(1) 
        done_mask = dones[:, 0].unsqueeze(1) 
        
        target_q_tot = team_reward + (gamma ** n_step) * next_q_tot * (1 - done_mask)
    
    loss_elementwise = F.huber_loss(q_tot, target_q_tot, reduction='none')
    td_errors = torch.abs(target_q_tot - q_tot).detach().cpu().numpy()
    
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

def select_action(agent_q_network, state, legal_actions):
    state = torch.unsqueeze(state.clone().detach(), 0).to(device)
    with torch.no_grad():
        q_values = agent_q_network(state).cpu().numpy()
    q_values = q_values[0]
    
    best_action = legal_actions[0]
    best_val = -float('inf')
    for action in legal_actions:
        if q_values[action] > best_val:
            best_val = q_values[action]
            best_action = action
    return best_action

# --- TRAINING LOOP ---

def train_rainbow_qmix(env, agent_q_networks, target_q_networks, mixer, target_mixer, 
               replay_buffer, n_episodes=500, 
               batch_size=512, 
               gamma=0.99, lr=0.0001,
               updates_per_step=1,
               shaping_weight=0.05,
               exploration_beta=0.1, 
               n_step=3):
    
    all_params = list(mixer.parameters())
    for net in agent_q_networks:
        all_params += list(net.parameters())
        
    optimizer = optim.Adam(all_params, lr=lr, eps=1.5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_episodes, eta_min=1e-6)
    
    legal_actions = [0, 1, 2, 3, 4]
    info = None
    agent_indexes = [1, 3]
    
    episode_rewards = []
    episode_scores = []
    visit_counts = {}
    
    beta_start = 0.4
    beta_frames = n_episodes 
    beta_by_episode = lambda ep: min(1.0, beta_start + ep * (1.0 - beta_start) / beta_frames)

    n_step_buffer = NStepBuffer(n_step=n_step, gamma=gamma)

    for episode in range(n_episodes):
        done = {agent_id: False for agent_id in agent_indexes}
        env.reset()
        n_step_buffer.reset()
        
        episode_reward = 0
        score = 0
        prev_dists = {}
        has_trained = False
        
        # Reset noise
        for net in agent_q_networks:
            net.reset_noise()
            
        per_beta = beta_by_episode(episode)

        while not all(done.values()):
            actions = [-1 for _ in env.agents]
            states = []
            
            current_observations = {}

            for i, agent_index in enumerate(agent_indexes):
                obs_agent = env.get_Observation(agent_index)
                
                if shaping_weight > 0 or exploration_beta > 0:
                    current_observations[agent_index] = obs_agent
                    if shaping_weight > 0:
                        prev_dists[agent_index] = get_min_food_dist(obs_agent, agent_index)

                state = torch.as_tensor(obs_agent, dtype=torch.float32).to(device)
                states.append(state)
                current_legal = info["legal_actions"][agent_index] if info is not None else legal_actions
                
                action = select_action(agent_q_networks[i], state, current_legal)
                actions[agent_index] = action

            next_states, rewards, terminations, info = env.step(actions)
            score -= info["score_change"]
            done = {key: value for key, value in terminations.items() if key in agent_indexes}
            episode_reward += rewards[1] + rewards[3]

            augmented_rewards = {}
            for agent_index in agent_indexes:
                total_extra = 0.0
                obs_curr = list(next_states.values())[agent_index]
                
                if shaping_weight > 0:
                    curr_dist = get_min_food_dist(obs_curr, agent_index)
                    shaping = (prev_dists[agent_index] - curr_dist) * shaping_weight
                    total_extra += shaping
                
                if exploration_beta > 0:
                    total_extra += get_exploration_bonus(
                        current_observations[agent_index], 
                        visit_counts, 
                        agent_index, 
                        beta=exploration_beta
                    )
                    
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
            
            # N-STEP BUFFER
            team_reward = rewards_converted[0][0] 
            team_done = terminations_converted[0][0]
            
            transition = n_step_buffer.add(
                states_converted, 
                actions_converted, 
                [team_reward, team_reward], 
                next_states_converted, 
                [team_done, team_done]
            )
            
            if transition:
                replay_buffer.add(transition)
            
            if replay_buffer.size() >= batch_size:
                has_trained = True
                for _ in range(updates_per_step):
                    batch, idxs, weights = replay_buffer.sample(batch_size, beta=per_beta)
                    
                    loss, td_errors = compute_td_loss(
                        agent_q_networks, target_q_networks, mixer, target_mixer, 
                        batch, weights=weights, gamma=gamma, n_step=n_step
                    )
                    
                    replay_buffer.update_priorities(idxs, td_errors)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(all_params, max_norm=10.0)
                    optimizer.step()
                    soft_update_target_network(agent_q_networks, target_q_networks, tau=0.01)
                    soft_update_mixer(mixer, target_mixer, tau=0.01)

        if has_trained: scheduler.step()
        episode_rewards.append(episode_reward)
        episode_scores.append(score)
        
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"Ep {episode + 1}/{n_episodes} | Rew: {avg_reward:.2f} | LR: {scheduler.get_last_lr()[0]:.5f}")
    
    return episode_rewards, episode_scores

def plot_training_curves(rewards, scores, filename="rainbow_upgraded.png", window=20):
    import matplotlib
    matplotlib.use('Agg')
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    def moving_avg(data, w):
        if len(data) < w: return data
        return np.convolve(data, np.ones(w)/w, mode='valid')
    axes[0].plot(rewards, alpha=0.3, color='blue')
    if len(rewards) >= window:
        axes[0].plot(range(window-1, len(rewards)), moving_avg(rewards, window), color='blue', linewidth=2, label=f'{window}-ep avg')
    axes[0].set_title('Training Rewards (Upgraded Rainbow)')
    axes[0].legend()
    axes[1].plot(scores, alpha=0.3, color='green')
    if len(scores) >= window:
        axes[1].plot(range(window-1, len(scores)), moving_avg(scores, window), color='green', linewidth=2, label=f'{window}-ep avg')
    axes[1].set_title('Game Scores')
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close(fig)

def save_models(agent_nets, mixer, filename="rainbow_upgraded.pt"):
    checkpoint = {'agent_nets': [net.state_dict() for net in agent_nets], 'mixer': mixer.state_dict()}
    torch.save(checkpoint, filename)
    print(f"Model saved to {filename}")

# --- MAIN EXECUTION ---
n_agents = int(len(env.agents) / 2)
action_dim_individual_agent = 5
obs_individual_agent = env.get_Observation(0)
obs_shape = obs_individual_agent.shape

# 1. Instantiate SEPARATE networks (now hidden_dim=512 default)
agent_net_1 = AgentQNetwork(obs_shape, action_dim_individual_agent).to(device)
agent_net_3 = AgentQNetwork(obs_shape, action_dim_individual_agent).to(device)

target_net_1 = AgentQNetwork(obs_shape, action_dim_individual_agent).to(device)
target_net_3 = AgentQNetwork(obs_shape, action_dim_individual_agent).to(device)

target_net_1.load_state_dict(agent_net_1.state_dict())
target_net_3.load_state_dict(agent_net_3.state_dict())

agent_q_networks = [agent_net_1, agent_net_3]
target_q_networks = [target_net_1, target_net_3]

mixer = SimpleQMixer(n_agents=n_agents, state_shape=obs_shape, embed_dim=32).to(device)
target_mixer = SimpleQMixer(n_agents=n_agents, state_shape=obs_shape, embed_dim=32).to(device)
target_mixer.load_state_dict(mixer.state_dict())

# 2. Use Simplified Buffer
replay_buffer = NaivePrioritizedBuffer(capacity=100_000, alpha=0.6)

print("Starting Upgraded Rainbow Training (Separate Nets, 512-dim, Ortho-Init)...")
rewards_exp, scores_exp = train_rainbow_qmix(
    env, agent_q_networks, target_q_networks, mixer, target_mixer, replay_buffer,
    n_episodes=500,
    batch_size=512,
    lr=0.0001,
    gamma=0.99,
    updates_per_step=1,
    shaping_weight=0.01,
    exploration_beta=0.1, 
    n_step=3 
)

save_models(agent_q_networks, mixer, filename="rainbow_upgraded.pt")
plot_training_curves(rewards_exp, scores_exp, filename="Rainbow_Upgraded_Run.png")