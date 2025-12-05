import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from gymPacMan import gymPacMan_parallel_env
from collections import deque
import math

# --- SETUP ---
device = "cpu"
if torch.cuda.is_available():
    device = torch.device("cuda:0")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
print(f"Device: {device}")


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
        w1 = torch.abs(self.hyper_w1(state_flat)).view(bs, self.n_agents, -1)
        b1 = self.hyper_b1(state_flat).view(bs, 1, -1)
        w2 = torch.abs(self.hyper_w2(state_flat)).view(bs, -1, 1)
        b2 = self.hyper_b2(state_flat).view(bs, 1, 1)
        hidden = F.elu(torch.bmm(agent_qs, w1) + b1)
        q_tot = torch.bmm(hidden, w2) + b2
        return q_tot.view(bs, -1, 1)


# --- N-STEP BUFFER ---
class NStepBuffer:
    def __init__(self, n_step, gamma):
        self.n_step = n_step
        self.gamma = gamma
        self.buffer = deque(maxlen=n_step)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        if len(self.buffer) < self.n_step:
            return None
        
        curr_state, curr_action, _, _, _ = self.buffer[0]
        n_state = self.buffer[-1][3]
        n_done = self.buffer[-1][4]
        
        R = 0.0
        for i in range(self.n_step):
            R += (self.gamma ** i) * self.buffer[i][2]
            if self.buffer[i][4]:
                n_state = self.buffer[i][3]
                n_done = True
                break
        
        return (curr_state, curr_action, R, n_state, n_done)

    def reset(self):
        self.buffer.clear()


# --- PER BUFFER ---
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
        if N == 0:
            return None
        probs = self.priorities[:N] ** self.alpha
        probs /= probs.sum()
        indices = np.random.choice(N, batch_size, p=probs, replace=False)
        weights = (N * probs[indices]) ** (-beta)
        weights /= weights.max()
        experiences = [self.buffer[i] for i in indices]
        
        states = np.array([e[0].cpu().numpy() for e in experiences], dtype=np.float32)
        actions = np.array([e[1] for e in experiences], dtype=np.int64)
        rewards = np.array([e[2] for e in experiences], dtype=np.float32)
        next_states = np.array([e[3].cpu().numpy() for e in experiences], dtype=np.float32)
        dones = np.array([e[4] for e in experiences], dtype=np.float32)
        
        return (states, actions, rewards, next_states, dones), indices, weights
    
    def update_priorities(self, indices, errors):
        for idx, err in zip(indices, errors):
            # Casting to float avoids DeprecationWarning
            self.priorities[idx] = float(err) + self.epsilon
    
    def size(self):
        return len(self.buffer)


# --- HELPERS ---
def get_exploration_bonus(obs, visit_counts, agent_index, beta=0.1):
    nonzero = np.nonzero(obs[1]).tolist()[0]
        
    pos = (int(nonzero[0]), int(nonzero[1]))
    
    team_id = agent_index % 2
    key = (team_id, pos)
    visit_counts[key] = visit_counts.get(key, 0) + 1
    return beta / np.sqrt(visit_counts[key])


def compute_td_loss(agent_net, target_net, mixer, target_mixer, batch, 
                    weights=None, gamma=0.99, n_step=1):
    states, actions, rewards, next_states, dones = batch
    
    batch_size = states.shape[0]
    n_agents = states.shape[1]
    
    states = torch.tensor(states, dtype=torch.float32, device=device)
    actions = torch.tensor(actions, dtype=torch.long, device=device)
    rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
    next_states = torch.tensor(next_states, dtype=torch.float32, device=device)
    dones = torch.tensor(dones, dtype=torch.float32, device=device)
    
    if weights is not None:
        weights = torch.tensor(weights, dtype=torch.float32, device=device)
    
    # Flatten states to process all agents in one forward pass
    states_flat = states.view(-1, *states.shape[2:])
    q_vals_all = agent_net(states_flat)
    q_vals_all = q_vals_all.view(batch_size, n_agents, -1)
    
    actions_expanded = actions.unsqueeze(-1)
    agent_q_values = q_vals_all.gather(2, actions_expanded).squeeze(-1)
    
    q_tot = mixer(agent_q_values, states).squeeze(-1)
    
    with torch.no_grad():
        next_states_flat = next_states.view(-1, *next_states.shape[2:])
        
        next_q_online = agent_net(next_states_flat).view(batch_size, n_agents, -1)
        max_actions = next_q_online.argmax(dim=2, keepdim=True)
        
        next_q_target = target_net(next_states_flat).view(batch_size, n_agents, -1)
        next_agent_q_values = next_q_target.gather(2, max_actions).squeeze(-1)
        
        next_q_tot = target_mixer(next_agent_q_values, next_states).squeeze(-1)
        
        target_q_tot = rewards.unsqueeze(1) + (gamma ** n_step) * next_q_tot * (1 - dones.unsqueeze(1))
    
    loss_elementwise = F.huber_loss(q_tot, target_q_tot, reduction='none')
    td_errors = torch.abs(target_q_tot - q_tot).detach().cpu().numpy().flatten()
    
    if weights is not None:
        loss = (loss_elementwise.squeeze() * weights).mean()
    else:
        loss = loss_elementwise.mean()
    
    return loss, td_errors


def soft_update(source_net, target_net, tau=0.01):
    for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
        target_param.data.copy_(tau * source_param.data + (1 - tau) * target_param.data)


def select_action(agent_net, state, legal_actions):
    state = state.unsqueeze(0).to(device)
    with torch.no_grad():
        q_values = agent_net(state).cpu().numpy()[0]
    
    best_action = max(legal_actions, key=lambda a: q_values[a])
    return best_action


# --- TRAINING LOOP ---
def train_rainbow_qmix(env, agent_net, target_net, mixer, target_mixer, 
                       replay_buffer, n_episodes=500, batch_size=64, 
                       gamma=0.98, lr=0.0001, exploration_beta=0.2, n_step=3,
                       tau=0.005):
    
    all_params = list(mixer.parameters()) + list(agent_net.parameters())
    optimizer = optim.Adam(all_params, lr=lr, eps=1.5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_episodes, eta_min=1e-6)
    
    legal_actions = [0, 1, 2, 3, 4]
    info = None
    agent_indexes = [1, 3]
    
    episode_rewards = []
    episode_scores = []
    visit_counts = {}
    
    beta_start = 0.4
    beta_by_episode = lambda ep: min(1.0, beta_start + ep * (1.0 - beta_start) / n_episodes)
    
    n_step_buffer = NStepBuffer(n_step=n_step, gamma=gamma)

    for episode in range(n_episodes):
        env.reset()
        n_step_buffer.reset()
        done = {agent_id: False for agent_id in agent_indexes}
        
        episode_reward = 0.0
        score = 0.0
        has_trained = False
        
        # Reset noise at start of episode for consistent exploration strategy (inference)
        agent_net.reset_noise()
        per_beta = beta_by_episode(episode)

        while not all(done.values()):
            states = []
            actions_taken = []
            current_observations = {}

            for agent_index in agent_indexes:
                obs_agent = env.get_Observation(agent_index)
                current_observations[agent_index] = obs_agent
                
                state = torch.as_tensor(obs_agent, dtype=torch.float32, device=device)
                states.append(state)
                
                current_legal = info["legal_actions"][agent_index] if info else legal_actions
                action = select_action(agent_net, state, current_legal)
                actions_taken.append(action)

            env_actions = [-1] * len(env.agents)
            for i, agent_index in enumerate(agent_indexes):
                env_actions[agent_index] = actions_taken[i]

            next_states_dict, rewards, terminations, info = env.step(env_actions)
            score -= info["score_change"]
            done = {k: v for k, v in terminations.items() if k in agent_indexes}
            
            episode_reward += rewards[1] + rewards[3]

            team_reward = 0.0
            for i, agent_index in enumerate(agent_indexes):
                agent_reward = rewards[agent_index]
                if exploration_beta > 0:
                    agent_reward += get_exploration_bonus(
                        current_observations[agent_index], 
                        visit_counts, 
                        agent_index, 
                        beta=exploration_beta
                    )
                team_reward += agent_reward
            team_reward /= len(agent_indexes)

            next_states_list = [
                torch.as_tensor(list(next_states_dict.values())[idx], dtype=torch.float32, device=device)
                for idx in agent_indexes
            ]
            
            states_stacked = torch.stack(states)
            next_states_stacked = torch.stack(next_states_list)
            team_done = any(terminations[idx] for idx in agent_indexes)
            
            transition = n_step_buffer.add(
                states_stacked, 
                actions_taken, 
                team_reward, 
                next_states_stacked, 
                team_done
            )
            
            if transition:
                replay_buffer.add(transition)
            
            if replay_buffer.size() >= batch_size:
                has_trained = True
                
                # IMPORTANT: Reset noise before training step
                # This ensures the network trains on a variety of noise conditions
                # rather than overfitting to the specific noise used for the current episode
                agent_net.reset_noise()
                target_net.reset_noise()

                batch, idxs, is_weights = replay_buffer.sample(batch_size, beta=per_beta)
                
                loss, td_errors = compute_td_loss(
                    agent_net, target_net, mixer, target_mixer, 
                    batch, weights=is_weights, gamma=gamma, n_step=n_step
                )
                
                replay_buffer.update_priorities(idxs, td_errors)
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(all_params, max_norm=10.0)
                optimizer.step()
                
                soft_update(agent_net, target_net, tau=tau)
                soft_update(mixer, target_mixer, tau=tau)

        if has_trained:
            scheduler.step()
        
        episode_rewards.append(episode_reward)
        episode_scores.append(score)
        
        if (episode + 1) % 5 == 0:
            avg_reward = np.mean(episode_rewards[-5:])
            avg_score = np.mean(episode_scores[-5:])
            print(f"Ep {episode + 1}/{n_episodes} | Rew: {avg_reward:.2f} | Score: {avg_score:.2f} | LR: {scheduler.get_last_lr()[0]:.6f}")
    
    return episode_rewards, episode_scores


def plot_training_curves(rewards, scores, filename="rainbow_qmix.png", window=20):
    import matplotlib
    matplotlib.use('Agg')
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    def moving_avg(data, w):
        if len(data) < w:
            return data
        return np.convolve(data, np.ones(w)/w, mode='valid')
    
    axes[0].plot(rewards, alpha=0.3, color='blue')
    if len(rewards) >= window:
        axes[0].plot(range(window-1, len(rewards)), moving_avg(rewards, window), 
                     color='blue', linewidth=2, label=f'{window}-ep avg')
    axes[0].set_title('Training Rewards')
    axes[0].set_xlabel('Episode')
    axes[0].legend()
    
    axes[1].plot(scores, alpha=0.3, color='green')
    if len(scores) >= window:
        axes[1].plot(range(window-1, len(scores)), moving_avg(scores, window), 
                     color='green', linewidth=2, label=f'{window}-ep avg')
    axes[1].set_title('Game Scores')
    axes[1].set_xlabel('Episode')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close(fig)
    print(f"Plot saved to {filename}")


def save_models(agent_net, mixer, filename="rainbow_qmix.pt"):
    checkpoint = {
        'agent_net': agent_net.state_dict(), 
        'mixer': mixer.state_dict()
    }
    torch.save(checkpoint, filename)
    print(f"Model saved to {filename}")


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    layout_name = 'tinyCapture.lay'
    layout_path = os.path.join('layouts', layout_name)

    env = gymPacMan_parallel_env(
        layout_file=layout_path,
        display=False,
        reward_forLegalAction=True,  # Set False for harder difficulty
        defenceReward=False,
        length=299,
        enemieName='randomTeam',
        self_play=False,
        random_layout=False
    )
    env.reset()

    n_agents = len(env.agents) // 2
    action_dim = 5
    obs_shape = env.get_Observation(0).shape

    agent_net = AgentQNetwork(obs_shape, action_dim).to(device)
    target_net = AgentQNetwork(obs_shape, action_dim).to(device)
    target_net.load_state_dict(agent_net.state_dict())

    mixer = SimpleQMixer(n_agents=n_agents, state_shape=obs_shape, embed_dim=32).to(device)
    target_mixer = SimpleQMixer(n_agents=n_agents, state_shape=obs_shape, embed_dim=32).to(device)
    target_mixer.load_state_dict(mixer.state_dict())

    replay_buffer = NaivePrioritizedBuffer(capacity=100_000, alpha=0.6)
    
    rewards, scores = train_rainbow_qmix(
        env, agent_net, target_net, mixer, target_mixer, replay_buffer,
        n_episodes=200,
        batch_size=256,
        lr=0.0003, 
        gamma=0.98,  
        exploration_beta=0.3,
        n_step=3,
        tau=0.005
    )

    save_models(agent_net, mixer)
    plot_training_curves(rewards, scores)