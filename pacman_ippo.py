import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from gymPacMan import gymPacMan_parallel_env

# --- CONFIG ---
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


NUM_STEPS = 2048       # Steps to collect per agent before updating
BATCH_SIZE = 64        # Minibatch size for PPO update
LR = 2.5e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
ENT_COEF = 0.01 #entropy penalty
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5
UPDATE_EPOCHS = 10     # How many times to re-use data
TOTAL_UPDATES = 100   # Total training loops

HIDDEN_DIM_SIZE = 256

class ActorCritic(nn.Module):
    def __init__(self, obs_shape, action_dim):
        super(ActorCritic, self).__init__()
        
        # Shared Feature Extractor (CNN)
        self.conv1 = nn.Conv2d(obs_shape[0], 32, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        
        # Calculate flat size
        with torch.no_grad():
            dummy = torch.zeros(1, *obs_shape)
            x = F.relu(self.conv1(dummy))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            self.flat_size = x.view(1, -1).shape[1]

        self.flatten = nn.Flatten()
        
        # Actor Head (Policy)
        self.actor = nn.Sequential(
            nn.Linear(self.flat_size, HIDDEN_DIM_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM_SIZE, action_dim)
        )
        
        # Critic Head (Value)
        self.critic = nn.Sequential(
            nn.Linear(self.flat_size, HIDDEN_DIM_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM_SIZE, 1)
        )

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.flatten(x)
        return x

    def get_action_and_value(self, x, action=None):
        hidden = self.forward(x)
        
        # Actor
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        
        if action is None:
            action = probs.sample()
            
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)

def compute_gae(rewards, values, dones, next_value):
    """ Calculate Generalized Advantage Estimation """
    advantages = torch.zeros_like(rewards)
    lastgaelam = 0
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            nextnonterminal = 1.0 - next_value # Simplified done logic
            nextvalues = next_value
        else:
            nextnonterminal = 1.0 - dones[t+1]
            nextvalues = values[t+1]
            
        delta = rewards[t] + GAMMA * nextvalues * nextnonterminal - values[t]
        advantages[t] = lastgaelam = delta + GAMMA * GAE_LAMBDA * nextnonterminal * lastgaelam
    
    returns = advantages + values
    return advantages, returns

def train_ppo(env):
    agent_ids = [1, 3] # The indices of your team
    obs_shape = env.get_Observation(agent_ids[0]).shape
    action_dim = 5
    
    agent = ActorCritic(obs_shape, action_dim).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=LR, eps=1e-5)

    num_agents = len(agent_ids)
    
    global_step = 0
    
    for update in range(1, TOTAL_UPDATES + 1):
        # 1. Initialize Storage Buffers
        obs_buf = torch.zeros((NUM_STEPS, num_agents) + obs_shape).to(device)
        actions_buf = torch.zeros((NUM_STEPS, num_agents)).to(device)
        logprobs_buf = torch.zeros((NUM_STEPS, num_agents)).to(device)
        rewards_buf = torch.zeros((NUM_STEPS, num_agents)).to(device)
        dones_buf = torch.zeros((NUM_STEPS, num_agents)).to(device)
        values_buf = torch.zeros((NUM_STEPS, num_agents)).to(device)

        env.reset()
        # Initial observation
        obs_list = [torch.tensor(env.get_Observation(i), dtype=torch.float32, device=device) for i in agent_ids]
        next_obs = torch.stack(obs_list) # Shape: (2, C, H, W)
        next_done = torch.zeros(num_agents).to(device)

        # 2. Collect Rollouts
        for step in range(NUM_STEPS):
            global_step += 1
            obs_buf[step] = next_obs
            dones_buf[step] = next_done
            
            with torch.no_grad():
                # Pass both agents through network simultaneously
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values_buf[step] = value.flatten()
            
            actions_buf[step] = action
            logprobs_buf[step] = logprob

            # Execute in Env
            cpu_actions = action.cpu().numpy()
            env_actions = [-1] * len(env.agents)
            for i, ag_id in enumerate(agent_ids):
                env_actions[ag_id] = cpu_actions[i]

            next_obs_dict, rewards_dict, dones_dict, _ = env.step(env_actions)

            # Process Feedback
            # Cooperative Reward: Sum rewards of team and give to both
            team_reward = sum([rewards_dict[i] for i in agent_ids])
            # Normalize reward slightly to help convergence
            team_reward = np.clip(team_reward, -1, 1) 

            rewards_buf[step] = torch.tensor([team_reward] * num_agents, device=device)
            
            # Prepare next step
            obs_list = [torch.tensor(next_obs_dict[i], dtype=torch.float32, device=device) for i in agent_ids]
            next_obs = torch.stack(obs_list)
            
            # Check if ANY of our agents died/finished
            any_done = any([dones_dict[i] for i in agent_ids])
            next_done = torch.tensor([float(any_done)] * num_agents, device=device)
            
            if any_done:
                env.reset()
                obs_list = [torch.tensor(env.get_Observation(i), dtype=torch.float32, device=device) for i in agent_ids]
                next_obs = torch.stack(obs_list)

        # 3. Bootstrap value for last step
        with torch.no_grad():
            _, _, _, next_value = agent.get_action_and_value(next_obs)
            next_value = next_value.flatten()

        # 4. Compute GAE (Advantages)
        advantages, returns = compute_gae(rewards_buf, values_buf, dones_buf, next_value)
        
        # 5. Flatten the collected data
        # We merge time and agent dimensions: (NUM_STEPS, AGENTS, ...) -> (NUM_STEPS * AGENTS, ...)
        b_obs = obs_buf.view(-1, *obs_shape)
        b_logprobs = logprobs_buf.view(-1)
        b_actions = actions_buf.view(-1)
        b_advantages = advantages.view(-1)
        b_returns = returns.view(-1)
        b_values = values_buf.view(-1)

        # 6. PPO Update
        b_inds = np.arange(BATCH_SIZE)
        clipfracs = []
        
        # Flattened dataset size
        dataset_size = NUM_STEPS * num_agents
        indices = np.arange(dataset_size)

        for epoch in range(UPDATE_EPOCHS):
            np.random.shuffle(indices)
            
            for start in range(0, dataset_size, BATCH_SIZE):
                end = start + BATCH_SIZE
                mb_inds = indices[start:end]
                
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # Calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > CLIP_EPS).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                # Normalize advantages (Critical for PPO stability)
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy Loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value Loss (clipped)
                newvalue = newvalue.view(-1)
                v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                v_clipped = b_values[mb_inds] + torch.clamp(
                    newvalue - b_values[mb_inds],
                    -CLIP_EPS,
                    CLIP_EPS,
                )
                v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()

                loss = pg_loss - ENT_COEF * entropy.mean() + v_loss * VF_COEF

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), MAX_GRAD_NORM)
                optimizer.step()

        print(f"Update {update}/{TOTAL_UPDATES} | Loss: {loss.item():.3f} | Mean Reward: {rewards_buf.sum().item() / num_agents:.2f}")

    torch.save(agent.state_dict(), "ppo_pacman.pt")

if __name__ == "__main__":
    layout_name = 'tinyCapture.lay'
    layout_path = os.path.join('layouts', layout_name)
    env = gymPacMan_parallel_env(
        layout_file=layout_path, display=False, reward_forLegalAction=True,
        defenceReward=False, length=1000, enemieName='randomTeam',
        self_play=False, random_layout=False
    )
    
    train_ppo(env)