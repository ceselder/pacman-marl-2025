import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from gymPacMan import gymPacMan_parallel_env

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_STEPS = 2048       # Steps to collect per agent before updating
BATCH_SIZE = 32        # Minibatch size for PPO update
LR = 2.5e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
ENT_COEF = 0.01 #entropy penalty
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5
UPDATE_EPOCHS = 10     # How many times to re-use data
TOTAL_UPDATES = 100   # Total training loops

HIDDEN_DIM_SIZE = 128



class ActorCritic(nn.Module):
    def __init__(self, obs_shape, action_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(obs_shape[0], 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Flatten()
        )
        with torch.no_grad():
            flat_size = self.conv(torch.zeros(1, *obs_shape)).shape[1]
        self.actor = nn.Sequential(nn.Linear(flat_size, HIDDEN_DIM_SIZE), nn.ReLU(), nn.Linear(HIDDEN_DIM_SIZE, action_dim))
        self.critic = nn.Sequential(nn.Linear(flat_size, HIDDEN_DIM_SIZE), nn.ReLU(), nn.Linear(HIDDEN_DIM_SIZE, 1))

    def get_action_and_value(self, x, action=None):
        hidden = self.conv(x)
        logits = self.actor(hidden)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), self.critic(hidden).squeeze(-1)

def compute_gae(rewards, values, dones, last_value, last_done):
    T = len(rewards)
    advantages = torch.zeros_like(rewards)
    lastgaelam = 0
    for t in reversed(range(T)):
        if t == T - 1:
            next_non_terminal = 1.0 - last_done
            next_values = last_value
        else:
            next_non_terminal = 1.0 - dones[t + 1]
            next_values = values[t + 1]
        delta = rewards[t] + GAMMA * next_values * next_non_terminal - values[t]
        advantages[t] = lastgaelam = delta + GAMMA * GAE_LAMBDA * next_non_terminal * lastgaelam
    return advantages, advantages + values

def train_ppo(env):
    agent_ids = [1, 3]
    num_agents = len(agent_ids)
    obs_shape = env.get_Observation(agent_ids[0]).shape
    
    agent = ActorCritic(obs_shape, 5).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=LR, eps=1e-5)

    for update in range(1, TOTAL_UPDATES + 1):
        # Storage - keep on CPU if CPU-bound, move batches to GPU
        obs_buf = torch.zeros((NUM_STEPS, num_agents, *obs_shape))
        actions_buf = torch.zeros((NUM_STEPS, num_agents), dtype=torch.long)
        logprobs_buf = torch.zeros((NUM_STEPS, num_agents))
        rewards_buf = torch.zeros((NUM_STEPS, num_agents))
        dones_buf = torch.zeros((NUM_STEPS, num_agents))
        values_buf = torch.zeros((NUM_STEPS, num_agents))

        env.reset()
        next_obs = torch.tensor(np.stack([env.get_Observation(i) for i in agent_ids]), dtype=torch.float32)
        next_done = torch.zeros(num_agents)

        # Rollout
        for step in range(NUM_STEPS):
            obs_buf[step] = next_obs
            dones_buf[step] = next_done
            
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs.to(device))
                action, logprob, value = action.cpu(), logprob.cpu(), value.cpu()
            
            actions_buf[step], logprobs_buf[step], values_buf[step] = action, logprob, value

            env_actions = [-1] * len(env.agents)
            for i, ag_id in enumerate(agent_ids):
                env_actions[ag_id] = action[i].item()

            next_obs_dict, rewards_dict, dones_dict, _ = env.step(env_actions)
            
            team_reward = np.clip(sum(rewards_dict[i] for i in agent_ids), -1, 1)
            rewards_buf[step] = team_reward
            
            any_done = any(dones_dict[i] for i in agent_ids)
            next_done = torch.full((num_agents,), float(any_done))
            
            if any_done:
                env.reset()
                next_obs = torch.tensor(np.stack([env.get_Observation(i) for i in agent_ids]), dtype=torch.float32)
            else:
                next_obs = torch.tensor(np.stack([next_obs_dict[i] for i in agent_ids]), dtype=torch.float32)

        # Bootstrap
        with torch.no_grad():
            _, _, _, last_value = agent.get_action_and_value(next_obs.to(device))
            last_value = last_value.cpu()

        # GAE
        advantages, returns = compute_gae(rewards_buf, values_buf, dones_buf, last_value, next_done)

        # Flatten
        b_obs = obs_buf.reshape(-1, *obs_shape)
        b_actions = actions_buf.reshape(-1)
        b_logprobs = logprobs_buf.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        
        dataset_size = NUM_STEPS * num_agents
        indices = np.arange(dataset_size)

        # PPO update
        for _ in range(UPDATE_EPOCHS):
            np.random.shuffle(indices)
            for start in range(0, dataset_size, BATCH_SIZE):
                mb = indices[start:start + BATCH_SIZE]
                
                mb_obs = b_obs[mb].to(device)
                mb_actions = b_actions[mb].to(device)
                mb_old_logprobs = b_logprobs[mb].to(device)
                mb_advantages = b_advantages[mb].to(device)
                mb_returns = b_returns[mb].to(device)
                
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(mb_obs, mb_actions)
                ratio = (newlogprob - mb_old_logprobs).exp()

                pg_loss = -torch.min(
                    ratio * mb_advantages,
                    torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * mb_advantages
                ).mean()
                v_loss = 0.5 * ((newvalue - mb_returns) ** 2).mean()
                loss = pg_loss - ENT_COEF * entropy.mean() + VF_COEF * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), MAX_GRAD_NORM)
                optimizer.step()

        print(f"Update {update}/{TOTAL_UPDATES} | Reward: {rewards_buf.sum().item() / num_agents:.2f}")

    torch.save(agent.state_dict(), "ppo_pacman.pt")

if __name__ == "__main__":
    env = gymPacMan_parallel_env(
        layout_file=os.path.join('layouts', 'tinyCapture.lay'),
        display=False, reward_forLegalAction=True, defenceReward=False,
        length=1000, enemieName='randomTeam', self_play=False, random_layout=False
    )
    train_ppo(env)