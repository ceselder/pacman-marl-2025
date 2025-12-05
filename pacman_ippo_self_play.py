import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from gymPacMan import gymPacMan_parallel_env
from collections import deque
import copy
import matplotlib as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_STEPS = 1024       # Steps to collect per agent before updating
BATCH_SIZE = 128        # Minibatch size for PPO update
LR = 2.5e-4
GAMMA = 0.98
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
ENT_COEF = 0.01 #entropy penalty
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5
UPDATE_EPOCHS = 10     # How many times to re-use data
TOTAL_UPDATES = 300   # Total training loops

HIDDEN_DIM = 512 #512 seems to work ok

OPPONENT_UPDATE_FREQ = 10  # Update opponent pool every N updates
OPPONENT_POOL_SIZE = 5     # Keep last N versions
SELF_PLAY_PROB = 0.5       # Prob of playing vs past self (vs latest)

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
        self.actor = nn.Sequential(nn.Linear(flat_size, HIDDEN_DIM), nn.ReLU(), nn.Linear(HIDDEN_DIM, action_dim))
        self.critic = nn.Sequential(nn.Linear(flat_size, HIDDEN_DIM), nn.ReLU(), nn.Linear(HIDDEN_DIM, 1))

    def get_action_and_value(self, x, action=None):
        hidden = self.conv(x)
        logits = self.actor(hidden)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), self.critic(hidden).squeeze(-1)
    
    def get_action(self, x):
        """Just get action, no grad tracking - for opponent."""
        with torch.no_grad():
            hidden = self.conv(x)
            logits = self.actor(hidden)
            return Categorical(logits=logits).sample()

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

def train_ppo_selfplay(env):
    # We train blue team (1, 3), opponent is red team (0, 2)
    train_ids = [1, 3]  # Our agents
    opp_ids = [0, 2]    # Opponent agents
    num_agents = len(train_ids)
    obs_shape = env.get_Observation(train_ids[0]).shape
    
    # Main agent (trains)
    agent = ActorCritic(obs_shape, 5).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=LR, eps=1e-5)
    
    # Opponent pool - past versions of our agent
    opponent_pool = deque(maxlen=OPPONENT_POOL_SIZE)
    opponent_pool.append(copy.deepcopy(agent.state_dict()))
    
    def get_opponent():
        """Sample an opponent from the pool."""
        opp = ActorCritic(obs_shape, 5).to(device)
        if np.random.random() < SELF_PLAY_PROB and len(opponent_pool) > 1:
            # Play against random past version
            opp.load_state_dict(np.random.choice(list(opponent_pool)[:-1]))
        else:
            # Play against latest
            opp.load_state_dict(opponent_pool[-1])
        opp.eval()
        return opp

    for update in range(1, TOTAL_UPDATES + 1):
        # Get opponent for this rollout
        opponent = get_opponent()
        
        # Storage for our team only
        obs_buf = torch.zeros((NUM_STEPS, num_agents, *obs_shape))
        actions_buf = torch.zeros((NUM_STEPS, num_agents), dtype=torch.long)
        logprobs_buf = torch.zeros((NUM_STEPS, num_agents))
        rewards_buf = torch.zeros((NUM_STEPS, num_agents))
        dones_buf = torch.zeros((NUM_STEPS, num_agents))
        values_buf = torch.zeros((NUM_STEPS, num_agents))

        env.reset()
        next_obs = torch.stack([env.get_Observation(i) for i in train_ids]).float()
        next_done = torch.zeros(num_agents)

        for step in range(NUM_STEPS):
            obs_buf[step] = next_obs
            dones_buf[step] = next_done
            
            # Our team's actions (with grad tracking for PPO)
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs.to(device))
                action, logprob, value = action.cpu(), logprob.cpu(), value.cpu()
            
            actions_buf[step], logprobs_buf[step], values_buf[step] = action, logprob, value

            # Opponent's actions
            opp_obs = torch.stack([env.get_Observation(i) for i in opp_ids]).float().to(device)
            opp_actions = opponent.get_action(opp_obs).cpu()

            # Build full action dict
            # env.step expects actions indexed by agent object, not int
            env_actions = {}
            for i, ag_id in enumerate(train_ids):
                env_actions[env.agents[ag_id]] = action[i].item()
            for i, ag_id in enumerate(opp_ids):
                env_actions[env.agents[ag_id]] = opp_actions[i].item()

            next_obs_dict, rewards_dict, dones_dict, _ = env.step(env_actions)
            
            # Reward for our team
            team_reward = sum(rewards_dict[env.agents[i]] for i in train_ids)
            rewards_buf[step] = team_reward  # Shared team reward
            
            any_done = any(dones_dict[env.agents[i]] for i in train_ids)
            next_done = torch.full((num_agents,), float(any_done))
            
            if any_done:
                env.reset()
                next_obs = torch.stack([env.get_Observation(i) for i in train_ids]).float()
            else:
                next_obs = torch.stack([next_obs_dict[env.agents[i]] for i in train_ids]).float()

        # Bootstrap
        with torch.no_grad():
            _, _, _, last_value = agent.get_action_and_value(next_obs.to(device))
            last_value = last_value.cpu()

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
                mb_adv = (b_advantages[mb] - b_advantages[mb].mean()) / (b_advantages[mb].std() + 1e-8)
                mb_adv = mb_adv.to(device)
                mb_returns = b_returns[mb].to(device)
                mb_old_logprobs = b_logprobs[mb].to(device)

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(mb_obs, mb_actions)
                ratio = (newlogprob - mb_old_logprobs).exp()

                pg_loss = -torch.min(
                    ratio * mb_adv,
                    torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * mb_adv
                ).mean()
                v_loss = 0.5 * ((newvalue - mb_returns) ** 2).mean()
                loss = pg_loss - ENT_COEF * entropy.mean() + VF_COEF * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), MAX_GRAD_NORM)
                optimizer.step()

        # Update opponent pool periodically
        if update % OPPONENT_UPDATE_FREQ == 0:
            opponent_pool.append(copy.deepcopy(agent.state_dict()))
            print(f"  -> Added to opponent pool (size: {len(opponent_pool)})")

        print(f"Update {update}/{TOTAL_UPDATES} | Reward: {rewards_buf.sum().item() / num_agents:.2f}")

    torch.save(agent.state_dict(), "ppo_pacman_selfplay.pt")

def plot_training(log):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    
    axes[0,0].plot(log['update'], log['reward'])
    axes[0,0].set_title('Rollout Reward')
    axes[0,0].set_xlabel('Update')
    
    axes[0,1].plot(log['update'], log['episode_return'])
    axes[0,1].set_title('Episode Return (10-ep avg)')
    axes[0,1].set_xlabel('Update')
    
    axes[0,2].plot(log['update'], log['entropy'])
    axes[0,2].set_title('Entropy')
    axes[0,2].axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='danger zone')
    axes[0,2].set_xlabel('Update')
    
    axes[1,0].plot(log['update'], log['pg_loss'])
    axes[1,0].set_title('Policy Loss')
    axes[1,0].set_xlabel('Update')
    
    axes[1,1].plot(log['update'], log['v_loss'])
    axes[1,1].set_title('Value Loss')
    axes[1,1].set_xlabel('Update')
    
    axes[1,2].plot(log['update'], log['clip_frac'])
    axes[1,2].set_title('Clip Fraction')
    axes[1,2].axhline(y=0.2, color='r', linestyle='--', alpha=0.5, label='too high')
    axes[1,2].set_xlabel('Update')
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150)

if __name__ == "__main__":
    env = gymPacMan_parallel_env(
        layout_file=os.path.join('layouts', 'tinyCapture.lay'),
        display=False, reward_forLegalAction=True, defenceReward=True,
        length=300, enemieName='randomTeam',
        self_play=True,  # <-- Enable self-play mode
        random_layout=False
    )
    log = train_ppo_selfplay(env)
    plot_training(log)