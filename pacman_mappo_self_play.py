import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from gymPacMan import gymPacMan_parallel_env
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
NUM_STEPS = 2048
BATCH_SIZE = 512
LR = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
ENT_COEF = 0.01
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5
UPDATE_EPOCHS = 4
TOTAL_UPDATES = 500


class MAPPOAgent(nn.Module):
    def __init__(self, obs_shape, action_dim, num_agents=2):
        super().__init__()
        
        # Shared conv backbone
        self.conv = nn.Sequential(
            nn.Conv2d(obs_shape[0], 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Flatten()
        )
        
        with torch.no_grad():
            flat_dim = self.conv(torch.zeros(1, *obs_shape)).shape[1]
        
        # Actor head (per-agent obs)
        self.actor = nn.Sequential(
            nn.Linear(flat_dim, 256), nn.ReLU(),
            nn.Linear(256, action_dim)
        )
        
        # Critic head (joint obs for centralized value)
        critic_flat = flat_dim * num_agents
        self.critic = nn.Sequential(
            nn.Linear(critic_flat, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )
    
    def get_action_and_value(self, obs, joint_obs):
        """
        obs: (batch, channels, H, W) - individual agent observation
        joint_obs: (batch, num_agents * channels, H, W) - stacked observations
        """
        # Actor
        h = self.conv(obs)
        logits = self.actor(h)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        # Critic - process each agent's obs separately then concat
        batch_size = joint_obs.shape[0]
        num_agents = joint_obs.shape[1] // obs.shape[1]
        ch_per_agent = obs.shape[1]
        
        feats = []
        for i in range(num_agents):
            agent_obs = joint_obs[:, i*ch_per_agent:(i+1)*ch_per_agent]
            feats.append(self.conv(agent_obs))
        joint_feat = torch.cat(feats, dim=1)
        value = self.critic(joint_feat).squeeze(-1)
        
        return action, log_prob, value, dist.entropy()
    
    def get_value(self, joint_obs, ch_per_agent):
        batch_size = joint_obs.shape[0]
        num_agents = joint_obs.shape[1] // ch_per_agent
        
        feats = []
        for i in range(num_agents):
            agent_obs = joint_obs[:, i*ch_per_agent:(i+1)*ch_per_agent]
            feats.append(self.conv(agent_obs))
        joint_feat = torch.cat(feats, dim=1)
        return self.critic(joint_feat).squeeze(-1)
    
    def evaluate(self, obs, joint_obs, action):
        # Actor
        h = self.conv(obs)
        logits = self.actor(h)
        dist = Categorical(logits=logits)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        # Critic
        ch_per_agent = obs.shape[1]
        num_agents = joint_obs.shape[1] // ch_per_agent
        
        feats = []
        for i in range(num_agents):
            agent_obs = joint_obs[:, i*ch_per_agent:(i+1)*ch_per_agent]
            feats.append(self.conv(agent_obs))
        joint_feat = torch.cat(feats, dim=1)
        value = self.critic(joint_feat).squeeze(-1)
        
        return value, log_prob, entropy


def compute_gae(rewards, values, dones, last_value, gamma=GAMMA, lam=GAE_LAMBDA):
    """Standard GAE computation."""
    T = len(rewards)
    advantages = torch.zeros_like(rewards)
    lastgaelam = 0
    
    for t in reversed(range(T)):
        if t == T - 1:
            next_value = last_value
            next_nonterminal = 1.0 - dones[t]
        else:
            next_value = values[t + 1]
            next_nonterminal = 1.0 - dones[t]
        
        delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]
        advantages[t] = lastgaelam = delta + gamma * lam * next_nonterminal * lastgaelam
    
    returns = advantages + values
    return advantages, returns


def train():
    # Environment setup - blue team (agents 1, 3) learns, red team (0, 2) is random
    env = gymPacMan_parallel_env(
        layout_file='layouts/tinyCapture.lay',
        display=False,
        reward_forLegalAction=False,  # Simpler reward
        defenceReward=True,
        length=300,
        enemieName='randomTeam',
        self_play=True
    )
    
    # Our agents are indices 1 and 3 (blue team)
    learner_ids = [1, 3]
    opponent_ids = [0, 2]
    
    obs_shape = env.get_Observation(0).shape
    num_agents = len(learner_ids)
    joint_channels = obs_shape[0] * num_agents
    
    agent = MAPPOAgent(obs_shape, 5, num_agents).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=LR)
    
    # Logging
    all_returns = []
    
    for update in range(1, TOTAL_UPDATES + 1):
        # Storage
        obs_buf = torch.zeros(NUM_STEPS, num_agents, *obs_shape)
        joint_obs_buf = torch.zeros(NUM_STEPS, num_agents, joint_channels, obs_shape[1], obs_shape[2])
        action_buf = torch.zeros(NUM_STEPS, num_agents, dtype=torch.long)
        logprob_buf = torch.zeros(NUM_STEPS, num_agents)
        reward_buf = torch.zeros(NUM_STEPS, num_agents)
        done_buf = torch.zeros(NUM_STEPS, num_agents)
        value_buf = torch.zeros(NUM_STEPS, num_agents)
        
        # Reset
        obs_dict, info = env.reset()
        episode_return = 0
        episode_returns = []
        
        for step in range(NUM_STEPS):
            # Get observations for our agents
            obs_list = [obs_dict[env.agents[i]].float() for i in learner_ids]
            obs = torch.stack(obs_list)  # (num_agents, C, H, W)
            
            # Joint observation (same for all agents in CTDE)
            joint_obs = torch.cat(obs_list, dim=0)  # (num_agents*C, H, W)
            joint_obs_batch = joint_obs.unsqueeze(0).expand(num_agents, -1, -1, -1)
            
            # Store
            obs_buf[step] = obs
            joint_obs_buf[step] = joint_obs_batch
            
            # Get actions
            with torch.no_grad():
                actions, log_probs, values, _ = agent.get_action_and_value(
                    obs.to(device), 
                    joint_obs_batch.to(device)
                )
                actions = actions.cpu()
                log_probs = log_probs.cpu()
                values = values.cpu()
            
            action_buf[step] = actions
            logprob_buf[step] = log_probs
            value_buf[step] = values
            
            # Build action dict for environment
            env_actions = {}
            for i, agent_id in enumerate(learner_ids):
                env_actions[env.agents[agent_id]] = actions[i].item()
            
            # Random actions for opponents
            for agent_id in opponent_ids:
                legal = info['legal_actions'][env.agents[agent_id]]
                env_actions[env.agents[agent_id]] = np.random.choice(legal)
            
            # Step environment
            next_obs_dict, rewards, dones, info = env.step(env_actions)
            
            # Get team reward (sum of individual rewards)
            team_reward = sum(rewards[env.agents[i]] for i in learner_ids)
            episode_return += team_reward
            
            # Store rewards (same team reward for both agents)
            for i in range(num_agents):
                reward_buf[step, i] = team_reward
            
            # Check done
            done = any(dones.values())
            done_buf[step] = float(done)
            
            if done:
                episode_returns.append(episode_return)
                episode_return = 0
                next_obs_dict, info = env.reset()
            
            obs_dict = next_obs_dict
        
        # Compute last value for GAE
        obs_list = [obs_dict[env.agents[i]].float() for i in learner_ids]
        obs = torch.stack(obs_list)
        joint_obs = torch.cat(obs_list, dim=0)
        joint_obs_batch = joint_obs.unsqueeze(0).expand(num_agents, -1, -1, -1)
        
        with torch.no_grad():
            last_value = agent.get_value(joint_obs_batch.to(device), obs_shape[0]).cpu()
        
        # Compute advantages and returns for each agent
        advantages = torch.zeros_like(reward_buf)
        returns = torch.zeros_like(reward_buf)
        
        for i in range(num_agents):
            adv, ret = compute_gae(
                reward_buf[:, i],
                value_buf[:, i],
                done_buf[:, i],
                last_value[i]
            )
            advantages[:, i] = adv
            returns[:, i] = ret
        
        # Flatten for batch processing
        b_obs = obs_buf.reshape(-1, *obs_shape)
        b_joint = joint_obs_buf.reshape(-1, joint_channels, obs_shape[1], obs_shape[2])
        b_actions = action_buf.reshape(-1)
        b_logprobs = logprob_buf.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        
        # PPO update
        batch_size = NUM_STEPS * num_agents
        indices = np.arange(batch_size)
        
        pg_losses, v_losses, entropies, clip_fracs = [], [], [], []
        
        for epoch in range(UPDATE_EPOCHS):
            np.random.shuffle(indices)
            
            for start in range(0, batch_size, BATCH_SIZE):
                end = start + BATCH_SIZE
                mb_idx = indices[start:end]
                
                # Get current policy outputs
                values, log_probs, entropy = agent.evaluate(
                    b_obs[mb_idx].to(device),
                    b_joint[mb_idx].to(device),
                    b_actions[mb_idx].to(device)
                )
                
                # Normalize advantages
                mb_adv = b_advantages[mb_idx].to(device)
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                
                # Policy loss
                ratio = (log_probs - b_logprobs[mb_idx].to(device)).exp()
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                
                # Value loss
                v_loss = 0.5 * ((values - b_returns[mb_idx].to(device)) ** 2).mean()
                
                # Entropy bonus
                ent_loss = entropy.mean()
                
                # Total loss
                loss = pg_loss + VF_COEF * v_loss - ENT_COEF * ent_loss
                
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), MAX_GRAD_NORM)
                optimizer.step()
                
                # Logging
                pg_losses.append(pg_loss.item())
                v_losses.append(v_loss.item())
                entropies.append(ent_loss.item())
                clip_fracs.append(((ratio - 1).abs() > CLIP_EPS).float().mean().item())
        
        # Log progress
        mean_return = np.mean(episode_returns) if episode_returns else 0
        all_returns.append(mean_return)
        
        print(f"Update {update:3d} | "
              f"Return: {mean_return:7.2f} | "
              f"PG Loss: {np.mean(pg_losses):.4f} | "
              f"V Loss: {np.mean(v_losses):.4f} | "
              f"Entropy: {np.mean(entropies):.4f} | "
              f"Clip: {np.mean(clip_fracs):.3f}")
    
    # Save
    torch.save(agent.state_dict(), "mappo_clean.pt")
    
    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(all_returns)
    plt.xlabel("Update")
    plt.ylabel("Episode Return")
    plt.title("MAPPO Training")
    plt.savefig("mappo_curve.png")
    plt.close()
    
    return agent


if __name__ == "__main__":
    train()