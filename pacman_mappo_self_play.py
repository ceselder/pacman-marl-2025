import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from gymPacMan import gymPacMan_parallel_env
from collections import deque
import copy
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters
NUM_STEPS = 2048
BATCH_SIZE = 512
LR = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
ENT_COEF = 0.015
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5
UPDATE_EPOCHS = 4
TOTAL_UPDATES = 200

# Self-play settings
OPPONENT_POOL_SIZE = 5
OPPONENT_UPDATE_FREQ = 10


def canonicalize_obs(obs, is_red_agent):
    """
    Convert observation to canonical "blue perspective".
    
    For blue agents (1, 3): observation is already correct
    For red agents (0, 2): need to flip horizontally and swap channels
    
    Channel layout (from gymPacMan.get_Observation):
    0: walls
    1: agent location (value = 1 + numCarrying)
    2: blue capsules  
    3: red capsules
    4: allies locations
    5: enemies locations
    6: blue food
    7: red food
    
    For red team, after canonicalization:
    - Flip everything horizontally
    - Swap blue/red capsules (2 <-> 3)
    - Swap blue/red food (6 <-> 7)
    - Allies/enemies stay the same (already relative to agent)
    """
    if not is_red_agent:
        return obs
    
    # Clone to avoid modifying original
    canon = obs.clone()
    
    # Horizontal flip
    canon = torch.flip(canon, dims=[-1])
    
    # Swap capsules: blue (2) <-> red (3)
    temp = canon[..., 2, :, :].clone()
    canon[..., 2, :, :] = canon[..., 3, :, :]
    canon[..., 3, :, :] = temp
    
    # Swap food: blue (6) <-> red (7)
    temp = canon[..., 6, :, :].clone()
    canon[..., 6, :, :] = canon[..., 7, :, :]
    canon[..., 7, :, :] = temp
    
    return canon


def canonicalize_action(action, is_red_agent):
    """
    Convert action from canonical space to environment space.
    
    Actions: 0=North, 1=East, 2=South, 3=West, 4=Stop
    
    For red team (flipped view), East and West are swapped.
    """
    if not is_red_agent:
        return action
    
    # Swap East (1) <-> West (3)
    if isinstance(action, torch.Tensor):
        mapper = torch.tensor([0, 3, 2, 1, 4], device=action.device)
        return mapper[action]
    else:
        mapper = {0: 0, 1: 3, 2: 2, 3: 1, 4: 4}
        return mapper[action]


class MAPPOAgent(nn.Module):
    def __init__(self, obs_shape, action_dim, num_agents=2):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(obs_shape[0], 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Flatten()
        )
        
        with torch.no_grad():
            flat_dim = self.conv(torch.zeros(1, *obs_shape)).shape[1]
        
        self.actor = nn.Sequential(
            nn.Linear(flat_dim, 256), nn.ReLU(),
            nn.Linear(256, action_dim)
        )
        
        # Centralized critic sees all agents' observations
        self.critic = nn.Sequential(
            nn.Linear(flat_dim * num_agents, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        self.flat_dim = flat_dim
    
    def get_action_and_value(self, obs, all_obs_list):
        """
        obs: (batch, C, H, W) - this agent's observation
        all_obs_list: list of (batch, C, H, W) - all agents' observations
        """
        # Actor
        h = self.conv(obs)
        logits = self.actor(h)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        # Critic - concat all agent features
        all_feats = [self.conv(o) for o in all_obs_list]
        joint_feat = torch.cat(all_feats, dim=1)
        value = self.critic(joint_feat).squeeze(-1)
        
        return action, log_prob, value, dist.entropy()
    
    def get_value(self, all_obs_list):
        all_feats = [self.conv(o) for o in all_obs_list]
        joint_feat = torch.cat(all_feats, dim=1)
        return self.critic(joint_feat).squeeze(-1)
    
    def evaluate(self, obs, all_obs_flat, action, num_agents, obs_shape):
        """
        obs: (batch, C, H, W)
        all_obs_flat: (batch, num_agents*C, H, W)
        """
        # Actor
        h = self.conv(obs)
        logits = self.actor(h)
        dist = Categorical(logits=logits)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        # Critic - need to split and process
        C = obs_shape[0]
        all_feats = []
        for i in range(num_agents):
            agent_obs = all_obs_flat[:, i*C:(i+1)*C, :, :]
            all_feats.append(self.conv(agent_obs))
        joint_feat = torch.cat(all_feats, dim=1)
        value = self.critic(joint_feat).squeeze(-1)
        
        return value, log_prob, entropy


def compute_gae(rewards, values, dones, last_value, gamma=GAMMA, lam=GAE_LAMBDA):
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
    env = gymPacMan_parallel_env(
        layout_file='layouts/tinyCapture.lay',
        display=False,
        reward_forLegalAction=False,
        defenceReward=True,
        length=300,
        enemieName='randomTeam',
        self_play=True
    )
    
    obs_shape = env.get_Observation(0).shape
    num_agents = 2
    
    agent = MAPPOAgent(obs_shape, 5, num_agents).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=LR)
    
    # Opponent pool for self-play
    opponent_pool = deque(maxlen=OPPONENT_POOL_SIZE)
    opponent_pool.append(copy.deepcopy(agent.state_dict()))
    
    all_returns = []
    
    for update in range(1, TOTAL_UPDATES + 1):
        # Randomly choose which side we play
        play_as_red = np.random.rand() > 0.5
        
        if play_as_red:
            learner_ids = [0, 2]  # Red team
            opponent_ids = [1, 3]  # Blue team
        else:
            learner_ids = [1, 3]  # Blue team
            opponent_ids = [0, 2]  # Red team
        
        # Load opponent from pool
        opponent = MAPPOAgent(obs_shape, 5, num_agents).to(device)
        opp_weights = opponent_pool[np.random.randint(len(opponent_pool))]
        opponent.load_state_dict(opp_weights)
        opponent.eval()
        
        # Storage - per agent
        obs_buf = torch.zeros(NUM_STEPS, num_agents, *obs_shape)
        joint_obs_buf = torch.zeros(NUM_STEPS, num_agents, obs_shape[0] * num_agents, obs_shape[1], obs_shape[2])
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
            # Get and canonicalize observations for learner agents
            learner_obs_raw = [obs_dict[env.agents[i]].float() for i in learner_ids]
            learner_obs_canon = [canonicalize_obs(o.unsqueeze(0), play_as_red).squeeze(0) for o in learner_obs_raw]
            learner_obs = torch.stack(learner_obs_canon)  # (2, C, H, W)
            
            # Joint observation for critic
            joint_obs = torch.cat(learner_obs_canon, dim=0)  # (2*C, H, W)
            joint_obs_batch = joint_obs.unsqueeze(0).expand(num_agents, -1, -1, -1)
            
            # Store
            obs_buf[step] = learner_obs
            joint_obs_buf[step] = joint_obs_batch
            
            # Get actions for learner agents
            with torch.no_grad():
                all_obs_list = [learner_obs[i:i+1].to(device) for i in range(num_agents)]
                
                actions = []
                log_probs = []
                values = []
                
                for i in range(num_agents):
                    act, lp, val, _ = agent.get_action_and_value(
                        learner_obs[i:i+1].to(device),
                        all_obs_list
                    )
                    actions.append(act)
                    log_probs.append(lp)
                    values.append(val)
                
                actions = torch.cat(actions)
                log_probs = torch.cat(log_probs)
                values = torch.cat(values)
            
            action_buf[step] = actions.cpu()
            logprob_buf[step] = log_probs.cpu()
            value_buf[step] = values.cpu()
            
            # Get actions for opponent agents (also canonicalized)
            opp_obs_raw = [obs_dict[env.agents[i]].float() for i in opponent_ids]
            opp_obs_canon = [canonicalize_obs(o.unsqueeze(0), not play_as_red).squeeze(0) for o in opp_obs_raw]
            opp_obs = torch.stack(opp_obs_canon)
            
            with torch.no_grad():
                opp_all_obs_list = [opp_obs[i:i+1].to(device) for i in range(num_agents)]
                opp_actions = []
                for i in range(num_agents):
                    act, _, _, _ = opponent.get_action_and_value(
                        opp_obs[i:i+1].to(device),
                        opp_all_obs_list
                    )
                    opp_actions.append(act)
                opp_actions = torch.cat(opp_actions)
            
            # Build action dict - convert canonical actions to env actions
            env_actions = {}
            for i, agent_id in enumerate(learner_ids):
                canon_act = actions[i].item()
                env_act = canonicalize_action(canon_act, play_as_red)
                env_actions[env.agents[agent_id]] = env_act
            
            for i, agent_id in enumerate(opponent_ids):
                canon_act = opp_actions[i].item()
                env_act = canonicalize_action(canon_act, not play_as_red)
                env_actions[env.agents[agent_id]] = env_act
            
            # Step
            next_obs_dict, rewards, dones, info = env.step(env_actions)
            
            # Team reward
            team_reward = sum(rewards[env.agents[i]] for i in learner_ids)
            episode_return += team_reward
            
            for i in range(num_agents):
                reward_buf[step, i] = team_reward
            
            done = any(dones.values())
            done_buf[step] = float(done)
            
            if done:
                episode_returns.append(episode_return)
                episode_return = 0
                next_obs_dict, info = env.reset()
            
            obs_dict = next_obs_dict
        
        # Compute last value
        learner_obs_raw = [obs_dict[env.agents[i]].float() for i in learner_ids]
        learner_obs_canon = [canonicalize_obs(o.unsqueeze(0), play_as_red).squeeze(0) for o in learner_obs_raw]
        learner_obs = torch.stack(learner_obs_canon)
        
        with torch.no_grad():
            all_obs_list = [learner_obs[i:i+1].to(device) for i in range(num_agents)]
            last_value = agent.get_value(all_obs_list).cpu()
        
        # GAE
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
        
        # Flatten
        b_obs = obs_buf.reshape(-1, *obs_shape)
        b_joint = joint_obs_buf.reshape(-1, obs_shape[0] * num_agents, obs_shape[1], obs_shape[2])
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
                
                values, log_probs, entropy = agent.evaluate(
                    b_obs[mb_idx].to(device),
                    b_joint[mb_idx].to(device),
                    b_actions[mb_idx].to(device),
                    num_agents,
                    obs_shape
                )
                
                mb_adv = b_advantages[mb_idx].to(device)
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                
                ratio = (log_probs - b_logprobs[mb_idx].to(device)).exp()
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                
                v_loss = 0.5 * ((values - b_returns[mb_idx].to(device)) ** 2).mean()
                ent_loss = entropy.mean()
                
                loss = pg_loss + VF_COEF * v_loss - ENT_COEF * ent_loss
                
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), MAX_GRAD_NORM)
                optimizer.step()
                
                pg_losses.append(pg_loss.item())
                v_losses.append(v_loss.item())
                entropies.append(ent_loss.item())
                clip_fracs.append(((ratio - 1).abs() > CLIP_EPS).float().mean().item())
        
        # Update opponent pool
        if update % OPPONENT_UPDATE_FREQ == 0:
            opponent_pool.append(copy.deepcopy(agent.state_dict()))
        
        # Logging
        mean_return = np.mean(episode_returns) if episode_returns else 0
        all_returns.append(mean_return)
        
        side = "Red" if play_as_red else "Blue"
        print(f"Update {update:4d} [{side}] | "
              f"Return: {mean_return:7.2f} | "
              f"PG: {np.mean(pg_losses):.4f} | "
              f"VL: {np.mean(v_losses):.4f} | "
              f"Ent: {np.mean(entropies):.3f} | "
              f"Clip: {np.mean(clip_fracs):.3f}")
        
        # Save periodically
        if update % 100 == 0:
            torch.save(agent.state_dict(), f"mappo_selfplay_{update}.pt")
    
    torch.save(agent.state_dict(), "mappo_selfplay_final.pt")
    
    plt.figure(figsize=(10, 5))
    plt.plot(all_returns)
    plt.xlabel("Update")
    plt.ylabel("Episode Return")
    plt.title("MAPPO Self-Play Training")
    plt.savefig("mappo_selfplay_curve.png")
    plt.close()
    
    return agent


if __name__ == "__main__":
    train()