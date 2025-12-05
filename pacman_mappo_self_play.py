import os
import copy
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from collections import deque
import matplotlib.pyplot as plt

# Ensure this matches your file structure
from gymPacMan import gymPacMan_parallel_env

# --- CONFIGURATION ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Checkpoint Settings
LOAD_CHECKPOINT_PATH = "mappo_100.pt"  # Path to file, or None to start fresh
# LOAD_CHECKPOINT_PATH = None 

# Hyperparameters
NUM_STEPS = 2048
BATCH_SIZE = 512            # Increased slightly for stability
LR = 2.5e-4                 # Standard PPO LR
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5
UPDATE_EPOCHS = 4
TOTAL_UPDATES = 2000        # Total updates to run

# Entropy Settings
ENT_START = 0.01            # Start with some exploration
ENT_END = 0.01              # Decay to low exploration
ENT_DECAY_UPDATES = 1000    # How many updates to decay over

# Rewards
LIVING_PENALTY = -0.01      # Penalty for every step (forces movement)
SHAPING_SCALE = 0.1         # Scale for heuristic distance reward

# Self-Play Settings
OPPONENT_POOL_SIZE = 5
OPPONENT_UPDATE_FREQ = 20
EVAL_FREQ = 25
EVAL_EPISODES = 5

# --- HELPER FUNCTIONS ---

def canonicalize_obs(obs, is_red_agent):
    """Canonicalize observation so red team sees the world as if they were blue."""
    if not is_red_agent:
        return obs
    
    is_batched = obs.dim() == 4
    if not is_batched:
        obs = obs.unsqueeze(0)
    
    # 1. Flip Spatial (180 degrees)
    canon = torch.flip(obs.clone(), dims=[2, 3])
    
    # 2. Swap Channels (Friend <-> Enemy)
    # Indices: 2=BlueFood, 3=RedFood, 6=BlueCapsule, 7=RedCapsule
    canon[:, [2, 3], :, :] = canon[:, [3, 2], :, :] 
    canon[:, [6, 7], :, :] = canon[:, [7, 6], :, :] 
    
    if not is_batched:
        canon = canon.squeeze(0)
    return canon

def canonicalize_action(action, is_red_agent):
    """
    Maps Agent Action -> Environment Action
    Blue: Identity
    Red: Swap East(1) <-> West(3)
    """
    if not is_red_agent:
        return action
    
    # Mapper: 0->0, 1->3, 2->2, 3->1, 4->4
    if isinstance(action, torch.Tensor):
        mapper = torch.tensor([0, 3, 2, 1, 4], device=action.device)
        return mapper[action]
    return {0: 0, 1: 3, 2: 2, 3: 1, 4: 4}[action]

def get_action_masks(info, agent_ids, env_agents_list, is_red_team):
    """
    Creates binary masks (1=Legal, 0=Illegal).
    Handles the coordinate flipping for Red team.
    """
    num_agents = len(agent_ids)
    masks = torch.zeros(num_agents, 5) 
    
    # Mapping for Red: Env East(1) is Agent West(3)
    # Env Map: 0:N, 1:E, 2:S, 3:W, 4:Stop
    # If Env says 1 is legal, Red Agent sees it as 3 being legal.
    red_env_to_agent = {0: 0, 1: 3, 2: 2, 3: 1, 4: 4}

    for i, aid in enumerate(agent_ids):
        agent_name = env_agents_list[aid]
        # GymPacman info usually has 'legal_actions' as a list of indices
        legal_actions = info['legal_actions'][agent_name]
        
        if not is_red_team:
            for act in legal_actions:
                masks[i, act] = 1.0
        else:
            for act in legal_actions:
                agent_act = red_env_to_agent.get(act, act)
                masks[i, agent_act] = 1.0
                
    return masks

def compute_heuristic_shaping(obs_curr, obs_next):
    """Calculates distance to nearest food."""
    # Simplified extraction of agent pos and food pos from channels
    # Channel 0 is usually walls, 1 is agents.
    # This is a basic implementation; assumes standard layout channels.
    
    def get_pos(obs):
        # Find agent position (Channel 1)
        # obs is (C, H, W)
        agent_locs = (obs[1] > 0).nonzero(as_tuple=False)
        if len(agent_locs) == 0: return None
        return agent_locs[0].float() # (y, x)

    pos_curr = get_pos(obs_curr)
    pos_next = get_pos(obs_next)
    
    if pos_curr is None or pos_next is None:
        return 0.0

    # Food Channel is 7 (Red Food) because we canonicalized to Blue perspective
    # (Blue eats Red food).
    food_locs = (obs_curr[7] > 0).nonzero(as_tuple=False).float()
    
    if len(food_locs) == 0: return 0.0

    # Distance to nearest food
    dist_curr = (food_locs - pos_curr).abs().sum(dim=1).min()
    dist_next = (food_locs - pos_next).abs().sum(dim=1).min()
    
    # Positive reward if distance decreased
    return (dist_curr - dist_next).item()

# --- AGENT CLASS ---

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
            nn.Linear(flat_dim, 512), nn.ReLU(),
            nn.Linear(512, 512),nn.ReLU(),
            nn.Linear(512, action_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(flat_dim * num_agents, 1024), nn.ReLU(),
            nn.Linear(1024, 1024),nn.ReLU(),
            nn.Linear(1024, 1)
        )

    def get_action_and_value(self, obs, all_obs_list, action_masks=None):
        h = self.conv(obs)
        logits = self.actor(h)
        
        # Action Masking
        if action_masks is not None:
            logits = logits.masked_fill(action_masks == 0, -1e9)

        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        all_feats = [self.conv(o) for o in all_obs_list]
        joint_feat = torch.cat(all_feats, dim=1)
        value = self.critic(joint_feat).squeeze(-1)
        return action, log_prob, value, dist.entropy()

    def get_value(self, all_obs_list):
        all_feats = [self.conv(o) for o in all_obs_list]
        joint_feat = torch.cat(all_feats, dim=1)
        return self.critic(joint_feat).squeeze(-1)

    def evaluate(self, obs, all_obs_flat, action, num_agents, obs_shape, action_masks=None):
        h = self.conv(obs)
        logits = self.actor(h)
        
        if action_masks is not None:
            logits = logits.masked_fill(action_masks == 0, -1e9)

        dist = Categorical(logits=logits)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        C = obs_shape[0]
        all_feats = []
        for i in range(num_agents):
            agent_obs = all_obs_flat[:, i*C:(i+1)*C, :, :]
            all_feats.append(self.conv(agent_obs))
        
        joint_feat = torch.cat(all_feats, dim=1)
        value = self.critic(joint_feat).squeeze(-1)
        return value, log_prob, entropy
    
    def get_deterministic_action(self, obs, action_masks=None):
        with torch.no_grad():
            h = self.conv(obs)
            logits = self.actor(h)
            if action_masks is not None:
                logits = logits.masked_fill(action_masks == 0, -1e9)
            return logits.argmax(dim=-1)

# --- TRAINING LOOP ---

def compute_gae(rewards, values, dones, last_value):
    T = len(rewards)
    advantages = torch.zeros_like(rewards)
    lastgaelam = 0
    for t in reversed(range(T)):
        if t == T - 1:
            next_val = last_value
            next_nonterm = 1.0 - dones[t]
        else:
            next_val = values[t + 1]
            next_nonterm = 1.0 - dones[t]
        delta = rewards[t] + GAMMA * next_val * next_nonterm - values[t]
        advantages[t] = lastgaelam = delta + GAMMA * GAE_LAMBDA * next_nonterm * lastgaelam
    return advantages, advantages + values

def evaluate_vs_random(agent, env, num_episodes=5):
    """Evaluate Blue (Agent) vs Red (Random)."""
    agent.eval()
    returns = []
    wins = 0
    
    learner_ids = [1, 3] # Blue
    opponent_ids = [0, 2] # Red
    
    for _ in range(num_episodes):
        obs_dict, info = env.reset()
        episode_return = 0
        done = False
        
        while not done:
            # Prepare Blue Obs
            learner_obs_raw = [obs_dict[env.agents[i]].float() for i in learner_ids]
            # No flip needed for Blue
            learner_obs = torch.stack(learner_obs_raw).to(device)
            
            # Get Blue Masks
            masks = get_action_masks(info, learner_ids, env.agents, is_red_team=False).to(device)
            
            with torch.no_grad():
                actions = agent.get_deterministic_action(learner_obs, action_masks=masks).cpu()
            
            env_actions = {}
            for i, aid in enumerate(learner_ids):
                env_actions[env.agents[aid]] = actions[i].item()
            
            # Random Red Actions
            for aid in opponent_ids:
                legal = info['legal_actions'][env.agents[aid]]
                env_actions[env.agents[aid]] = np.random.choice(legal)
            
            obs_dict, rewards, dones, info = env.step(env_actions)
            episode_return += sum(rewards[env.agents[i]] for i in learner_ids)
            done = any(dones.values())
        
        returns.append(episode_return)
        if env.game.state.data.score > 0: wins += 1
    
    agent.train()
    return np.mean(returns), wins / num_episodes

def train():
    # 1. Setup Environments
    env = gymPacMan_parallel_env(
        layout_file='layouts/bloxCapture.lay', # Changed to bloxCapture or tinyCapture
        display=False,
        reward_forLegalAction=False,
        defenceReward=True,
        length=300,
        enemieName='randomTeam',
        self_play=True
    )
    
    eval_env = gymPacMan_parallel_env(
        layout_file='layouts/bloxCapture.lay',
        display=False,
        reward_forLegalAction=False,
        defenceReward=True,
        length=300,
        enemieName='randomTeam',
        self_play=True
    )
    
    obs_shape = env.get_Observation(0).shape
    num_agents = 2
    
    # 2. Initialize Agent
    agent = MAPPOAgent(obs_shape, 5, num_agents).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=LR, eps=1e-5)
    
    # 3. Load Checkpoint (If exists)
    start_update = 1
    if LOAD_CHECKPOINT_PATH and os.path.exists(LOAD_CHECKPOINT_PATH):
        print(f"Loading checkpoint from: {LOAD_CHECKPOINT_PATH}")
        try:
            state_dict = torch.load(LOAD_CHECKPOINT_PATH, map_location=device)
            agent.load_state_dict(state_dict)
            print("Successfully loaded agent weights.")
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            print("Starting from scratch...")
    else:
        print("Starting training from scratch.")

    # 4. Opponent Pool Setup
    # Start pool with current agent (whether loaded or random)
    opponent_pool = deque(maxlen=OPPONENT_POOL_SIZE)
    opponent_pool.append(copy.deepcopy(agent.state_dict()))
    
    # Logging
    log = {'reward': [], 'entropy': [], 'eval_ret': []}
    
    # --- TRAINING LOOP ---
    for update in range(start_update, TOTAL_UPDATES + 1):
        
        # Calculate Entropy Coefficient (Decay)
        frac = min(1.0, (update - 1.0) / ENT_DECAY_UPDATES)
        current_ent_coef = ENT_START - frac * (ENT_START - ENT_END)

        # Randomize Sides (Self Play)
        play_as_red = np.random.rand() > 0.5
        if play_as_red:
            learner_ids = [0, 2]
            opponent_ids = [1, 3]
        else:
            learner_ids = [1, 3]
            opponent_ids = [0, 2]

        # Load Opponent
        opponent = MAPPOAgent(obs_shape, 5, num_agents).to(device)
        opponent.load_state_dict(opponent_pool[np.random.randint(len(opponent_pool))])
        opponent.eval()
        
        # Buffers
        obs_buf = torch.zeros(NUM_STEPS, num_agents, *obs_shape)
        joint_obs_buf = torch.zeros(NUM_STEPS, num_agents, obs_shape[0]*num_agents, obs_shape[1], obs_shape[2])
        mask_buf = torch.zeros(NUM_STEPS, num_agents, 5) # Mask Buffer
        action_buf = torch.zeros(NUM_STEPS, num_agents, dtype=torch.long)
        logprob_buf = torch.zeros(NUM_STEPS, num_agents)
        reward_buf = torch.zeros(NUM_STEPS, num_agents)
        done_buf = torch.zeros(NUM_STEPS, num_agents)
        value_buf = torch.zeros(NUM_STEPS, num_agents)
        
        obs_dict, info = env.reset()
        
        # Rollout
        for step in range(NUM_STEPS):
            # Process Learner Obs
            learner_obs_raw = [obs_dict[env.agents[i]].float() for i in learner_ids]
            learner_obs_canon = [canonicalize_obs(o, play_as_red) for o in learner_obs_raw]
            learner_obs = torch.stack(learner_obs_canon)
            
            joint_obs = torch.cat(learner_obs_canon, dim=0)
            
            obs_buf[step] = learner_obs
            joint_obs_buf[step] = joint_obs.unsqueeze(0).expand(num_agents, -1, -1, -1)
            
            # Create Masks
            curr_masks = get_action_masks(info, learner_ids, env.agents, play_as_red)
            mask_buf[step] = curr_masks
            
            # Get Action
            with torch.no_grad():
                all_obs_list = [learner_obs[i:i+1].to(device) for i in range(num_agents)]
                actions, log_probs, values = [], [], []
                for i in range(num_agents):
                    act, lp, val, _ = agent.get_action_and_value(
                        learner_obs[i:i+1].to(device), 
                        all_obs_list,
                        action_masks=curr_masks[i:i+1].to(device)
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
            
            # Opponent Logic (Masked too for sanity)
            opp_obs_raw = [obs_dict[env.agents[i]].float() for i in opponent_ids]
            opp_obs_canon = [canonicalize_obs(o, not play_as_red) for o in opp_obs_raw]
            opp_obs = torch.stack(opp_obs_canon)
            
            with torch.no_grad():
                opp_list = [opp_obs[i:i+1].to(device) for i in range(num_agents)]
                opp_masks = get_action_masks(info, opponent_ids, env.agents, not play_as_red)
                opp_actions = []
                for i in range(num_agents):
                    a, _, _, _ = opponent.get_action_and_value(
                        opp_obs[i:i+1].to(device), 
                        opp_list,
                        action_masks=opp_masks[i:i+1].to(device)
                    )
                    opp_actions.append(a)
                opp_actions = torch.cat(opp_actions)
            
            # Map Actions to Env
            env_actions = {}
            for i, aid in enumerate(learner_ids):
                env_actions[env.agents[aid]] = canonicalize_action(actions[i], play_as_red).item()
            for i, aid in enumerate(opponent_ids):
                env_actions[env.agents[aid]] = canonicalize_action(opp_actions[i], not play_as_red).item()
            
            # Step
            next_obs_dict, rewards, dones, next_info = env.step(env_actions)
            
            # Rewards & Shaping
            team_reward = sum(rewards[env.agents[i]] for i in learner_ids)
            next_obs_raw = [next_obs_dict[env.agents[i]].float() for i in learner_ids]
            next_obs_canon = [canonicalize_obs(o, play_as_red) for o in next_obs_raw]
            
            shaping = [compute_heuristic_shaping(learner_obs_canon[i], next_obs_canon[i]) 
                       for i in range(num_agents)]
            
            for i in range(num_agents):
                # ADDED LIVING PENALTY HERE
                reward_buf[step, i] = team_reward + SHAPING_SCALE * shaping[i] + LIVING_PENALTY
            
            done = any(dones.values())
            done_buf[step] = float(done)
            
            obs_dict = next_obs_dict
            info = next_info
            
            if done:
                obs_dict, info = env.reset()

        # GAE Calculation
        learner_obs_raw = [obs_dict[env.agents[i]].float() for i in learner_ids]
        learner_obs_canon = [canonicalize_obs(o, play_as_red) for o in learner_obs_raw]
        learner_obs = torch.stack(learner_obs_canon)
        with torch.no_grad():
            all_list = [learner_obs[i:i+1].to(device) for i in range(num_agents)]
            last_value = agent.get_value(all_list).cpu().item()

        advantages = torch.zeros_like(reward_buf)
        returns = torch.zeros_like(reward_buf)
        for i in range(num_agents):
            adv, ret = compute_gae(reward_buf[:, i], value_buf[:, i], done_buf[:, i], last_value)
            advantages[:, i] = adv
            returns[:, i] = ret
            
        # Flatten for Update
        b_obs = obs_buf.reshape(-1, *obs_shape)
        b_joint = joint_obs_buf.reshape(-1, obs_shape[0]*num_agents, obs_shape[1], obs_shape[2])
        b_act = action_buf.reshape(-1)
        b_logp = logprob_buf.reshape(-1)
        b_adv = advantages.reshape(-1)
        b_ret = returns.reshape(-1)
        b_masks = mask_buf.reshape(-1, 5)

        # PPO Update
        inds = np.arange(NUM_STEPS * num_agents)
        loss_log = []
        ent_log = []
        
        for _ in range(UPDATE_EPOCHS):
            np.random.shuffle(inds)
            for start in range(0, len(inds), BATCH_SIZE):
                mb = inds[start:start+BATCH_SIZE]
                
                vals, lps, ent = agent.evaluate(
                    b_obs[mb].to(device), 
                    b_joint[mb].to(device), 
                    b_act[mb].to(device),
                    num_agents, obs_shape,
                    action_masks=b_masks[mb].to(device) # Pass Masks
                )
                
                norm_adv = (b_adv[mb] - b_adv[mb].mean()) / (b_adv[mb].std() + 1e-8)
                ratio = (lps - b_logp[mb].to(device)).exp()
                
                pg = -torch.min(
                    norm_adv.to(device) * ratio,
                    norm_adv.to(device) * torch.clamp(ratio, 1-CLIP_EPS, 1+CLIP_EPS)
                ).mean()
                vl = 0.5 * ((vals - b_ret[mb].to(device))**2).mean()
                
                # Dynamic Entropy Coefficient
                loss = pg + VF_COEF*vl - current_ent_coef * ent.mean()
                
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), MAX_GRAD_NORM)
                optimizer.step()
                
                ent_log.append(ent.mean().item())

        # Post-Update
        if update % OPPONENT_UPDATE_FREQ == 0:
            opponent_pool.append(copy.deepcopy(agent.state_dict()))
            
        mean_rew = reward_buf.sum().item() / num_agents
        mean_ent = np.mean(ent_log)
        
        # Eval
        eval_str = ""
        if update % EVAL_FREQ == 0:
            e_ret, e_win = evaluate_vs_random(agent, eval_env, EVAL_EPISODES)
            log['eval_ret'].append(e_ret)
            eval_str = f"| Eval: {e_ret:.1f} (Win: {e_win*100:.0f}%)"
            
            # Auto-save
            torch.save(agent.state_dict(), f"mappo_{update}.pt")

        print(f"Upd {update} | Rew: {mean_rew:.1f} | Ent: {mean_ent:.3f} (coef: {current_ent_coef:.3f}) {eval_str}")
        
        log['reward'].append(mean_rew)
        log['entropy'].append(mean_ent)

    # Final Save
    torch.save(agent.state_dict(), "mappo_pacman_final.pt")
    print("Training finished.")

if __name__ == "__main__":
    train()