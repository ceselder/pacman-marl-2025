import os
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from gymPacMan import gymPacMan_parallel_env
from collections import deque
import copy
import matplotlib.pyplot as plt

# --- Hyperparameters ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_STEPS = 2048        
BATCH_SIZE = 256        
LR = 2.5e-4             
GAMMA = 0.99            
GAE_LAMBDA = 0.95       
CLIP_EPS = 0.2          
ENT_COEF = 0.025        
VF_COEF = 0.5           
MAX_GRAD_NORM = 0.5     
UPDATE_EPOCHS = 5       
TOTAL_UPDATES = 300     

# Architecture Dimensions
VALUE_HIDDEN_DIM = 512
CRITIC_HIDDEN_DIM = 1024 

# Training Settings
REWARD_SCALE = 1.0        
WARMUP_UPDATES = 50       
OPPONENT_UPDATE_FREQ = 10  
OPPONENT_POOL_SIZE = 5     
SELF_PLAY_PROB = 0.5       
EVAL_FREQ = 10             
EVAL_EPISODES = 5          
SNAPSHOT_FREQ = 50        

# --- Agent Architecture ---
class MAPPOAgent(nn.Module):
    def __init__(self, obs_shape, action_dim, num_agents=2):
        super().__init__()
        
        # --- Decentralized Actor ---
        self.actor_conv = nn.Sequential(
            nn.Conv2d(obs_shape[0], 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Flatten()
        )
        
        with torch.no_grad():
            dummy_obs = torch.zeros(1, *obs_shape)
            actor_flat = self.actor_conv(dummy_obs).shape[1]
            
        self.actor = nn.Sequential(
            nn.Linear(actor_flat, VALUE_HIDDEN_DIM),   nn.ReLU(),
            nn.Linear(VALUE_HIDDEN_DIM, VALUE_HIDDEN_DIM), nn.ReLU(), 
            nn.Linear(VALUE_HIDDEN_DIM, action_dim)
        )

        # --- Centralized Critic ---
        critic_input_channels = obs_shape[0] * num_agents
        self.critic_conv = nn.Sequential(
            nn.Conv2d(critic_input_channels, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Flatten()
        )
        
        with torch.no_grad():
            dummy_state = torch.zeros(1, critic_input_channels, obs_shape[1], obs_shape[2])
            critic_flat = self.critic_conv(dummy_state).shape[1]

        self.critic = nn.Sequential(
            nn.Linear(critic_flat, CRITIC_HIDDEN_DIM),  nn.ReLU(),
            nn.Linear(CRITIC_HIDDEN_DIM, CRITIC_HIDDEN_DIM),   nn.ReLU(), 
            nn.Linear(CRITIC_HIDDEN_DIM, 1)
        )

    def get_action(self, x):
        hidden = self.actor_conv(x)
        logits = self.actor(hidden)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action)

    def get_value(self, state):
        hidden = self.critic_conv(state)
        return self.critic(hidden).squeeze(-1)

    def evaluate(self, obs, state, action):
        a_hidden = self.actor_conv(obs)
        logits = self.actor(a_hidden)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(action)
        entropy = dist.entropy()
        
        c_hidden = self.critic_conv(state)
        values = self.critic(c_hidden).squeeze(-1)
        
        return values, log_probs, entropy

# --- Helpers ---
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

def process_obs(obs, is_red_team): 
    """
    Standardizes observations:
    1. Swaps Channels (Red Food <-> Blue Food)
    2. HORIZONTAL FLIP (To match myTeam.py logic)
    """
    if not is_red_team:
        return obs 

    new_obs = obs.clone()
    
    # 1. Swap Channels
    new_obs[:, 2, :, :] = obs[:, 3, :, :] # Capsules
    new_obs[:, 3, :, :] = obs[:, 2, :, :]
    new_obs[:, 6, :, :] = obs[:, 7, :, :] # Food
    new_obs[:, 7, :, :] = obs[:, 6, :, :]
    
    # 2. HORIZONTAL FLIP (Batch, C, H, W) -> Flip W (dim 3)
    new_obs = torch.flip(new_obs, [3])
    
    return new_obs

def invert_actions(actions):
    """
    Maps actions for Horizontal Mirroring.
    0 (North) -> 0 (North)
    1 (East)  -> 3 (West)
    2 (South) -> 2 (South)
    3 (West)  -> 1 (East)
    4 (Stop)  -> 4 (Stop)
    """
    mapper = torch.tensor([0, 3, 2, 1, 4], device=actions.device)
    return mapper[actions]

def evaluate_vs_random(agent, eval_env, num_episodes=5):
    agent.eval()
    total_rewards = []
    learner_ids = [1, 3] # Blue
    
    for _ in range(num_episodes):
        obs_dict, _ = eval_env.reset(enemieName='randomTeam')
        obs = torch.stack([obs_dict[eval_env.agents[i]] for i in learner_ids]).float()
        processed_obs = process_obs(obs, is_red_team=False)
        done = False
        episode_reward = 0
        
        while not done:
            with torch.no_grad():
                actions, _ = agent.get_action(processed_obs.to(device))
                actions = actions.cpu()
            
            env_actions = {}
            for i, ag_id in enumerate(learner_ids):
                env_actions[eval_env.agents[ag_id]] = actions[i].item()
                
            next_obs_dict, rewards_dict, dones_dict, _ = eval_env.step(env_actions)
            step_reward = sum(rewards_dict[eval_env.agents[i]] for i in learner_ids)
            episode_reward += step_reward
            
            if any(dones_dict.values()):
                done = True
            else:
                obs = torch.stack([next_obs_dict[eval_env.agents[i]] for i in learner_ids]).float()
                processed_obs = process_obs(obs, is_red_team=False)
                
        total_rewards.append(episode_reward)
    
    agent.train()
    return np.mean(total_rewards)

# --- Main Training Loop ---
def train_mappo(train_env, eval_env):
    dummy_obs = train_env.get_Observation(0)
    obs_shape = dummy_obs.shape
    num_agents = 2
    state_shape = (obs_shape[0] * num_agents, obs_shape[1], obs_shape[2])
    
    agent = MAPPOAgent(obs_shape, 5, num_agents).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=LR, eps=1e-5)
    
    opponent_pool = deque(maxlen=OPPONENT_POOL_SIZE)
    opponent_pool.append(copy.deepcopy(agent.state_dict()))
    
    log = {
        'update': [], 'reward': [], 'eval_reward': [], 'episode_return': [], 
        'pg_loss': [], 'v_loss': [], 'entropy': [], 'clip_frac': []
    }
    episode_returns = []

    def get_opponent():
        opp = MAPPOAgent(obs_shape, 5, num_agents).to(device)
        if np.random.random() < SELF_PLAY_PROB and len(opponent_pool) > 1:
            opp.load_state_dict(np.random.choice(list(opponent_pool)[:-1]))
        else:
            opp.load_state_dict(opponent_pool[-1])
        opp.eval()
        return opp

    for update in range(1, TOTAL_UPDATES + 1):
        
        if update <= WARMUP_UPDATES:
            current_opp_is_random = True
            is_red_learner = False 
            train_ids = [1, 3]
            opp_ids = [0, 2]
        else:
            current_opp_is_random = False
            opponent = get_opponent()
            if np.random.rand() > 0.5:
                train_ids = [1, 3]; opp_ids = [0, 2]; is_red_learner = False
            else:
                train_ids = [0, 2]; opp_ids = [1, 3]; is_red_learner = True

        obs_buf = torch.zeros((NUM_STEPS, num_agents, *obs_shape))
        state_buf = torch.zeros((NUM_STEPS, num_agents, *state_shape))
        actions_buf = torch.zeros((NUM_STEPS, num_agents), dtype=torch.long)
        logprobs_buf = torch.zeros((NUM_STEPS, num_agents))
        rewards_buf = torch.zeros((NUM_STEPS, num_agents))
        dones_buf = torch.zeros((NUM_STEPS, num_agents))
        values_buf = torch.zeros((NUM_STEPS, num_agents))

        train_env.reset()
        raw_obs = torch.stack([train_env.get_Observation(i) for i in train_ids]).float()
        next_obs = process_obs(raw_obs, is_red_team=is_red_learner)
        curr_global_state = torch.cat([next_obs[0], next_obs[1]], dim=0)
        next_state = torch.stack([curr_global_state] * num_agents) 
        next_done = torch.zeros(num_agents)
        current_ep_reward = 0

        for step in range(NUM_STEPS):
            obs_buf[step] = next_obs
            state_buf[step] = next_state
            dones_buf[step] = next_done
            
            with torch.no_grad():
                action, logprob = agent.get_action(next_obs.to(device))
                value = agent.get_value(next_state.to(device))
                action, logprob, value = action.cpu(), logprob.cpu(), value.cpu()
            
            actions_buf[step], logprobs_buf[step], values_buf[step] = action, logprob, value

            # Learner Actions
            if is_red_learner:
                real_learner_actions = invert_actions(action)
            else:
                real_learner_actions = action

            # Opponent Actions
            if current_opp_is_random:
                opp_actions_scalar = torch.randint(0, 5, (num_agents,))
                real_opp_actions = opp_actions_scalar 
            else:
                opp_raw_obs = torch.stack([train_env.get_Observation(i) for i in opp_ids]).float()
                opp_is_red = not is_red_learner
                opp_obs = process_obs(opp_raw_obs, is_red_team=opp_is_red).to(device)
                opp_actions_scalar, _ = opponent.get_action(opp_obs)
                opp_actions_scalar = opp_actions_scalar.cpu()

                if opp_is_red:
                    real_opp_actions = invert_actions(opp_actions_scalar)
                else:
                    real_opp_actions = opp_actions_scalar

            # Execute
            env_actions = {}
            for i, ag_id in enumerate(train_ids):
                env_actions[train_env.agents[ag_id]] = real_learner_actions[i].item()
            for i, ag_id in enumerate(opp_ids):
                env_actions[train_env.agents[ag_id]] = real_opp_actions[i].item()

            next_obs_dict, rewards_dict, dones_dict, _ = train_env.step(env_actions)
            
            team_reward = sum(rewards_dict[train_env.agents[i]] for i in train_ids)
            rewards_buf[step] = team_reward * REWARD_SCALE
            current_ep_reward += team_reward
            
            any_done = any(dones_dict[train_env.agents[i]] for i in train_ids)
            next_done = torch.full((num_agents,), float(any_done))
            
            if any_done:
                episode_returns.append(current_ep_reward)
                current_ep_reward = 0
                train_env.reset()
                raw_obs = torch.stack([train_env.get_Observation(i) for i in train_ids]).float()
            else:
                raw_obs = torch.stack([next_obs_dict[train_env.agents[i]] for i in train_ids]).float()
            
            next_obs = process_obs(raw_obs, is_red_team=is_red_learner)
            curr_global_state = torch.cat([next_obs[0], next_obs[1]], dim=0)
            next_state = torch.stack([curr_global_state] * num_agents)

        # GAE
        with torch.no_grad():
            last_value = agent.get_value(next_state.to(device)).cpu()
        advantages, returns = compute_gae(rewards_buf, values_buf, dones_buf, last_value, next_done)

        b_obs, b_state = obs_buf.reshape(-1, *obs_shape), state_buf.reshape(-1, *state_shape)
        b_actions, b_logprobs = actions_buf.reshape(-1), logprobs_buf.reshape(-1)
        b_advantages, b_returns = advantages.reshape(-1), returns.reshape(-1)
        
        indices = np.arange(NUM_STEPS * num_agents)
        pg_losses, v_losses, entropies, clip_fracs = [], [], [], []

        for _ in range(UPDATE_EPOCHS):
            np.random.shuffle(indices)
            for start in range(0, len(indices), BATCH_SIZE):
                mb = indices[start:start + BATCH_SIZE]
                mb_adv = (b_advantages[mb] - b_advantages[mb].mean()) / (b_advantages[mb].std() + 1e-8)

                newvalue, newlogprob, entropy = agent.evaluate(b_obs[mb].to(device), b_state[mb].to(device), b_actions[mb].to(device))
                ratio = (newlogprob - b_logprobs[mb].to(device)).exp()
                
                with torch.no_grad():
                    clip_fracs.append(((ratio - 1.0).abs() > CLIP_EPS).float().mean().item())

                pg_loss = -torch.min(ratio * mb_adv.to(device), torch.clamp(ratio, 1-CLIP_EPS, 1+CLIP_EPS) * mb_adv.to(device)).mean()
                v_loss = 0.5 * ((newvalue - b_returns[mb].to(device)) ** 2).mean()
                loss = pg_loss - ENT_COEF * entropy.mean() + VF_COEF * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), MAX_GRAD_NORM)
                optimizer.step()
                
                pg_losses.append(pg_loss.item()); v_losses.append(v_loss.item()); entropies.append(entropy.mean().item())

        if update % OPPONENT_UPDATE_FREQ == 0:
            opponent_pool.append(copy.deepcopy(agent.state_dict()))

        if update % EVAL_FREQ == 0:
            eval_score = evaluate_vs_random(agent, eval_env, num_episodes=EVAL_EPISODES)
            log['eval_reward'].append(eval_score)
        else:
            log['eval_reward'].append(log['eval_reward'][-1] if log['eval_reward'] else 0)

        if update % SNAPSHOT_FREQ == 0:
            fn = f"mappo_pacman_update_{update}.pt"
            torch.save(agent.state_dict(), fn)
            print(f"--- Saved {fn} ---")

        mean_reward = rewards_buf.sum().item() / num_agents
        mean_ep_return = np.mean(episode_returns[-10:]) if episode_returns else 0
        side = "Warm" if update <= WARMUP_UPDATES else ("Red" if is_red_learner else "Blue")
        
        print(f"Upd {update:3d}/{TOTAL_UPDATES} [{side:<4}] | Rwd: {mean_reward:6.1f} | EpRet: {mean_ep_return:6.1f} | Eval: {log['eval_reward'][-1]:6.1f} | Ent: {np.mean(entropies):.3f} | Clip: {np.mean(clip_fracs):.3f}")
        
        log['update'].append(update)
        log['reward'].append(mean_reward)
        log['episode_return'].append(mean_ep_return)
        log['pg_loss'].append(np.mean(pg_losses))
        log['v_loss'].append(np.mean(v_losses))
        log['entropy'].append(np.mean(entropies))
        log['clip_frac'].append(np.mean(clip_fracs))

    torch.save(agent.state_dict(), "mappo_pacman_final.pt")
    return log

def plot_training(log):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    axes[0,0].plot(log['update'], log['reward'])
    axes[0,0].set_title('Rollout Reward (Scaled)')
    axes[0,0].set_xlabel('Update')
    axes[0,0].grid(True)
    
    axes[0,1].plot(log['update'], log['episode_return'], label='Self-Play')
    axes[0,1].plot(log['update'], log['eval_reward'], label='vs Random', color='orange')
    axes[0,1].set_title('Episode Return')
    axes[0,1].legend()
    axes[0,1].set_xlabel('Update')
    axes[0,1].grid(True)
    
    axes[0,2].plot(log['update'], log['entropy'])
    axes[0,2].set_title('Entropy')
    axes[0,2].set_xlabel('Update')
    axes[0,2].grid(True)
    
    axes[1,0].plot(log['update'], log['pg_loss'])
    axes[1,0].set_title('Policy Loss')
    axes[1,0].set_xlabel('Update')
    axes[1,0].grid(True)
    
    axes[1,1].plot(log['update'], log['v_loss'])
    axes[1,1].set_title('Value Loss')
    axes[1,1].set_xlabel('Update')
    axes[1,1].grid(True)
    
    axes[1,2].plot(log['update'], log['clip_frac'])
    axes[1,2].set_title('Clip Fraction')
    axes[1,2].set_xlabel('Update')
    axes[1,2].axhline(y=0.2, color='r', linestyle='--', alpha=0.5)
    axes[1,2].grid(True)
    
    plt.tight_layout()
    plt.savefig('mappo_training_curves.png', dpi=150)

if __name__ == "__main__":
    train_env = gymPacMan_parallel_env(layout_file=os.path.join('layouts', 'tinyCapture.lay'), display=False, reward_forLegalAction=False, defenceReward=True, length=300, enemieName='randomTeam', self_play=True, random_layout=False)
    eval_env = gymPacMan_parallel_env(layout_file=os.path.join('layouts', 'tinyCapture.lay'), display=False, reward_forLegalAction=False, defenceReward=True, length=300, enemieName='randomTeam', self_play=False, random_layout=False)
    
    print("Starting MAPPO with WARMUP Curriculum (Horizontal Mirror)...")
    log = train_mappo(train_env, eval_env)
    plot_training(log)