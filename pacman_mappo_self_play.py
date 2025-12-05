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

NUM_STEPS = 2048        # Steps to collect per agent per update
BATCH_SIZE = 256        # Minibatch size
LR = 1e-4             # Learning Rate
GAMMA = 0.985            # Discount factor
GAE_LAMBDA = 0.95       # GAE parameter
CLIP_EPS = 0.15         # PPO Clip range
ENT_COEF = 0.02        # Entropy coefficient
VF_COEF = 0.5           # Value Function coefficient
MAX_GRAD_NORM = 0.5     # Gradient clipping
UPDATE_EPOCHS = 5       # PPO Update epochs
TOTAL_UPDATES = 500     # Total training iterations

HIDDEN_DIM = 512
HIDDEN_DIM_CRITIC = 1024

OPPONENT_UPDATE_FREQ = 10  # Add current agent to pool every N updates
OPPONENT_POOL_SIZE = 5     # Max opponents to keep
SELF_PLAY_PROB = 0.5       # Probability of playing against older versions
EVAL_FREQ = 10             # Evaluate against random every N updates
EVAL_EPISODES = 5          # Number of episodes for evaluation

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
            nn.Linear(actor_flat, HIDDEN_DIM),   nn.ReLU(),  # Layer 1
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),   nn.ReLU(),  # Layer 2 (Input must match Prev Output)
            nn.Linear(HIDDEN_DIM, action_dim)                # Output
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
            nn.Linear(critic_flat, HIDDEN_DIM_CRITIC),  nn.ReLU(),
            nn.Linear(HIDDEN_DIM_CRITIC, HIDDEN_DIM_CRITIC),   nn.ReLU(),
            nn.Linear(HIDDEN_DIM_CRITIC, 1)
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

# --- Helper Functions ---

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
    #yeah okay so they said I dont have to do this but im pretty sure if I wanna do self play I do actually
    if not is_red_team:
        return obs # Blue is already default

    new_obs = obs.clone()
    
    # Swap Capsules: Blue(2) <-> Red(3)
    new_obs[:, 2, :, :] = obs[:, 3, :, :]
    new_obs[:, 3, :, :] = obs[:, 2, :, :]
    
    # Swap Food: Blue(6) <-> Red(7)
    new_obs[:, 6, :, :] = obs[:, 7, :, :]
    new_obs[:, 7, :, :] = obs[:, 6, :, :]
    
    return new_obs

def evaluate_vs_random(agent, eval_env, num_episodes=5):
    agent.eval()
    total_rewards = []
    
    # Blue Team IDs
    learner_ids = [1, 3]
    
    for _ in range(num_episodes):
        obs_dict, _ = eval_env.reset(enemieName='randomTeam')
        
        # Get raw observations for Blue agents
        obs = torch.stack([obs_dict[eval_env.agents[i]] for i in learner_ids]).float()
        
        # Blue obs do not need processing (is_red_team=False)
        processed_obs = process_obs(obs, is_red_team=False)
        
        done = False
        episode_reward = 0
        
        while not done:
            with torch.no_grad():
                # Get action from Policy
                actions, _ = agent.get_action(processed_obs.to(device))
                actions = actions.cpu()
            
            # Create Action Dictionary for Env
            # Note: We only provide actions for [1, 3]. 
            # The Env's internal 'randomTeam' handles [0, 2] automatically because self_play=False.
            env_actions = {}
            for i, ag_id in enumerate(learner_ids):
                env_actions[eval_env.agents[ag_id]] = actions[i].item()
                
            next_obs_dict, rewards_dict, dones_dict, _ = eval_env.step(env_actions)
            
            # Sum reward for Blue Team
            step_reward = sum(rewards_dict[eval_env.agents[i]] for i in learner_ids)
            episode_reward += step_reward
            
            if any(dones_dict.values()):
                done = True
            else:
                obs = torch.stack([next_obs_dict[eval_env.agents[i]] for i in learner_ids]).float()
                processed_obs = process_obs(obs, is_red_team=False)
                
        total_rewards.append(episode_reward)
        
    avg_reward = np.mean(total_rewards)
    agent.train() # Switch back to train mode
    return avg_reward

# --- Main Training Loop ---

def train_mappo(env, eval_env):
    # Initialize info
    dummy_obs = env.get_Observation(0)
    obs_shape = dummy_obs.shape
    num_agents = 2
    state_shape = (obs_shape[0] * num_agents, obs_shape[1], obs_shape[2])
    
    agent = MAPPOAgent(obs_shape, 5, num_agents).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=LR, eps=1e-5)
    
    # Self-play pool
    opponent_pool = deque(maxlen=OPPONENT_POOL_SIZE)
    opponent_pool.append(copy.deepcopy(agent.state_dict()))
    
    # Logs
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
        # 1. Evaluate periodically
        if update % EVAL_FREQ == 0:
            print(f"--- Evaluating vs Random (Update {update}) ---")
            eval_score = evaluate_vs_random(agent, eval_env, num_episodes=EVAL_EPISODES)
            log['eval_reward'].append(eval_score)
            print(f"    Mean Eval Reward: {eval_score:.2f}")
        else:
            # Fill with NaN or previous to keep array lengths consistent (optional)
            if log['eval_reward']:
                log['eval_reward'].append(log['eval_reward'][-1])
            else:
                log['eval_reward'].append(0)

        # 2. Setup Self-Play Match
        opponent = get_opponent()
        
        # I know it sais you dont have to do this but for self play im pretty sure we do
        # for self play atleast
        if np.random.rand() > 0.5:
            train_ids = [1, 3] # We are Blue
            opp_ids = [0, 2]   # Opponent is Red
            is_red_learner = False
        else:
            train_ids = [0, 2] # We are Red
            opp_ids = [1, 3]   # Opponent is Blue
            is_red_learner = True
        
        # Buffers
        obs_buf = torch.zeros((NUM_STEPS, num_agents, *obs_shape))
        state_buf = torch.zeros((NUM_STEPS, num_agents, *state_shape))
        actions_buf = torch.zeros((NUM_STEPS, num_agents), dtype=torch.long)
        logprobs_buf = torch.zeros((NUM_STEPS, num_agents))
        rewards_buf = torch.zeros((NUM_STEPS, num_agents))
        dones_buf = torch.zeros((NUM_STEPS, num_agents))
        values_buf = torch.zeros((NUM_STEPS, num_agents))

        # Reset Env
        env.reset()
        
        # Initial Obs
        raw_obs = torch.stack([env.get_Observation(i) for i in train_ids]).float()
        # Canonicalize
        next_obs = process_obs(raw_obs, is_red_team=is_red_learner)
        
        # Create Global State (Concatenate Canonical Obs)
        curr_global_state = torch.cat([next_obs[0], next_obs[1]], dim=0)
        next_state = torch.stack([curr_global_state] * num_agents) 

        next_done = torch.zeros(num_agents)
        current_ep_reward = 0

        # --- Rollout ---
        for step in range(NUM_STEPS):
            obs_buf[step] = next_obs
            state_buf[step] = next_state
            dones_buf[step] = next_done
            
            with torch.no_grad():
                action, logprob = agent.get_action(next_obs.to(device))
                value = agent.get_value(next_state.to(device))
                action, logprob, value = action.cpu(), logprob.cpu(), value.cpu()
            
            actions_buf[step], logprobs_buf[step], values_buf[step] = action, logprob, value

            # Opponent Logic (Frozen)
            opp_raw_obs = torch.stack([env.get_Observation(i) for i in opp_ids]).float()
            # Opponent is the OTHER team, so invert 'is_red_learner'
            opp_obs = process_obs(opp_raw_obs, is_red_team=(not is_red_learner)).to(device)
            opp_actions, _ = opponent.get_action(opp_obs)
            opp_actions = opp_actions.cpu()

            # Execute
            env_actions = {}
            for i, ag_id in enumerate(train_ids):
                env_actions[env.agents[ag_id]] = action[i].item()
            for i, ag_id in enumerate(opp_ids):
                env_actions[env.agents[ag_id]] = opp_actions[i].item()

            next_obs_dict, rewards_dict, dones_dict, _ = env.step(env_actions)
            
            team_reward = sum(rewards_dict[env.agents[i]] for i in train_ids)
            rewards_buf[step] = team_reward
            current_ep_reward += team_reward
            
            any_done = any(dones_dict[env.agents[i]] for i in train_ids)
            next_done = torch.full((num_agents,), float(any_done))
            
            if any_done:
                episode_returns.append(current_ep_reward)
                current_ep_reward = 0
                env.reset()
                raw_obs = torch.stack([env.get_Observation(i) for i in train_ids]).float()
            else:
                raw_obs = torch.stack([next_obs_dict[env.agents[i]] for i in train_ids]).float()
            
            next_obs = process_obs(raw_obs, is_red_team=is_red_learner)
            curr_global_state = torch.cat([next_obs[0], next_obs[1]], dim=0)
            next_state = torch.stack([curr_global_state] * num_agents)

        # --- GAE ---
        with torch.no_grad():
            last_value = agent.get_value(next_state.to(device)).cpu()

        advantages, returns = compute_gae(rewards_buf, values_buf, dones_buf, last_value, next_done)

        # --- Update ---
        b_obs = obs_buf.reshape(-1, *obs_shape)
        b_state = state_buf.reshape(-1, *state_shape)
        b_actions = actions_buf.reshape(-1)
        b_logprobs = logprobs_buf.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        
        dataset_size = NUM_STEPS * num_agents
        indices = np.arange(dataset_size)
        
        pg_losses, v_losses, entropies, clip_fracs = [], [], [], []

        for _ in range(UPDATE_EPOCHS):
            np.random.shuffle(indices)
            for start in range(0, dataset_size, BATCH_SIZE):
                mb = indices[start:start + BATCH_SIZE]
                
                mb_obs = b_obs[mb].to(device)
                mb_state = b_state[mb].to(device)
                mb_actions = b_actions[mb].to(device)
                mb_adv = b_advantages[mb].to(device)
                mb_returns = b_returns[mb].to(device)
                mb_old_logprobs = b_logprobs[mb].to(device)
                
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                newvalue, newlogprob, entropy = agent.evaluate(mb_obs, mb_state, mb_actions)
                ratio = (newlogprob - mb_old_logprobs).exp()
                
                with torch.no_grad():
                    clip_frac = ((ratio - 1.0).abs() > CLIP_EPS).float().mean().item()
                    clip_fracs.append(clip_frac)

                pg_loss = -torch.min(
                    ratio * mb_adv,
                    torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * mb_adv
                ).mean()
                
                v_loss = 0.5 * ((newvalue - mb_returns) ** 2).mean()
                ent = entropy.mean()
                loss = pg_loss - ENT_COEF * ent + VF_COEF * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), MAX_GRAD_NORM)
                optimizer.step()
                
                pg_losses.append(pg_loss.item())
                v_losses.append(v_loss.item())
                entropies.append(ent.item())

        if update % OPPONENT_UPDATE_FREQ == 0:
            opponent_pool.append(copy.deepcopy(agent.state_dict()))

        # Console Log
        mean_reward = rewards_buf.sum().item() / num_agents
        mean_ep_return = np.mean(episode_returns[-10:]) if episode_returns else 0
        side = "Red" if is_red_learner else "Blue"
        
        print(f"Update {update:3d}/{TOTAL_UPDATES} [{side:<4}] | "
              f"Reward: {mean_reward:6.2f} | "
              f"Mean Episodic return: {mean_ep_return:6.2f} | "
              f"Entropy: {np.mean(entropies):.3f} | "
              f"PLoss: {np.mean(pg_losses):6.3f} | "
              f"VLoss: {np.mean(v_losses):6.3f} | "
              f"Clip: {np.mean(clip_fracs):.3f}")

        # Store logs
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
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    
    # 1. Rollout Reward
    axes[0,0].plot(log['update'], log['reward'])
    axes[0,0].set_title('Rollout Reward')
    axes[0,0].set_xlabel('Update')
    
    # 2. Episode Return vs Eval Return
    axes[0,1].plot(log['update'], log['episode_return'], label='Self-Play')
    axes[0,1].plot(log['update'], log['eval_reward'], label='vs Random', color='orange')
    axes[0,1].set_title('Episode Return (10-ep avg)')
    axes[0,1].legend()
    axes[0,1].set_xlabel('Update')
    
    # 3. Entropy
    axes[0,2].plot(log['update'], log['entropy'])
    axes[0,2].set_title('Entropy')
    axes[0,2].set_xlabel('Update')
    
    # 4. Losses
    axes[1,0].plot(log['update'], log['pg_loss'])
    axes[1,0].set_title('Policy Loss')
    axes[1,0].set_xlabel('Update')
    
    axes[1,1].plot(log['update'], log['v_loss'])
    axes[1,1].set_title('Value Loss')
    axes[1,1].set_xlabel('Update')
    
    # 5. Clip Fraction
    axes[1,2].plot(log['update'], log['clip_frac'])
    axes[1,2].set_title('Clip Fraction')
    axes[1,2].set_xlabel('Update')
    
    plt.tight_layout()
    plt.savefig('mappo_training_curves.png', dpi=150)

if __name__ == "__main__":
    # 1. Create Training Environment (Self-Play enabled)
    train_env = gymPacMan_parallel_env(
        layout_file=os.path.join('layouts', 'bloxCapture.lay'),
        display=False, 
        reward_forLegalAction=True, 
        defenceReward=True,
        length=300, 
        enemieName='randomTeam',
        self_play=True,  # Crucial for training
        random_layout=False
    )
    
    # 2. Create Evaluation Environment (Self-Play disabled)
    # This allows 'randomTeam' to control the opponent logic natively
    eval_env = gymPacMan_parallel_env(
        layout_file=os.path.join('layouts', 'bloxCapture.lay'),
        display=False,
        reward_forLegalAction=True,
        defenceReward=True,
        length=300,
        enemieName='randomTeam',
        self_play=False, # Crucial for valid evaluation
        random_layout=False
    )
    
    print("Starting MAPPO Training with Self-Play and Periodic Evaluation...")
    log = train_mappo(train_env, eval_env)
    plot_training(log)
    print("Training Complete.")