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

# --- Hyperparameters (more conservative) ---
NUM_STEPS = 2048
BATCH_SIZE = 256          # Smaller batches = more updates = smoother
LR = 3e-4                 # Reduced from 5e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.15           # Reduced from 0.2 for stability
ENT_COEF = 0.03           # Slightly lower entropy
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5
UPDATE_EPOCHS = 4
TOTAL_UPDATES = 1000

# Settings
OPPONENT_POOL_SIZE = 5
OPPONENT_UPDATE_FREQ = 20
SHAPING_SCALE = 0.1
EVAL_FREQ = 20            # Eval every N updates
EVAL_EPISODES = 10        # Episodes per eval


def canonicalize_obs(obs, is_red_agent):
    """Canonicalize observation so red team sees the world as if they were blue."""
    if not is_red_agent:
        return obs
    
    is_batched = obs.dim() == 4
    if not is_batched:
        obs = obs.unsqueeze(0)
    
    canon = torch.flip(obs.clone(), dims=[-1])
    canon[:, [2, 3], :, :] = canon[:, [3, 2], :, :]  # Swap capsules
    canon[:, [6, 7], :, :] = canon[:, [7, 6], :, :]  # Swap food
    
    if not is_batched:
        canon = canon.squeeze(0)
    return canon


def canonicalize_action(action, is_red_agent):
    """Swap East <-> West for red team."""
    if not is_red_agent:
        return action
    if isinstance(action, torch.Tensor):
        mapper = torch.tensor([0, 3, 2, 1, 4], device=action.device)
        return mapper[action]
    return {0: 0, 1: 3, 2: 2, 3: 1, 4: 4}[action]


def get_agent_state(obs_canon):
    agent_ch = obs_canon[1]
    locs = (agent_ch > 0).nonzero(as_tuple=False)
    if len(locs) == 0:
        return None, 0
    y, x = locs[0][0].item(), locs[0][1].item()
    carrying = agent_ch[y, x].item() - 1
    return (y, x), carrying


def compute_heuristic_shaping(obs_curr, obs_next):
    pos_curr, carry_curr = get_agent_state(obs_curr)
    pos_next, carry_next = get_agent_state(obs_next)
    
    if pos_curr is None or pos_next is None:
        return 0.0
    if carry_next > carry_curr or (carry_curr > 0 and carry_next == 0):
        return 0.0

    if carry_curr > 0:
        dist_curr = pos_curr[1]
        dist_next = pos_next[1]
    else:
        food_ch = obs_curr[7]
        food_locs = (food_ch > 0).nonzero(as_tuple=False).float()
        if len(food_locs) == 0:
            return 0.0
        curr_p = torch.tensor(pos_curr).float()
        next_p = torch.tensor(pos_next).float()
        dist_curr = (food_locs - curr_p).abs().sum(dim=1).min().item()
        dist_next = (food_locs - next_p).abs().sum(dim=1).min().item()

    return dist_curr - dist_next


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

    def get_action_and_value(self, obs, all_obs_list):
        h = self.conv(obs)
        logits = self.actor(h)
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

    def evaluate(self, obs, all_obs_flat, action, num_agents, obs_shape):
        h = self.conv(obs)
        logits = self.actor(h)
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
    
    def get_deterministic_action(self, obs):
        """Get greedy action for evaluation."""
        with torch.no_grad():
            h = self.conv(obs)
            logits = self.actor(h)
            return logits.argmax(dim=-1)


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


def evaluate_vs_random(agent, env, num_episodes=10):
    """Evaluate agent playing as blue vs random red."""
    agent.eval()
    returns = []
    wins = 0
    
    learner_ids = [1, 3]
    opponent_ids = [0, 2]
    
    for _ in range(num_episodes):
        obs_dict, info = env.reset()
        episode_return = 0
        done = False
        
        while not done:
            # Get learner observations (no canonicalization needed - playing blue)
            learner_obs = torch.stack([obs_dict[env.agents[i]].float() for i in learner_ids])
            
            # Deterministic actions for eval
            with torch.no_grad():
                actions = agent.get_deterministic_action(learner_obs.to(device)).cpu()
            
            # Random opponent actions
            env_actions = {}
            for i, aid in enumerate(learner_ids):
                env_actions[env.agents[aid]] = actions[i].item()
            for aid in opponent_ids:
                legal = info['legal_actions'][env.agents[aid]]
                env_actions[env.agents[aid]] = np.random.choice(legal)
            
            obs_dict, rewards, dones, info = env.step(env_actions)
            episode_return += sum(rewards[env.agents[i]] for i in learner_ids)
            done = any(dones.values())
        
        returns.append(episode_return)
        # Win if positive score (simplified)
        final_score = env.game.state.data.score
        if final_score > 0:  # Blue (our team) winning
            wins += 1
    
    agent.train()
    return np.mean(returns), np.std(returns), wins / num_episodes


def train():
    env = gymPacMan_parallel_env(
        layout_file='layouts/bloxCapture.lay',
        display=False,
        reward_forLegalAction=False,
        defenceReward=True,
        length=300,
        enemieName='randomTeam',
        self_play=True
    )
    
    # Separate eval env
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
    
    agent = MAPPOAgent(obs_shape, 5, num_agents).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=LR, eps=1e-5)

    state_dict = torch.load("mappo_100.pt", map_location=device)
    agent.load_state_dict(state_dict)


    opponent_pool = deque(maxlen=OPPONENT_POOL_SIZE)
    opponent_pool.append(copy.deepcopy(agent.state_dict()))
    
    # Logging
    log = {
        'update': [],
        'reward': [],
        'ep_return': [],
        'eval_return': [],
        'eval_winrate': [],
        'entropy': [],
        'clip_frac': [],
        'pg_loss': [],
        'v_loss': []
    }
    
    for update in range(1, TOTAL_UPDATES + 1):
        # Random side selection
        play_as_red = np.random.rand() > 0.5
        if play_as_red:
            learner_ids = [0, 2]
            opponent_ids = [1, 3]
        else:
            learner_ids = [1, 3]
            opponent_ids = [0, 2]

        # Load opponent from pool
        opponent = MAPPOAgent(obs_shape, 5, num_agents).to(device)
        opponent.load_state_dict(opponent_pool[np.random.randint(len(opponent_pool))])
        opponent.eval()
        
        # Buffers
        obs_buf = torch.zeros(NUM_STEPS, num_agents, *obs_shape)
        joint_obs_buf = torch.zeros(NUM_STEPS, num_agents, obs_shape[0]*num_agents, obs_shape[1], obs_shape[2])
        action_buf = torch.zeros(NUM_STEPS, num_agents, dtype=torch.long)
        logprob_buf = torch.zeros(NUM_STEPS, num_agents)
        reward_buf = torch.zeros(NUM_STEPS, num_agents)
        done_buf = torch.zeros(NUM_STEPS, num_agents)
        value_buf = torch.zeros(NUM_STEPS, num_agents)
        
        obs_dict, _ = env.reset()
        episode_return = 0
        episode_returns = []
        
        for step in range(NUM_STEPS):
            learner_obs_raw = [obs_dict[env.agents[i]].float() for i in learner_ids]
            learner_obs_canon = [canonicalize_obs(o, play_as_red) for o in learner_obs_raw]
            learner_obs = torch.stack(learner_obs_canon)
            
            joint_obs = torch.cat(learner_obs_canon, dim=0)
            joint_obs_batch = joint_obs.unsqueeze(0).expand(num_agents, -1, -1, -1)
            
            obs_buf[step] = learner_obs
            joint_obs_buf[step] = joint_obs_batch
            
            with torch.no_grad():
                all_obs_list = [learner_obs[i:i+1].to(device) for i in range(num_agents)]
                actions, log_probs, values = [], [], []
                for i in range(num_agents):
                    act, lp, val, _ = agent.get_action_and_value(
                        learner_obs[i:i+1].to(device), all_obs_list
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
            
            # Opponent actions
            opp_obs_raw = [obs_dict[env.agents[i]].float() for i in opponent_ids]
            opp_obs_canon = [canonicalize_obs(o, not play_as_red) for o in opp_obs_raw]
            opp_obs = torch.stack(opp_obs_canon)
            
            with torch.no_grad():
                opp_list = [opp_obs[i:i+1].to(device) for i in range(num_agents)]
                opp_actions = []
                for i in range(num_agents):
                    a, _, _, _ = opponent.get_action_and_value(opp_obs[i:i+1].to(device), opp_list)
                    opp_actions.append(a)
                opp_actions = torch.cat(opp_actions)
            
            env_actions = {}
            for i, aid in enumerate(learner_ids):
                env_actions[env.agents[aid]] = canonicalize_action(actions[i], play_as_red).item()
            for i, aid in enumerate(opponent_ids):
                env_actions[env.agents[aid]] = canonicalize_action(opp_actions[i], not play_as_red).item()
            
            next_obs_dict, rewards, dones, _ = env.step(env_actions)
            
            # Shaping
            next_obs_raw = [next_obs_dict[env.agents[i]].float() for i in learner_ids]
            next_obs_canon = [canonicalize_obs(o, play_as_red) for o in next_obs_raw]
            shaping = [compute_heuristic_shaping(learner_obs_canon[i], next_obs_canon[i]) 
                       for i in range(num_agents)]
            
            team_reward = sum(rewards[env.agents[i]] for i in learner_ids)
            episode_return += team_reward
            
            for i in range(num_agents):
                reward_buf[step, i] = team_reward + SHAPING_SCALE * shaping[i]
            
            done = any(dones.values())
            done_buf[step] = float(done)
            obs_dict = next_obs_dict
            
            if done:
                episode_returns.append(episode_return)
                episode_return = 0
                obs_dict, _ = env.reset()

        # GAE
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
            
        # PPO Update
        b_obs = obs_buf.reshape(-1, *obs_shape)
        b_joint = joint_obs_buf.reshape(-1, obs_shape[0]*num_agents, obs_shape[1], obs_shape[2])
        b_act = action_buf.reshape(-1)
        b_logp = logprob_buf.reshape(-1)
        b_adv = advantages.reshape(-1)
        b_ret = returns.reshape(-1)
        
        inds = np.arange(NUM_STEPS * num_agents)
        pg_l, v_l, ent_l, clip_l = [], [], [], []
        
        for _ in range(UPDATE_EPOCHS):
            np.random.shuffle(inds)
            for start in range(0, len(inds), BATCH_SIZE):
                mb = inds[start:start+BATCH_SIZE]
                
                vals, lps, ent = agent.evaluate(
                    b_obs[mb].to(device), 
                    b_joint[mb].to(device), 
                    b_act[mb].to(device),
                    num_agents, obs_shape
                )
                
                norm_adv = (b_adv[mb] - b_adv[mb].mean()) / (b_adv[mb].std() + 1e-8)
                ratio = (lps - b_logp[mb].to(device)).exp()
                
                pg = -torch.min(
                    norm_adv.to(device) * ratio,
                    norm_adv.to(device) * torch.clamp(ratio, 1-CLIP_EPS, 1+CLIP_EPS)
                ).mean()
                vl = 0.5 * ((vals - b_ret[mb].to(device))**2).mean()
                loss = pg + VF_COEF*vl - ENT_COEF*ent.mean()
                
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), MAX_GRAD_NORM)
                optimizer.step()
                
                pg_l.append(pg.item())
                v_l.append(vl.item())
                ent_l.append(ent.mean().item())
                clip_l.append(((ratio - 1).abs() > CLIP_EPS).float().mean().item())

        # Update opponent pool
        if update % OPPONENT_UPDATE_FREQ == 0:
            opponent_pool.append(copy.deepcopy(agent.state_dict()))
        
        # Evaluation
        if update % EVAL_FREQ == 0:
            eval_ret, eval_std, eval_wr = evaluate_vs_random(agent, eval_env, EVAL_EPISODES)
            print(f"(EVAL RESULTS) ret: {eval_ret} {eval_std} {eval_wr}")
            log['eval_return'].append(eval_ret)
            log['eval_winrate'].append(eval_wr)
        else:
            log['eval_return'].append(log['eval_return'][-1] if log['eval_return'] else 0)
            log['eval_winrate'].append(log['eval_winrate'][-1] if log['eval_winrate'] else 0)
        
        # Logging
        mean_reward = reward_buf.sum().item() / num_agents
        mean_ep_return = np.mean(episode_returns) if episode_returns else 0
        
        log['update'].append(update)
        log['reward'].append(mean_reward)
        log['ep_return'].append(mean_ep_return)
        log['entropy'].append(np.mean(ent_l))
        log['clip_frac'].append(np.mean(clip_l))
        log['pg_loss'].append(np.mean(pg_l))
        log['v_loss'].append(np.mean(v_l))
        
        side = "Red " if play_as_red else "Blue"
        eval_str = f"| Eval: {log['eval_return'][-1]:6.1f} ({log['eval_winrate'][-1]*100:4.1f}%)" if update % EVAL_FREQ == 0 else ""
        print(f"Upd {update:4d} [{side}] | "
              f"Rew: {mean_reward:7.1f} | "
              f"EpRet: {mean_ep_return:6.1f} | "
              f"Ent: {np.mean(ent_l):.3f} | "
              f"Clip: {np.mean(clip_l):.3f} "
              f"{eval_str}")
        
        if update % 100 == 0:
            torch.save(agent.state_dict(), f"mappo_{update}.pt")

    torch.save(agent.state_dict(), "mappo_final.pt")
    
    # Plotting
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    axes[0, 0].plot(log['update'], log['reward'])
    axes[0, 0].set_title('Rollout Reward')
    axes[0, 0].set_xlabel('Update')
    
    axes[0, 1].plot(log['update'], log['ep_return'])
    axes[0, 1].set_title('Episode Return (Training)')
    axes[0, 1].set_xlabel('Update')
    
    axes[0, 2].plot(log['update'], log['eval_return'], label='Return')
    axes[0, 2].set_title('Eval vs Random')
    axes[0, 2].set_xlabel('Update')
    ax2 = axes[0, 2].twinx()
    ax2.plot(log['update'], [w*100 for w in log['eval_winrate']], 'r-', alpha=0.5, label='Win%')
    ax2.set_ylabel('Win Rate %', color='r')
    
    axes[1, 0].plot(log['update'], log['entropy'])
    axes[1, 0].set_title('Entropy')
    axes[1, 0].axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
    axes[1, 0].set_xlabel('Update')
    
    axes[1, 1].plot(log['update'], log['clip_frac'])
    axes[1, 1].set_title('Clip Fraction')
    axes[1, 1].axhline(y=0.2, color='r', linestyle='--', alpha=0.5)
    axes[1, 1].set_xlabel('Update')
    
    axes[1, 2].plot(log['update'], log['v_loss'])
    axes[1, 2].set_title('Value Loss')
    axes[1, 2].set_xlabel('Update')
    
    plt.tight_layout()
    plt.savefig("mappo_training.png", dpi=150)
    plt.close()
    
    print(f"\nTraining complete! Final eval: {log['eval_return'][-1]:.1f} return, {log['eval_winrate'][-1]*100:.1f}% winrate")


if __name__ == "__main__":
    train()