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

# --- Hyperparameters ---
NUM_STEPS = 2048
BATCH_SIZE = 512
LR = 5e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
ENT_COEF = 0.05
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5
UPDATE_EPOCHS = 4
TOTAL_UPDATES = 1000

# Settings
OPPONENT_POOL_SIZE = 5
OPPONENT_UPDATE_FREQ = 20
SHAPING_SCALE = 0
WARMUP_UPDATES = 0


def canonicalize_obs(obs, is_red_agent, debug=False):
    if not is_red_agent:
        return obs
    
    canon = obs.clone()
    canon = torch.flip(canon, dims=[-1])  # Horizontal flip
    
    if debug:
        print(f"  [canonicalize] Input shape: {obs.shape}, dim={obs.dim()}")
        print(f"  [canonicalize] Before swap - ch6 sum: {canon[6].sum().item():.0f}, ch7 sum: {canon[7].sum().item():.0f}")
    
    # Swap capsules: blue (2) <-> red (3)
    # Swap food: blue (6) <-> red (7)
    if canon.dim() == 3:
        temp = canon[2, :, :].clone()
        canon[2, :, :] = canon[3, :, :]
        canon[3, :, :] = temp
        
        temp = canon[6, :, :].clone()
        canon[6, :, :] = canon[7, :, :]
        canon[7, :, :] = temp
    else:  # dim == 4, shape is (1, C, H, W)
        temp = canon[:, 2, :, :].clone()
        canon[:, 2, :, :] = canon[:, 3, :, :]
        canon[:, 3, :, :] = temp
        
        temp = canon[:, 6, :, :].clone()
        canon[:, 6, :, :] = canon[:, 7, :, :]
        canon[:, 7, :, :] = temp
    
    if debug:
        print(f"  [canonicalize] After swap - ch6 sum: {canon[6].sum().item():.0f}, ch7 sum: {canon[7].sum().item():.0f}")
    
    return canon


def canonicalize_action(action, is_red_agent):
    if not is_red_agent:
        return action
    
    if isinstance(action, torch.Tensor):
        mapper = torch.tensor([0, 3, 2, 1, 4], device=action.device)
        return mapper[action]
    else:
        mapper = {0: 0, 1: 3, 2: 2, 3: 1, 4: 4}
        return mapper[action]


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
    
    # Did we eat food? -> No shaping (env handles it)
    if carry_next > carry_curr:
        return 0.0 
    
    # Did we score? -> No shaping (env handles it)
    if carry_curr > 0 and carry_next == 0:
        return 0.0

    target_dist_curr = 0
    target_dist_next = 0
    
    if carry_curr > 0:
        # GO HOME (Left side x=0 in canonical view)
        target_dist_curr = pos_curr[1]
        target_dist_next = pos_next[1]
    else:
        # HUNT FOOD (Channel 7)
        food_ch = obs_curr[7]
        food_locs = (food_ch > 0).nonzero(as_tuple=False).float()
        if len(food_locs) == 0:
            return 0.0
        
        curr_p = torch.tensor(pos_curr).float()
        next_p = torch.tensor(pos_next).float()
        
        target_dist_curr = (food_locs - curr_p).abs().sum(dim=1).min().item()
        target_dist_next = (food_locs - next_p).abs().sum(dim=1).min().item()

    diff = target_dist_curr - target_dist_next
    return diff


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
            nn.Linear(512, action_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(flat_dim * num_agents, 512), nn.ReLU(),
            nn.Linear(512, 1)
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


def debug_obs(env, learner_ids, opponent_ids, play_as_red, step=0):
    """Print compressed debug info about observations."""
    if step != 0:  # Only print on first step of rollout
        return
    
    print(f"\n{'='*60}")
    print(f"DEBUG: Playing as {'RED' if play_as_red else 'BLUE'}")
    print(f"Learner IDs: {learner_ids}, Opponent IDs: {opponent_ids}")
    
    # Agent positions
    positions = [env.game.state.getAgentPosition(i) for i in range(4)]
    print(f"Agent positions (x,y): {positions}")
    
    # Get raw obs for first learner
    raw_obs = env.get_Observation(learner_ids[0]).float()
    
    # Canonicalize with debug
    print(f"\nCalling canonicalize_obs with play_as_red={play_as_red}:")
    canon_obs = canonicalize_obs(raw_obs.unsqueeze(0), play_as_red, debug=True).squeeze(0)
    
    # Channel summary
    channel_names = ['walls', 'agent_loc', 'blue_caps', 'red_caps', 'allies', 'enemies', 'blue_food', 'red_food']
    
    print(f"\nRaw obs (agent {learner_ids[0]}):")
    for i, name in enumerate(channel_names):
        nonzero = (raw_obs[i] > 0).sum().item()
        if nonzero > 0 and nonzero < 20:  # Print positions if sparse
            locs = (raw_obs[i] > 0).nonzero(as_tuple=False).tolist()
            print(f"  {name:12s}: {nonzero:3d} nonzero, locs={locs[:5]}{'...' if len(locs) > 5 else ''}")
        else:
            print(f"  {name:12s}: {nonzero:3d} nonzero")
    
    print(f"\nCanon obs (agent {learner_ids[0]}, play_as_red={play_as_red}):")
    for i, name in enumerate(channel_names):
        nonzero = (canon_obs[i] > 0).sum().item()
        if nonzero > 0 and nonzero < 20:
            locs = (canon_obs[i] > 0).nonzero(as_tuple=False).tolist()
            print(f"  {name:12s}: {nonzero:3d} nonzero, locs={locs[:5]}{'...' if len(locs) > 5 else ''}")
        else:
            print(f"  {name:12s}: {nonzero:3d} nonzero")
    print(f"{'='*60}\n")


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
    
    opponent_pool = deque(maxlen=OPPONENT_POOL_SIZE)
    opponent_pool.append(copy.deepcopy(agent.state_dict()))
    
    all_returns = []
    all_rewards = []
    
    for update in range(1, TOTAL_UPDATES + 1):
        # --- CURRICULUM LOGIC ---
        if update <= WARMUP_UPDATES:
            play_as_red = False
            learner_ids = [1, 3]
            opponent_ids = [0, 2]
            use_self_play_opp = False
        else:
            play_as_red = np.random.rand() > 0.5
            if play_as_red:
                learner_ids = [0, 2]
                opponent_ids = [1, 3]
            else:
                learner_ids = [1, 3]
                opponent_ids = [0, 2]
            use_self_play_opp = True

        # Load Opponent
        if use_self_play_opp:
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
        
        # Debug print on first step only
        debug_obs(env, learner_ids, opponent_ids, play_as_red, step=0)
        
        episode_return = 0
        episode_returns = []
        
        for step in range(NUM_STEPS):
            # 1. Process Learner Obs
            learner_obs_raw = [obs_dict[env.agents[i]].float() for i in learner_ids]
            learner_obs_canon = [canonicalize_obs(o.unsqueeze(0), play_as_red).squeeze(0) for o in learner_obs_raw]
            learner_obs = torch.stack(learner_obs_canon)
            
            # 2. Joint Obs
            joint_obs = torch.cat(learner_obs_canon, dim=0)
            joint_obs_batch = joint_obs.unsqueeze(0).expand(num_agents, -1, -1, -1)
            
            obs_buf[step] = learner_obs
            joint_obs_buf[step] = joint_obs_batch
            
            # 3. Learner Action
            with torch.no_grad():
                all_obs_list = [learner_obs[i:i+1].to(device) for i in range(num_agents)]
                
                actions = []
                log_probs = []
                values = []
                for i in range(num_agents):
                    act, lp, val, _ = agent.get_action_and_value(learner_obs[i:i+1].to(device), all_obs_list)
                    actions.append(act)
                    log_probs.append(lp)
                    values.append(val)
                actions = torch.cat(actions)
                log_probs = torch.cat(log_probs)
                values = torch.cat(values)
            
            action_buf[step] = actions.cpu()
            logprob_buf[step] = log_probs.cpu()
            value_buf[step] = values.cpu()
            
            # 4. Opponent Action
            if not use_self_play_opp:
                opp_actions_env = torch.randint(0, 5, (2,)).tolist()
            else:
                opp_obs_raw = [obs_dict[env.agents[i]].float() for i in opponent_ids]
                opp_obs_canon = [canonicalize_obs(o.unsqueeze(0), not play_as_red).squeeze(0) for o in opp_obs_raw]
                opp_obs = torch.stack(opp_obs_canon)
                with torch.no_grad():
                    opp_list = [opp_obs[i:i+1].to(device) for i in range(num_agents)]
                    o_acts = []
                    for i in range(num_agents):
                        a, _, _, _ = opponent.get_action_and_value(opp_obs[i:i+1].to(device), opp_list)
                        o_acts.append(a)
                    o_acts = torch.cat(o_acts)
                
                opp_actions_env = []
                for i in range(num_agents):
                    opp_actions_env.append(canonicalize_action(o_acts[i], not play_as_red).item())

            # 5. Env Step
            env_actions = {}
            for i, aid in enumerate(learner_ids):
                env_actions[env.agents[aid]] = canonicalize_action(actions[i], play_as_red).item()
            for i, aid in enumerate(opponent_ids):
                env_actions[env.agents[aid]] = opp_actions_env[i]
                
            next_obs_dict, rewards, dones, _ = env.step(env_actions)
            
            # 6. Rewards & Shaping
            next_obs_raw = [next_obs_dict[env.agents[i]].float() for i in learner_ids]
            next_obs_canon = [canonicalize_obs(o.unsqueeze(0), play_as_red).squeeze(0) for o in next_obs_raw]
            
            shaping = []
            for i in range(num_agents):
                s = compute_heuristic_shaping(learner_obs_canon[i], next_obs_canon[i])
                shaping.append(s)
            
            team_reward = sum(rewards[env.agents[i]] for i in learner_ids)
            episode_return += team_reward
            
            for i in range(num_agents):
                reward_buf[step, i] = team_reward + (shaping[i] * SHAPING_SCALE)
            
            done = any(dones.values())
            done_buf[step] = float(done)
            obs_dict = next_obs_dict
            
            if done:
                episode_returns.append(episode_return)
                episode_return = 0
                obs_dict, _ = env.reset()

        # Compute Last Value
        learner_obs_raw = [obs_dict[env.agents[i]].float() for i in learner_ids]
        learner_obs_canon = [canonicalize_obs(o.unsqueeze(0), play_as_red).squeeze(0) for o in learner_obs_raw]
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
                
                pg = -torch.min(norm_adv.to(device)*ratio, norm_adv.to(device)*torch.clamp(ratio, 1-CLIP_EPS, 1+CLIP_EPS)).mean()
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

        if update % OPPONENT_UPDATE_FREQ == 0:
            opponent_pool.append(copy.deepcopy(agent.state_dict()))
        
        # Logging
        mean_reward = reward_buf.sum().item() / num_agents
        mean_ep_return = np.mean(episode_returns) if episode_returns else 0
        all_returns.append(mean_ep_return)
        all_rewards.append(mean_reward)
        
        mode = "Warmup" if update <= WARMUP_UPDATES else ("Red" if play_as_red else "Blue")
        
        print(f"Upd {update:4d} [{mode:6s}] | "
              f"Reward: {mean_reward:8.2f} | "
              f"EpRet: {mean_ep_return:7.2f} | "
              f"Ent: {np.mean(ent_l):.3f} | "
              f"VL: {np.mean(v_l):.4f} | "
              f"PG: {np.mean(pg_l):.4f} | "
              f"Clip: {np.mean(clip_l):.3f}")
        
        # Save periodically
        if update % 100 == 0:
            torch.save(agent.state_dict(), f"mappo_{update}.pt")

    torch.save(agent.state_dict(), "mappo_final.pt")
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(all_rewards)
    ax1.set_xlabel("Update")
    ax1.set_ylabel("Rollout Reward")
    ax1.set_title("Reward per Update")
    
    ax2.plot(all_returns)
    ax2.set_xlabel("Update")
    ax2.set_ylabel("Episode Return")
    ax2.set_title("Mean Episode Return")
    
    plt.tight_layout()
    plt.savefig("mappo_curves.png")
    plt.close()


if __name__ == "__main__":
    train()