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
ENT_COEF = 0.02         
VF_COEF = 0.5           
MAX_GRAD_NORM = 0.5     
UPDATE_EPOCHS = 5       
TOTAL_UPDATES = 300     

VALUE_HIDDEN_DIM = 512
CRITIC_HIDDEN_DIM = 1024 

# --- REWARD SETTINGS ---
REWARD_SCALE = 1.0        # Real events (eating) are 1.0
SHAPING_SCALE = 1.0       # Multiplier for the distance shaping
STEP_PENALTY = -0.05      # Penalty for existing (Time pressure)

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
        
        self.actor_conv = nn.Sequential(
            nn.Conv2d(obs_shape[0], 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Flatten()
        )
        with torch.no_grad():
            dummy = torch.zeros(1, *obs_shape)
            flat = self.actor_conv(dummy).shape[1]
            
        self.actor = nn.Sequential(
            nn.Linear(flat, VALUE_HIDDEN_DIM), nn.ReLU(),
            nn.Linear(VALUE_HIDDEN_DIM, VALUE_HIDDEN_DIM), nn.ReLU(), 
            nn.Linear(VALUE_HIDDEN_DIM, action_dim)
        )

        critic_ch = obs_shape[0] * num_agents
        self.critic_conv = nn.Sequential(
            nn.Conv2d(critic_ch, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Flatten()
        )
        with torch.no_grad():
            dummy_c = torch.zeros(1, critic_ch, obs_shape[1], obs_shape[2])
            flat_c = self.critic_conv(dummy_c).shape[1]

        self.critic = nn.Sequential(
            nn.Linear(flat_c, CRITIC_HIDDEN_DIM), nn.ReLU(),
            nn.Linear(CRITIC_HIDDEN_DIM, CRITIC_HIDDEN_DIM), nn.ReLU(), 
            nn.Linear(CRITIC_HIDDEN_DIM, 1)
        )

    def get_action(self, x):
        h = self.actor_conv(x)
        logits = self.actor(h)
        dist = Categorical(logits=logits)
        return dist.sample(), dist.log_prob(dist.sample())

    def get_value(self, state):
        h = self.critic_conv(state)
        return self.critic(h).squeeze(-1)

    def evaluate(self, obs, state, action):
        h = self.actor_conv(obs)
        logits = self.actor(h)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(action)
        ent = dist.entropy()
        h_c = self.critic_conv(state)
        vals = self.critic(h_c).squeeze(-1)
        return vals, log_probs, ent

# --- Helper Functions ---
def compute_gae(rewards, values, dones, last_value, last_done):
    T = len(rewards)
    adv = torch.zeros_like(rewards)
    lastgaelam = 0
    for t in reversed(range(T)):
        if t == T - 1:
            next_non_terminal = 1.0 - last_done
            next_values = last_value
        else:
            next_non_terminal = 1.0 - dones[t + 1]
            next_values = values[t + 1]
        delta = rewards[t] + GAMMA * next_values * next_non_terminal - values[t]
        adv[t] = lastgaelam = delta + GAMMA * GAE_LAMBDA * next_non_terminal * lastgaelam
    return adv, adv + values

def process_obs(obs, is_red_team): 
    if not is_red_team: return obs 
    new_obs = obs.clone()
    new_obs[:, 2, :, :] = obs[:, 3, :, :]
    new_obs[:, 3, :, :] = obs[:, 2, :, :]
    new_obs[:, 6, :, :] = obs[:, 7, :, :]
    new_obs[:, 7, :, :] = obs[:, 6, :, :]
    new_obs = torch.flip(new_obs, [3]) # Horizontal Flip
    return new_obs

def invert_actions(actions):
    mapper = torch.tensor([0, 3, 2, 1, 4], device=actions.device)
    return mapper[actions]

def get_shaping_reward(obs, prev_obs):
    """
    Returns +0.1 if closer to target, -0.1 if further.
    Target is Food (if carrying=0) or Home (if carrying>0).
    """
    batch_size, _, _, W = obs.shape
    rewards = torch.zeros(batch_size, device=obs.device)
    obs_cpu = obs.cpu()
    prev_obs_cpu = prev_obs.cpu()
    
    for i in range(batch_size):
        # Find Agent
        curr_loc = (obs_cpu[i, 1] > 0).nonzero()
        prev_loc = (prev_obs_cpu[i, 1] > 0).nonzero()
        if len(curr_loc) == 0 or len(prev_loc) == 0: continue

        c_pos = curr_loc[0].float()
        p_pos = prev_loc[0].float()
        
        # Check Carrying (Channel 1 value is 1 + carrying)
        carrying = obs_cpu[i, 1, int(c_pos[0]), int(c_pos[1])] - 1
        
        curr_dist, prev_dist = 0, 0
        
        if carrying > 0:
            # Go Home (Right side in standardized view)
            curr_dist = W - c_pos[1]
            prev_dist = W - p_pos[1]
        else:
            # Go to Food (Channel 7)
            food = (obs_cpu[i, 7] == 1).nonzero().float()
            if len(food) == 0: continue
            
            curr_dist = (food - c_pos).abs().sum(dim=1).min()
            prev_dist = (food - p_pos).abs().sum(dim=1).min()
            
        # Reward = Improvement in distance * 0.1
        diff = prev_dist - curr_dist
        rewards[i] = diff * 0.1
        
    return rewards.to(device)

def evaluate_vs_random(agent, eval_env, num_episodes=5):
    agent.eval()
    total = []
    ids = [1, 3]
    for _ in range(num_episodes):
        d, _ = eval_env.reset(enemieName='randomTeam')
        obs = torch.stack([d[eval_env.agents[i]] for i in ids]).float()
        p_obs = process_obs(obs, False)
        done, ret = False, 0
        while not done:
            with torch.no_grad(): act, _ = agent.get_action(p_obs.to(device))
            env_act = {eval_env.agents[id]: act[i].item() for i, id in enumerate(ids)}
            nd, rw, dones, _ = eval_env.step(env_act)
            ret += sum(rw[eval_env.agents[i]] for i in ids)
            done = any(dones.values())
            obs = torch.stack([nd[eval_env.agents[i]] for i in ids]).float()
            p_obs = process_obs(obs, False)
        total.append(ret)
    agent.train()
    return np.mean(total)

# --- Main Loop ---
def train_mappo(train_env, eval_env):
    obs_shape = train_env.get_Observation(0).shape
    num_agents = 2
    state_shape = (obs_shape[0]*2, obs_shape[1], obs_shape[2])
    
    agent = MAPPOAgent(obs_shape, 5).to(device)
    opt = torch.optim.Adam(agent.parameters(), lr=LR)
    
    pool = deque(maxlen=OPPONENT_POOL_SIZE)
    pool.append(copy.deepcopy(agent.state_dict()))
    
    log = {'update':[], 'reward':[], 'eval':[], 'ep_ret':[], 'loss':[]}
    ep_rets = []

    def get_opp():
        opp = MAPPOAgent(obs_shape, 5).to(device)
        if np.random.rand() < SELF_PLAY_PROB and len(pool) > 1:
            opp.load_state_dict(np.random.choice(list(pool)[:-1]))
        else: opp.load_state_dict(pool[-1])
        opp.eval()
        return opp

    for update in range(1, TOTAL_UPDATES+1):
        if update <= WARMUP_UPDATES:
            opp_random, red_learner = True, False
            t_ids, o_ids = [1, 3], [0, 2]
        else:
            opp_random = False
            opp = get_opp()
            if np.random.rand() > 0.5: t_ids, o_ids, red_learner = [1, 3], [0, 2], False
            else: t_ids, o_ids, red_learner = [0, 2], [1, 3], True

        # Buffers
        b_obs = torch.zeros(NUM_STEPS, 2, *obs_shape)
        b_state = torch.zeros(NUM_STEPS, 2, *state_shape)
        b_act = torch.zeros(NUM_STEPS, 2, dtype=torch.long)
        b_logp = torch.zeros(NUM_STEPS, 2)
        b_rew = torch.zeros(NUM_STEPS, 2)
        b_done = torch.zeros(NUM_STEPS, 2)
        b_val = torch.zeros(NUM_STEPS, 2)

        train_env.reset()
        raw = torch.stack([train_env.get_Observation(i) for i in t_ids]).float()
        
        # Keep track of previous obs for shaping
        prev_obs_processed = process_obs(raw, red_learner)
        curr_obs_processed = prev_obs_processed
        
        curr_state = torch.stack([torch.cat([curr_obs_processed[0], curr_obs_processed[1]])]*2)
        curr_done = torch.zeros(2)
        cur_ep_ret = 0

        for step in range(NUM_STEPS):
            b_obs[step] = curr_obs_processed
            b_state[step] = curr_state
            b_done[step] = curr_done

            with torch.no_grad():
                act, logp = agent.get_action(curr_obs_processed.to(device))
                val = agent.get_value(curr_state.to(device))
                act, logp, val = act.cpu(), logp.cpu(), val.cpu()

            b_act[step], b_logp[step], b_val[step] = act, logp, val
            
            real_act = invert_actions(act) if red_learner else act
            
            if opp_random: 
                opp_act_real = torch.randint(0, 5, (2,))
            else:
                raw_opp = torch.stack([train_env.get_Observation(i) for i in o_ids]).float()
                p_opp = process_obs(raw_opp, not red_learner).to(device)
                o_act, _ = opp.get_action(p_opp)
                opp_act_real = invert_actions(o_act.cpu()) if (not red_learner) else o_act.cpu()

            env_act = {}
            for i, aid in enumerate(t_ids): env_act[train_env.agents[aid]] = real_act[i].item()
            for i, aid in enumerate(o_ids): env_act[train_env.agents[aid]] = opp_act_real[i].item()

            nd, rw, dones, _ = train_env.step(env_act)
            
            # --- SHAPING LOGIC ---
            # 1. Real Reward
            real_r = sum(rw[train_env.agents[i]] for i in t_ids)
            cur_ep_ret += real_r
            
            # 2. Next Obs
            raw = torch.stack([nd[train_env.agents[i]] for i in t_ids]).float() # New Raw
            next_obs_processed = process_obs(raw, red_learner)
            
            # 3. Shaping Reward (Based on distance delta)
            # Compare curr_obs_processed (t) with next_obs_processed (t+1)
            # Note: My helper uses (Prev - Curr), so we pass (curr, next)
            # Wait, if we moved closer, dist(next) < dist(curr).
            # Diff = dist(curr) - dist(next). Positive.
            shaping = get_shaping_reward(next_obs_processed, curr_obs_processed).cpu()
            
            # 4. Total Reward = Real + Shaping - Penalty
            # Sum shaping across agents? No, usually per agent.
            # But rewards_buf is (NUM_STEPS, 2).
            # We apply the team reward to both, plus individual shaping.
            for i in range(2):
                b_rew[step, i] = (real_r * REWARD_SCALE) + (shaping[i] * SHAPING_SCALE) + STEP_PENALTY

            # Update loop vars
            curr_obs_processed = next_obs_processed
            curr_state = torch.stack([torch.cat([curr_obs_processed[0], curr_obs_processed[1]])]*2)
            
            done = any(dones[train_env.agents[i]] for i in t_ids)
            curr_done = torch.full((2,), float(done))

            if done:
                ep_rets.append(cur_ep_ret)
                cur_ep_ret = 0
                train_env.reset()
                raw = torch.stack([train_env.get_Observation(i) for i in t_ids]).float()
                prev_obs_processed = process_obs(raw, red_learner)
                curr_obs_processed = prev_obs_processed # Reset obs
                curr_state = torch.stack([torch.cat([curr_obs_processed[0], curr_obs_processed[1]])]*2)

        # Update
        with torch.no_grad(): last_val = agent.get_value(curr_state.to(device)).cpu()
        adv, ret = compute_gae(b_rew, b_val, b_done, last_val, curr_done)
        
        flat_inds = np.arange(NUM_STEPS*2)
        flat_obs = b_obs.reshape(-1, *obs_shape)
        flat_st = b_state.reshape(-1, *state_shape)
        flat_act = b_act.reshape(-1)
        flat_logp = b_logp.reshape(-1)
        flat_adv = adv.reshape(-1)
        flat_ret = ret.reshape(-1)

        pg_l, v_l, ents, clips = [], [], [], []
        for _ in range(UPDATE_EPOCHS):
            np.random.shuffle(flat_inds)
            for st in range(0, len(flat_inds), BATCH_SIZE):
                mb = flat_inds[st:st+BATCH_SIZE]
                mb_adv = (flat_adv[mb] - flat_adv[mb].mean()) / (flat_adv[mb].std() + 1e-8)
                
                vals, lps, ent = agent.evaluate(flat_obs[mb].to(device), flat_st[mb].to(device), flat_act[mb].to(device))
                ratio = (lps - flat_logp[mb].to(device)).exp()
                pg = -torch.min(ratio*mb_adv.to(device), torch.clamp(ratio, 1-CLIP_EPS, 1+CLIP_EPS)*mb_adv.to(device)).mean()
                vl = 0.5 * ((vals - flat_ret[mb].to(device))**2).mean()
                loss = pg - ENT_COEF*ent.mean() + VF_COEF*vl
                
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(agent.parameters(), MAX_GRAD_NORM); opt.step()
                pg_l.append(pg.item()); v_l.append(vl.item()); ents.append(ent.mean().item()); 
                clips.append(((ratio-1).abs()>CLIP_EPS).float().mean().item())

        if update % OPPONENT_UPDATE_FREQ == 0: pool.append(copy.deepcopy(agent.state_dict()))
        if update % EVAL_FREQ == 0: log['eval'].append(evaluate_vs_random(agent, eval_env))
        else: log['eval'].append(log['eval'][-1] if log['eval'] else 0)
        
        m_r = b_rew.sum().item()/2
        m_ep = np.mean(ep_rets[-10:]) if ep_rets else 0
        side = "Warm" if update <= WARMUP_UPDATES else ("Red" if red_learner else "Blue")
        print(f"Upd {update:3d} [{side}] | Rwd: {m_r:6.1f} | EpRet: {m_ep:6.2f} | Eval: {log['eval'][-1]:6.2f} | Clip: {np.mean(clips):.3f}")
        
        log['update'].append(update); log['reward'].append(m_r); log['ep_ret'].append(m_ep)

    torch.save(agent.state_dict(), "mappo_pacman_final.pt")
    plt.plot(log['update'], log['ep_ret']); plt.title("Episodic Return"); plt.savefig("curve.png")

if __name__ == "__main__":
    t_env = gymPacMan_parallel_env(layout_file='layouts/tinyCapture.lay', display=False, reward_forLegalAction=True, defenceReward=True, length=300, enemieName='randomTeam', self_play=True, random_layout=False)
    e_env = gymPacMan_parallel_env(layout_file='layouts/tinyCapture.lay', display=False, reward_forLegalAction=True, defenceReward=True, length=300, enemieName='randomTeam', self_play=False, random_layout=False)
    train_mappo(t_env, e_env)