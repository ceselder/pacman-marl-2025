import numpy as np
import math
import torch
import torch.nn as nn
from torch.distributions import Categorical
from gymPacMan import gymPacMan_parallel_env
from collections import deque
import copy
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============================================
# HYPERPARAMETERS
# ============================================
NUM_STEPS = 2048
BATCH_SIZE = 256
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.15
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5
UPDATE_EPOCHS = 4
TOTAL_UPDATES = 2000

# Annealed hyperparameters
LR_START = 3e-4
LR_END = 5e-5
ENT_COEF_START = 0.020
ENT_COEF_END = 0.004

# Settings
OPPONENT_POOL_SIZE = 100 
OPPONENT_UPDATE_FREQ = 10 
SHAPING_SCALE = 0.1
EVAL_FREQ = 50
EVAL_EPISODES = 10

# Teams
EASY_TEAMS = ['randomTeam']
MEDIUM_TEAMS = ['MCTSTeam', 'heuristicTeam']
HARD_TEAMS = ['baselineTeam', 'AstarTeam', 'approxQTeam']
BENCH_TEAMS = ['AstarTeam', 'approxQTeam', 'baselineTeam', 'MCTSTeam']

# Checkpoint
LOAD_CHECKPOINT = None
START_UPDATES = 0

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Helper to initialize layers for RL stability."""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class ViTBody(nn.Module):
    def __init__(self, obs_shape, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        c, h, w = obs_shape
        self.num_patches = h * w
        
        # 1. The "Hybrid" Stem (CNN)
        self.feature_extractor = nn.Sequential(
            layer_init(nn.Conv2d(c, 32, kernel_size=3, padding=1)),
            nn.GELU(),
            layer_init(nn.Conv2d(32, d_model, kernel_size=3, padding=1)),
            nn.GELU(),
        )
        
        # 2. Learned Position Embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, d_model))
        
        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model * 4, 
            dropout=0.0, 
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Init position embeds
        nn.init.normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        B, C, H, W = x.shape
        
        # 1. Extract Local Features (CNN)
        x = self.feature_extractor(x) # (B, d_model, H, W)
        
        # 2. Flatten spatial dims
        x = x.flatten(2).transpose(1, 2)
        
        # 3. Add Position Embeddings
        x = x + self.pos_embed
        
        # 4. Transformer (Global Mixing)
        x = self.transformer(x)
        
        # 5. Global Average Pooling
        x = x.mean(dim=1)
        
        return x

class MAPPOAgent(nn.Module):
    def __init__(self, obs_shape, action_dim, num_agents):
        super().__init__()
        
        d_model = 64
        nhead = 4
        num_layers = 2
        
        self.actor_body = ViTBody(obs_shape, d_model, nhead, num_layers)
        self.actor_head = layer_init(nn.Linear(d_model, action_dim), std=0.01)

        self.critic_body = ViTBody(obs_shape, d_model, nhead, num_layers)
        self.critic_head = layer_init(nn.Linear(d_model, 1), std=1.0)

    def get_value(self, state):
        hidden = self.critic_body(state)
        return self.critic_head(hidden).squeeze(-1)

    # === CHANGED: Added action_mask argument ===
    def get_action_and_value(self, x, state=None, action=None, action_mask=None):
        # 1. Actor
        actor_hidden = self.actor_body(x)
        logits = self.actor_head(actor_hidden)
        
        # --- MASKING LOGIC ---
        if action_mask is not None:
            # Set illegal actions to negative infinity
            logits = logits.masked_fill(action_mask == 0, -1e9)
        # ---------------------

        probs = Categorical(logits=logits)
        
        if action is None:
            action = probs.sample()
            
        # 2. Critic
        if state is None: state = x
        value = self.get_value(state)
        
        return action, probs.log_prob(action), value, probs.entropy()

    def get_deterministic_action(self, x):
        hidden = self.actor_body(x)
        logits = self.actor_head(hidden)
        return torch.argmax(logits, dim=1)

    # === CHANGED: Added b_masks argument ===
    def evaluate(self, b_obs, b_merged_obs, b_action, num_agents, obs_shape, b_masks=None):
        """Called during PPO update step"""
        # Actor pass
        actor_hidden = self.actor_body(b_obs)
        logits = self.actor_head(actor_hidden)
        
        # --- MASKING LOGIC ---
        if b_masks is not None:
            logits = logits.masked_fill(b_masks == 0, -1e9)
        # ---------------------
        
        probs = Categorical(logits=logits)
        log_probs = probs.log_prob(b_action)
        entropy = probs.entropy()
        
        # Critic pass
        critic_hidden = self.critic_body(b_merged_obs)
        values = self.critic_head(critic_hidden).squeeze(-1)
        
        return values, log_probs, entropy


def canonicalize_obs(obs, is_red_agent):
    if not is_red_agent:
        return obs
    is_batched = obs.dim() == 4
    if not is_batched: obs = obs.unsqueeze(0)
    canon = torch.flip(obs.clone(), dims=[-1])
    canon[:, [2, 3], :, :] = canon[:, [3, 2], :, :]
    canon[:, [6, 7], :, :] = canon[:, [7, 6], :, :]
    if not is_batched: canon = canon.squeeze(0)
    return canon


def canonicalize_action(action, is_red_agent):
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
    living_punishment = -0.15 
    dist_moved = abs(pos_curr[0] - pos_next[0]) + abs(pos_curr[1] - pos_next[1])
    if dist_moved > 1.5:
        return living_punishment - 0.5 - (0.25 * carry_curr)
    if dist_moved < 0.1:
        return living_punishment - 0.075 
    return living_punishment


def merge_obs_for_critic(obs_list):
    merged = obs_list[0].clone()
    team_locs = torch.zeros_like(merged[1])
    for obs in obs_list:
        team_locs = torch.maximum(team_locs, obs[1])
    merged[1] = team_locs
    merged[4] = torch.zeros_like(merged[4])
    return merged


def compute_gae(rewards, values, dones, last_value, gamma):
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
        delta = rewards[t] + gamma * next_val * next_nonterm - values[t]
        advantages[t] = lastgaelam = delta + gamma * GAE_LAMBDA * next_nonterm * lastgaelam
    return advantages, advantages + values


def evaluate_vs_bots(agent, num_episodes=20):
    agent.eval()
    returns = []
    wins = 0
    learner_ids = [1, 3]
    
    for i in range(num_episodes):
        opp_name = BENCH_TEAMS[i % len(BENCH_TEAMS)]
        eval_env = gymPacMan_parallel_env(
            layout_file='layouts/bloxCapture.lay',
            display=False,
            reward_forLegalAction=True,
            defenceReward=True,
            length=300,
            enemieName=opp_name,
            self_play=False
        )
        obs_dict, info = eval_env.reset()
        episode_return = 0
        done = False
        while not done:
            learner_obs = torch.stack([obs_dict[eval_env.agents[i]].float() for i in learner_ids])
            with torch.no_grad():
                actions = agent.get_deterministic_action(learner_obs.to(device)).cpu()
            env_actions = {}
            for i, aid in enumerate(learner_ids):
                env_actions[eval_env.agents[aid]] = actions[i].item()
            obs_dict, rewards, dones, info = eval_env.step(env_actions)
            episode_return += sum(rewards[eval_env.agents[i]] for i in learner_ids)
            done = any(dones.values())
        returns.append(episode_return)
        final_score = eval_env.game.state.data.score
        if final_score < 0: wins += 1
    
    agent.train()
    return np.mean(returns), np.std(returns), wins / num_episodes


def train():
    env_selfplay = gymPacMan_parallel_env(layout_file='layouts/bloxCapture.lay', display=False, reward_forLegalAction=True, defenceReward=True, length=300, enemieName='randomTeam', self_play=True)
    env_bot = gymPacMan_parallel_env(layout_file='layouts/bloxCapture.lay', display=False, reward_forLegalAction=True, defenceReward=True, length=300, enemieName='randomTeam', self_play=False)
    
    obs_shape = env_selfplay.get_Observation(0).shape
    num_agents = 2
    
    agent = MAPPOAgent(obs_shape, 5, num_agents).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=LR_START, eps=1e-5)
    
    if LOAD_CHECKPOINT is not None:
        print(f"Loading checkpoint: {LOAD_CHECKPOINT}")
        agent.load_state_dict(torch.load(LOAD_CHECKPOINT, map_location=device))
    
    opponent_pool = deque(maxlen=OPPONENT_POOL_SIZE)
    opponent_pool.append(copy.deepcopy(agent.state_dict()))
    
    log = {
        'update': [], 'reward': [], 'ep_return': [], 'eval_return': [], 'eval_winrate': [],
        'train_winrate': [], 'entropy': [], 'clip_frac': [], 'pg_loss': [], 'v_loss': [],
        'lr': [], 'ent_coef': []
    }
    
    for update in range(START_UPDATES, TOTAL_UPDATES + 1):
        progress = update / TOTAL_UPDATES
        lr = LR_START - (LR_START - LR_END) * progress
        ent_coef = ENT_COEF_START - (ENT_COEF_START - ENT_COEF_END) * progress
        for param_group in optimizer.param_groups: param_group['lr'] = lr
        
        if update <= 200:
            use_bot_opponent = True
            play_as_red = False
            opp_name = np.random.choice(EASY_TEAMS)
            env = env_bot
            env.reset(enemieName=opp_name)
        elif update <= 1000:
            use_bot_opponent = True
            play_as_red = False
            opp_name = np.random.choice(EASY_TEAMS + MEDIUM_TEAMS + (HARD_TEAMS * 5))
            env = env_bot
            env.reset(enemieName=opp_name)
        else:
            rand_val = np.random.rand()
            if rand_val < 0.50:
                use_bot_opponent = False
                play_as_red = np.random.rand() > 0.5 
                opp_name = "Self(Curr)"
                env = env_selfplay
                opponent = MAPPOAgent(obs_shape, 5, num_agents).to(device)
                opponent.load_state_dict(agent.state_dict())
                opponent.eval()
            elif rand_val < 0.70:
                use_bot_opponent = False
                play_as_red = np.random.rand() > 0.5
                opp_name = "Self(Old)"
                env = env_selfplay
                opponent = MAPPOAgent(obs_shape, 5, num_agents).to(device)
                opponent.load_state_dict(opponent_pool[np.random.randint(len(opponent_pool))])
                opponent.eval()
            else:
                use_bot_opponent = True
                play_as_red = False
                opp_name = np.random.choice((HARD_TEAMS * 5) + MEDIUM_TEAMS + EASY_TEAMS)
                env = env_bot
                env.reset(enemieName=opp_name)
        
        if play_as_red:
            learner_ids = [0, 2]
            opponent_ids = [1, 3]
        else:
            learner_ids = [1, 3]
            opponent_ids = [0, 2]
        
        batch_wins = []

        obs_buf = torch.zeros(NUM_STEPS, num_agents, *obs_shape)
        merged_obs_buf = torch.zeros(NUM_STEPS, num_agents, *obs_shape)
        action_buf = torch.zeros(NUM_STEPS, num_agents, dtype=torch.long)
        logprob_buf = torch.zeros(NUM_STEPS, num_agents)
        reward_buf = torch.zeros(NUM_STEPS, num_agents)
        done_buf = torch.zeros(NUM_STEPS, num_agents)
        value_buf = torch.zeros(NUM_STEPS, num_agents)
        # === CHANGED: Added mask buffer ===
        mask_buf = torch.zeros(NUM_STEPS, num_agents, 5) 
        
        obs_dict, info = env.reset()
        episode_return = 0
        episode_returns = []
        
        for step in range(NUM_STEPS):
            learner_obs_raw = [obs_dict[env.agents[i]].float() for i in learner_ids]
            learner_obs_canon = [canonicalize_obs(o, play_as_red) for o in learner_obs_raw]
            learner_obs = torch.stack(learner_obs_canon)
            
            merged_obs = merge_obs_for_critic(learner_obs_canon)
            merged_obs_batch = merged_obs.unsqueeze(0).expand(num_agents, -1, -1, -1)

            obs_buf[step] = learner_obs
            merged_obs_buf[step] = merged_obs_batch

            # === CHANGED: Mask Extraction Logic ===
            current_masks = torch.zeros(num_agents, 5).to(device)
            # info contains 'legal_actions' for the *current* state (from reset or step)
            if 'legal_actions' in info:
                for idx, agent_idx in enumerate(learner_ids):
                    agent_obj = env.agents[agent_idx]
                    legal_moves = info['legal_actions'][agent_obj]
                    current_masks[idx, legal_moves] = 1.0
            else:
                current_masks.fill_(1.0)
            
            mask_buf[step] = current_masks.cpu()
            # ======================================

            with torch.no_grad():
                actions, log_probs, values = [], [], []
                for i in range(num_agents):
                    local_obs = learner_obs[i:i+1].to(device)
                    global_obs = merged_obs_batch[i].unsqueeze(0).to(device)
                    # === CHANGED: Pass local mask ===
                    local_mask = current_masks[i:i+1]
                    
                    act, lp, val, _ = agent.get_action_and_value(
                        local_obs, 
                        global_obs,
                        action_mask=local_mask 
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
            
            env_actions = {}
            for i, aid in enumerate(learner_ids):
                env_actions[env.agents[aid]] = canonicalize_action(actions[i], play_as_red).item()
            
            if not use_bot_opponent:
                opp_obs_raw = [obs_dict[env.agents[i]].float() for i in opponent_ids]
                opp_obs_canon = [canonicalize_obs(o, not play_as_red) for o in opp_obs_raw]
                opp_obs = torch.stack(opp_obs_canon)
                with torch.no_grad():
                    opp_list = [opp_obs[i:i+1].to(device) for i in range(num_agents)]
                    opp_actions = []
                    for i in range(num_agents):
                        # Note: Opponent masking is complex because we need their legal moves, 
                        # but standard eval usually ignores opponent masking or assumes valid returns.
                        # For simplicity/speed we let opponent run unmasked or random-valid if it hits wall.
                        a, _, _, _ = opponent.get_action_and_value(opp_obs[i:i+1].to(device), opp_list)
                        opp_actions.append(a)
                    opp_actions = torch.cat(opp_actions)
                for i, aid in enumerate(opponent_ids):
                    env_actions[env.agents[aid]] = canonicalize_action(opp_actions[i], not play_as_red).item()
            
            next_obs_dict, rewards, dones, info = env.step(env_actions)
            
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
                final_score = env.game.state.data.score
                if play_as_red: is_win = 1 if final_score > 0 else 0
                else: is_win = 1 if final_score < 0 else 0
                batch_wins.append(is_win)
                episode_returns.append(episode_return)
                episode_return = 0
                obs_dict, info = env.reset()

        learner_obs_raw = [obs_dict[env.agents[i]].float() for i in learner_ids]
        learner_obs_canon = [canonicalize_obs(o, play_as_red) for o in learner_obs_raw]
        final_merged_obs = merge_obs_for_critic(learner_obs_canon).unsqueeze(0).to(device)
        
        with torch.no_grad():
            last_value = agent.get_value(final_merged_obs).cpu().item()

        advantages = torch.zeros_like(reward_buf)
        returns = torch.zeros_like(reward_buf)
        for i in range(num_agents):
            adv, ret = compute_gae(reward_buf[:, i], value_buf[:, i], done_buf[:, i], last_value, GAMMA)
            advantages[:, i] = adv
            returns[:, i] = ret
            
        b_obs = obs_buf.reshape(-1, *obs_shape)
        b_merged = merged_obs_buf.reshape(-1, *obs_shape)
        b_act = action_buf.reshape(-1)
        b_logp = logprob_buf.reshape(-1)
        b_adv = advantages.reshape(-1)
        b_ret = returns.reshape(-1)
        # === CHANGED: Flatten masks ===
        b_masks = mask_buf.reshape(-1, 5)
        
        inds = np.arange(NUM_STEPS * num_agents)
        pg_l, v_l, ent_l, clip_l = [], [], [], []
        
        for _ in range(UPDATE_EPOCHS):
            np.random.shuffle(inds)
            for start in range(0, len(inds), BATCH_SIZE):
                mb = inds[start:start+BATCH_SIZE]
                
                vals, lps, ent = agent.evaluate(
                    b_obs[mb].to(device), 
                    b_merged[mb].to(device), 
                    b_act[mb].to(device),
                    num_agents, obs_shape,
                    b_masks=b_masks[mb].to(device) # === CHANGED: Pass masks ===
                )
                
                norm_adv = (b_adv[mb] - b_adv[mb].mean()) / (b_adv[mb].std() + 1e-8)
                ratio = (lps - b_logp[mb].to(device)).exp()
                
                pg = -torch.min(
                    norm_adv.to(device) * ratio,
                    norm_adv.to(device) * torch.clamp(ratio, 1-CLIP_EPS, 1+CLIP_EPS)
                ).mean()
                vl = 0.5 * ((vals - b_ret[mb].to(device))**2).mean()
                loss = pg + VF_COEF*vl - ent_coef*ent.mean()
                
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
        
        if len(batch_wins) > 0: current_win_rate = np.mean(batch_wins)
        else: current_win_rate = log['train_winrate'][-1] if log['train_winrate'] else 0.0
        log['train_winrate'].append(current_win_rate)

        if update > 0 and update % EVAL_FREQ == 0:
            eval_ret, eval_std, eval_wr = evaluate_vs_bots(agent, EVAL_EPISODES)
            log['eval_return'].append(eval_ret)
            log['eval_winrate'].append(eval_wr)
        else:
            log['eval_return'].append(log['eval_return'][-1] if log['eval_return'] else 0)
            log['eval_winrate'].append(log['eval_winrate'][-1] if log['eval_winrate'] else 0)
        
        mean_reward = reward_buf.sum().item() / num_agents
        mean_ep_return = np.mean(episode_returns) if episode_returns else 0
        
        log['update'].append(update)
        log['reward'].append(mean_reward)
        log['ep_return'].append(mean_ep_return)
        log['entropy'].append(np.mean(ent_l))
        log['clip_frac'].append(np.mean(clip_l))
        log['pg_loss'].append(np.mean(pg_l))
        log['v_loss'].append(np.mean(v_l))
        log['lr'].append(lr)
        log['ent_coef'].append(ent_coef)
        
        side = "Red " if play_as_red else "Blue"
        eval_str = f"| Eval(MCTS): {log['eval_return'][-1]:6.1f} ({log['eval_winrate'][-1]*100:4.1f}%)" if update % EVAL_FREQ == 0 else ""
        print(f"Upd {update:4d} [{side}|{opp_name:10s}] | "
              f"Win: {current_win_rate*100:5.1f}% | "
              f"Rew: {mean_reward:7.1f} | "
              f"EpRet: {mean_ep_return:6.1f} | "
              f"Ent: {np.mean(ent_l):.3f} | "
              f"Clip: {np.mean(clip_l):.3f} | "
              f"LR: {lr:.2e} "
              f"{eval_str}")
        
        if update % 250 == 0:
            torch.save(agent.state_dict(), f"mappo_resnet_{update}.pt")

    torch.save(agent.state_dict(), "mappo_resnet_final.pt")
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes[0, 0].plot(log['update'], log['reward']); axes[0, 0].set_title('Rollout Reward'); axes[0, 0].set_xlabel('Update')
    axes[0, 1].plot(log['update'], log['train_winrate'], color='green'); axes[0, 1].set_title('Training Win Rate'); axes[0, 1].set_ylabel('Win Rate (0-1)'); axes[0, 1].set_xlabel('Update')
    axes[0, 2].plot(log['update'], log['eval_return']); axes[0, 2].set_title('Eval vs MCTS'); axes[0, 2].set_xlabel('Update')
    ax2 = axes[0, 2].twinx(); ax2.plot(log['update'], [w*100 for w in log['eval_winrate']], 'r-', alpha=0.5); ax2.set_ylabel('Win Rate %', color='r')
    axes[1, 0].plot(log['update'], log['entropy']); axes[1, 0].set_title('Entropy'); axes[1, 0].axhline(y=0.5, color='r', linestyle='--', alpha=0.5); axes[1, 0].set_xlabel('Update')
    axes[1, 1].plot(log['update'], log['clip_frac']); axes[1, 1].set_title('Clip Fraction'); axes[1, 1].axhline(y=0.2, color='r', linestyle='--', alpha=0.5); axes[1, 1].set_xlabel('Update')
    axes[1, 2].plot(log['update'], log['v_loss']); axes[1, 2].set_title('Value Loss'); axes[1, 2].set_xlabel('Update')
    axes[2, 0].plot(log['update'], log['lr']); axes[2, 0].set_title('Learning Rate'); axes[2, 0].set_xlabel('Update')
    axes[2, 1].plot(log['update'], log['ent_coef']); axes[2, 1].set_title('Entropy Coef'); axes[2, 1].set_xlabel('Update')
    axes[2, 2].plot(log['update'], log['pg_loss']); axes[2, 2].set_title('Policy Loss'); axes[2, 2].set_xlabel('Update')
    plt.tight_layout(); plt.savefig("mappo_resnet_training.png", dpi=150); plt.close()
    
    print(f"\n{'='*60}\nFinal eval (MCTS): {log['eval_return'][-1]:.1f} return, {log['eval_winrate'][-1]*100:.1f}% winrate\n{'='*60}")

if __name__ == "__main__":
    train()