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
TOTAL_UPDATES = 1000

# === LEARNING RATES ===
LR_START = 2e-4
LR_END = 5e-5
ENT_COEF_START = 0.1 
ENT_COEF_END = 0.005

# Settings
OPPONENT_POOL_SIZE = 100 
OPPONENT_UPDATE_FREQ = 25 
SHAPING_SCALE = 0.25 
EVAL_FREQ = 50
EVAL_EPISODES = 10

# === CURRICULUM TEAMS ===
PHASE1_TEAMS = ['baselineTeam', 'randomTeam']
PHASE2_TEAMS = ['AstarTeam', 'approxQTeam', 'randomTeam']
SANITY_TEAMS = ['AstarTeam', 'approxQTeam', 'randomTeam']
HARD_TEAM = 'MCTSTeam'

LOAD_CHECKPOINT = None

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, stride=1)
        self.gn1 = nn.GroupNorm(4, channels) 
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, stride=1)
        self.gn2 = nn.GroupNorm(4, channels)
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.gn1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.gn2(out)
        out += residual
        out = self.act(out)
        return out

# ==============================================================================
# === NEW: HYBRID ViT-CNN AGENT ===
# ==============================================================================
class ViTMAPPOAgent(nn.Module):
    def __init__(self, obs_shape, action_dim, num_agents=2):
        super().__init__()
        self.obs_shape = obs_shape
        
        # 1. CNN Front-End (Feature Extraction)
        # We keep the ResNet structure but add a stride at the end to create "Tokens"
        C = 32 
        self.cnn_encoder = nn.Sequential(
            nn.Conv2d(obs_shape[0], C, kernel_size=3, padding=1, stride=1),
            nn.GELU(),
            ResidualBlock(C),
            ResidualBlock(C),
            nn.Conv2d(C, C, kernel_size=3, padding=1, stride=1), 
            nn.GELU()
        )
        
        # Calculate embedding dim
        with torch.no_grad():
            dummy = torch.zeros(1, *obs_shape)
            cnn_out = self.cnn_encoder(dummy)
            # cnn_out shape: [1, 32, 10, 10]
            self.n_tokens = cnn_out.shape[2] * cnn_out.shape[3] # 100
            self.embed_dim = cnn_out.shape[1] # 32
        
        # Project to a larger dimension for the Transformer
        self.d_model = 128
        self.projection = nn.Linear(self.embed_dim, self.d_model)
        
        # 2. Positional Embeddings (Learnable)
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_tokens, self.d_model) * 0.02)
        
        # 3. Transformer Encoder
        # norm_first=True is CRITICAL for RL stability
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, 
            nhead=4, 
            dim_feedforward=512, 
            dropout=0.0, 
            activation="gelu",
            batch_first=True,
            norm_first=True 
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # 4. Heads
        self.actor = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, action_dim)
        )

        self.critic = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, 1)
        )
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if module.bias is not None:
                module.bias.data.fill_(0.0)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward_backbone(self, x):
        # 1. CNN Extract
        features = self.cnn_encoder(x) # [B, 32, 10, 10]
        
        # 2. Flatten spatial dims to make tokens
        # [B, 32, 10, 10] -> [B, 32, 100] -> [B, 100, 32]
        B, C, H, W = features.shape
        x = features.flatten(2).transpose(1, 2) 
        
        # 3. Project & Add Position
        x = self.projection(x) # [B, 100, 128]
        x = x + self.pos_embed
        
        # 4. Transformer
        x = self.transformer(x) # [B, 100, 128]
        
        # 5. Global Average Pooling (Aggregating all tokens)
        # We crush the 100 tokens into 1 summary vector
        x = x.mean(dim=1) # [B, 128]
        return x

    def get_action_and_value(self, obs, all_obs_list):
        # Actor
        h = self.forward_backbone(obs)
        logits = self.actor(h)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        # Critic (Centralized)
        merged = merge_obs_for_critic([o.squeeze(0) for o in all_obs_list]).unsqueeze(0)
        critic_feat = self.forward_backbone(merged.to(obs.device))
        value = self.critic(critic_feat).squeeze(-1)
        
        return action, log_prob, value, dist.entropy()

    def get_value(self, all_obs_list):
        if all_obs_list[0].dim() == 4:
            merged = merge_obs_for_critic([o.squeeze(0) for o in all_obs_list]).unsqueeze(0)
        else:
            merged = merge_obs_for_critic(all_obs_list).unsqueeze(0)
        
        critic_feat = self.forward_backbone(merged.to(all_obs_list[0].device))
        return self.critic(critic_feat).squeeze(-1)

    def evaluate(self, obs, merged_obs, action, num_agents, obs_shape):
        # Actor
        h = self.forward_backbone(obs)
        logits = self.actor(h)
        dist = Categorical(logits=logits)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        # Critic
        critic_feat = self.forward_backbone(merged_obs)
        value = self.critic(critic_feat).squeeze(-1)
        return value, log_prob, entropy
    
    def get_deterministic_action(self, obs):
        with torch.no_grad():
            h = self.forward_backbone(obs)
            logits = self.actor(h)
            return logits.argmax(dim=-1)

def canonicalize_obs(obs, is_red_agent):
    if not is_red_agent:
        return obs
    
    is_batched = obs.dim() == 4
    if not is_batched:
        obs = obs.unsqueeze(0)
    
    canon = torch.flip(obs.clone(), dims=[-1])
    canon[:, [2, 3], :, :] = canon[:, [3, 2], :, :]
    canon[:, [6, 7], :, :] = canon[:, [7, 6], :, :]
    
    if not is_batched:
        canon = canon.squeeze(0)
    return canon

def canonicalize_action(action, is_red_agent):
    if not is_red_agent:
        return action
    if isinstance(action, torch.Tensor):
        mapper = torch.tensor([0, 3, 2, 1, 4], device=action.device)
        return mapper[action]
    return {0: 0, 1: 3, 2: 2, 3: 1, 4: 4}[action]

def detect_channels(obs_canon):
    agent_idx = 1 
    food_idx = 0 
    
    ch1_pixels = (obs_canon[1] > 0).sum().item()
    if ch1_pixels <= 2 and ch1_pixels > 0:
        agent_idx = 1
    else:
        min_px = 999
        for c in range(obs_canon.shape[0]):
            px = (obs_canon[c] > 0).sum().item()
            if 0 < px < 5:
                agent_idx = c
                break

    for c in range(obs_canon.shape[0]):
        if c == agent_idx: continue
        px = (obs_canon[c] > 0).sum().item()
        if 10 < px < 150: 
            food_idx = c
            break
            
    return agent_idx, food_idx

def get_agent_state(obs_canon, agent_idx):
    agent_ch = obs_canon[agent_idx]
    locs = (agent_ch > 0).nonzero(as_tuple=False)
    
    if len(locs) == 0:
        return None, 0
    y, x = locs[0][0].item(), locs[0][1].item()
    val = agent_ch[y, x].item()
    carrying = val - 1 if val > 0 else 0 
    return (y, x), carrying

def compute_heuristic_shaping(obs_curr, obs_next):
    agent_idx, food_idx = detect_channels(obs_curr)
    
    pos_curr, carry_curr = get_agent_state(obs_curr, agent_idx)
    pos_next, carry_next = get_agent_state(obs_next, agent_idx)
    
    if pos_curr is None or pos_next is None:
        return 0.0
    
    if carry_next > carry_curr or (carry_curr > 0 and carry_next == 0):
        return 0.0

    dist_moved = abs(pos_curr[0] - pos_next[0]) + abs(pos_curr[1] - pos_next[1])
    
    if dist_moved > 1.5: return -0.5 
    if dist_moved < 0.1: return -0.05
    
    food_ch = obs_curr[food_idx]
    food_locs = (food_ch > 0).nonzero(as_tuple=False).float()
    
    if len(food_locs) == 0:
        return 0.0
        
    curr_p = torch.tensor(pos_curr).float()
    next_p = torch.tensor(pos_next).float()
    
    dist_curr = (food_locs - curr_p).abs().sum(dim=1).min().item()
    dist_next = (food_locs - next_p).abs().sum(dim=1).min().item()

    return (dist_curr - dist_next) * 0.1

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
    opp_name = HARD_TEAM
    
    for _ in range(num_episodes):
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
    env_selfplay = gymPacMan_parallel_env(
        layout_file='layouts/bloxCapture.lay',
        display=False,
        reward_forLegalAction=True,
        defenceReward=True,
        length=300,
        enemieName='randomTeam',
        self_play=True
    )
    env_bot = gymPacMan_parallel_env(
        layout_file='layouts/bloxCapture.lay',
        display=False,
        reward_forLegalAction=True,
        defenceReward=True,
        length=300,
        enemieName='randomTeam',
        self_play=False
    )
    
    obs_shape = env_selfplay.get_Observation(0).shape
    num_agents = 2
    # === SWITCHED TO ViTMAPPOAgent ===
    agent = ViTMAPPOAgent(obs_shape, 5, num_agents).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=LR_START, eps=1e-5)
    
    if LOAD_CHECKPOINT is not None:
        state_dict = torch.load(LOAD_CHECKPOINT, map_location=device)
        agent.load_state_dict(state_dict)
    
    opponent_pool = deque(maxlen=OPPONENT_POOL_SIZE)
    opponent_pool.append(copy.deepcopy(agent.state_dict()))
    
    log = {'update': [], 'reward': [], 'ep_return': [], 'eval_return': [], 'eval_winrate': [], 'train_winrate': [], 'train_score': [], 'entropy': [], 'clip_frac': [], 'pg_loss': [], 'v_loss': [], 'lr': [], 'ent_coef': []}
    
    for update in range(1, TOTAL_UPDATES + 1):
        progress = update / TOTAL_UPDATES
        lr = LR_START - (LR_START - LR_END) * progress
        ent_coef = ENT_COEF_START - (ENT_COEF_START - ENT_COEF_END) * progress
        for param_group in optimizer.param_groups: param_group['lr'] = lr
        
        # === OPPONENT SELECTION ===
        if update <= 200:
            use_bot_opponent = True
            play_as_red = False
            opp_name = np.random.choice(PHASE1_TEAMS)
            env = env_bot
            env.reset(enemieName=opp_name)
        elif update <= 400:
            use_bot_opponent = True
            play_as_red = False
            opp_name = np.random.choice(PHASE2_TEAMS)
            env = env_bot
            env.reset(enemieName=opp_name)
        else:
            rand_val = np.random.rand()
            if rand_val < 0.60:
                use_bot_opponent = False
                play_as_red = np.random.rand() > 0.5 
                opp_name = "Self(Curr)"
                env = env_selfplay
                # === SWITCHED TO ViTMAPPOAgent ===
                opponent = ViTMAPPOAgent(obs_shape, 5, num_agents).to(device)
                opponent.load_state_dict(agent.state_dict())
                opponent.eval()
            elif rand_val < 0.80:
                use_bot_opponent = False
                play_as_red = np.random.rand() > 0.5
                opp_name = "Self(Old)"
                env = env_selfplay
                # === SWITCHED TO ViTMAPPOAgent ===
                opponent = ViTMAPPOAgent(obs_shape, 5, num_agents).to(device)
                opponent.load_state_dict(opponent_pool[np.random.randint(len(opponent_pool))])
                opponent.eval()
            elif rand_val < 0.90:
                use_bot_opponent = True
                play_as_red = False
                opp_name = np.random.choice(SANITY_TEAMS)
                env = env_bot
                env.reset(enemieName=opp_name)
            else:
                use_bot_opponent = True
                play_as_red = False
                opp_name = HARD_TEAM
                env = env_bot
                env.reset(enemieName=opp_name)
        
        if play_as_red:
            learner_ids = [0, 2]
            opponent_ids = [1, 3]
        else:
            learner_ids = [1, 3]
            opponent_ids = [0, 2]
        
        batch_wins = []
        batch_scores = []

        obs_buf = torch.zeros(NUM_STEPS, num_agents, *obs_shape)
        merged_obs_buf = torch.zeros(NUM_STEPS, num_agents, *obs_shape)
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
            
            merged_obs = merge_obs_for_critic(learner_obs_canon)
            merged_obs_batch = merged_obs.unsqueeze(0).expand(num_agents, -1, -1, -1)
            
            obs_buf[step] = learner_obs
            merged_obs_buf[step] = merged_obs_batch
            
            with torch.no_grad():
                all_obs_list = [learner_obs[i:i+1].to(device) for i in range(num_agents)]
                actions, log_probs, values = [], [], []
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
                        a, _, _, _ = opponent.get_action_and_value(opp_obs[i:i+1].to(device), opp_list)
                        opp_actions.append(a)
                    opp_actions = torch.cat(opp_actions)
                for i, aid in enumerate(opponent_ids):
                    env_actions[env.agents[aid]] = canonicalize_action(opp_actions[i], not play_as_red).item()
            
            next_obs_dict, rewards, dones, _ = env.step(env_actions)
            
            next_obs_raw = [next_obs_dict[env.agents[i]].float() for i in learner_ids]
            next_obs_canon = [canonicalize_obs(o, play_as_red) for o in next_obs_raw]
            shaping = [compute_heuristic_shaping(learner_obs_canon[i], next_obs_canon[i]) for i in range(num_agents)]
            
            team_reward = sum(rewards[env.agents[i]] for i in learner_ids)
            episode_return += team_reward
            
            for i in range(num_agents):
                reward_buf[step, i] = team_reward + SHAPING_SCALE * shaping[i]
            
            done = any(dones.values())
            done_buf[step] = float(done)
            obs_dict = next_obs_dict
            
            if done:
                final_score = env.game.state.data.score
                if play_as_red:
                    is_win = 1 if final_score > 0 else 0
                else:
                    is_win = 1 if final_score < 0 else 0
                batch_wins.append(is_win)
                batch_scores.append(final_score)
                
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
            adv, ret = compute_gae(reward_buf[:, i], value_buf[:, i], done_buf[:, i], last_value, GAMMA)
            advantages[:, i] = adv
            returns[:, i] = ret
            
        # PPO Update
        b_obs = obs_buf.reshape(-1, *obs_shape)
        b_merged = merged_obs_buf.reshape(-1, *obs_shape)
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
                vals, lps, ent = agent.evaluate(b_obs[mb].to(device), b_merged[mb].to(device), b_act[mb].to(device), num_agents, obs_shape)
                norm_adv = (b_adv[mb] - b_adv[mb].mean()) / (b_adv[mb].std() + 1e-8)
                ratio = (lps - b_logp[mb].to(device)).exp()
                pg = -torch.min(norm_adv.to(device) * ratio, norm_adv.to(device) * torch.clamp(ratio, 1-CLIP_EPS, 1+CLIP_EPS)).mean()
                vl = 0.5 * ((vals - b_ret[mb].to(device))**2).mean()
                loss = pg + VF_COEF*vl - ent_coef*ent.mean()
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), MAX_GRAD_NORM)
                optimizer.step()
                pg_l.append(pg.item()); v_l.append(vl.item()); ent_l.append(ent.mean().item()); clip_l.append(((ratio - 1).abs() > CLIP_EPS).float().mean().item())

        if update % OPPONENT_UPDATE_FREQ == 0:
            opponent_pool.append(copy.deepcopy(agent.state_dict()))
        
        current_win_rate = np.mean(batch_wins) if batch_wins else (log['train_winrate'][-1] if log['train_winrate'] else 0.0)
        current_avg_score = np.mean(batch_scores) if batch_scores else 0.0
        
        log['train_winrate'].append(current_win_rate)
        log['train_score'].append(current_avg_score)

        if update % EVAL_FREQ == 0:
            eval_ret, eval_std, eval_wr = evaluate_vs_bots(agent, EVAL_EPISODES)
            log['eval_return'].append(eval_ret); log['eval_winrate'].append(eval_wr)
        else:
            log['eval_return'].append(log['eval_return'][-1] if log['eval_return'] else 0)
            log['eval_winrate'].append(log['eval_winrate'][-1] if log['eval_winrate'] else 0)
        
        mean_reward = reward_buf.sum().item() / num_agents
        mean_ep_return = np.mean(episode_returns) if episode_returns else 0
        
        log['update'].append(update); log['reward'].append(mean_reward); log['ep_return'].append(mean_ep_return)
        log['entropy'].append(np.mean(ent_l)); log['clip_frac'].append(np.mean(clip_l)); log['pg_loss'].append(np.mean(pg_l)); log['v_loss'].append(np.mean(v_l)); log['lr'].append(lr); log['ent_coef'].append(ent_coef)
        
        side = "Red " if play_as_red else "Blue"
        eval_str = f"| Eval(MCTS): {log['eval_return'][-1]:6.1f} ({log['eval_winrate'][-1]*100:4.1f}%)" if update % EVAL_FREQ == 0 else ""
        print(f"Upd {update:4d} [{side}|{opp_name:10s}] | Win: {current_win_rate*100:5.1f}% | Score: {current_avg_score:5.1f} | Rew: {mean_reward:7.1f} | EpRet: {mean_ep_return:6.1f} | Ent: {np.mean(ent_l):.3f} | {eval_str}")
        
        if update % 250 == 0: torch.save(agent.state_dict(), f"mappo_resnet_transformer_{update}.pt")

    torch.save(agent.state_dict(), "mappo_resnet_transformer_final.pt")
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes[0, 0].plot(log['update'], log['reward']); axes[0, 0].set_title('Rollout Reward')
    axes[0, 1].plot(log['update'], log['train_winrate'], color='green'); axes[0, 1].set_title('Training Win Rate')
    axes[0, 2].plot(log['update'], log['eval_return']); ax2 = axes[0, 2].twinx(); ax2.plot(log['update'], [w*100 for w in log['eval_winrate']], 'r-', alpha=0.5)
    axes[1, 0].plot(log['update'], log['entropy'])
    axes[1, 1].plot(log['update'], log['clip_frac'])
    axes[1, 2].plot(log['update'], log['v_loss'])
    axes[2, 0].plot(log['update'], log['lr'])
    axes[2, 1].plot(log['update'], log['ent_coef'])
    axes[2, 2].plot(log['update'], log['pg_loss'])
    plt.tight_layout(); plt.savefig("mappo_resnet_training.png", dpi=150); plt.close()

if __name__ == "__main__":
    train()