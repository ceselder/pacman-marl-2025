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
# HYPERPARAMETERS - resnet OF ALL RUNS
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
LR_START = 2e-4 #original 2e4
LR_END = 7e-5
ENT_COEF_START = 0.0175 #reduce back if its just for 
ENT_COEF_END = 0.0025

# Settings
OPPONENT_POOL_SIZE = 100 
OPPONENT_UPDATE_FREQ = 10 
SHAPING_SCALE = 0.1
EVAL_FREQ = 50
EVAL_EPISODES = 10

# Phase 1, just learn against randoms, learn to get actual reward.
EASY_TEAMS = ['randomTeam']

# Isolating this one because it takes ages to train on
MEDIUM_TEAMS = ['MCTSTeam', 'heuristicTeam']

# Phase 3 Teams (Hardest), hardest to beat, train on these + self play
HARD_TEAMS = ['baselineTeam', 'AstarTeam', 'approxQTeam']

# Checkpoint
LOAD_CHECKPOINT = None
START_UPDATES = 0

BENCH_TEAMS = ['AstarTeam', 'approxQTeam', 'baselineTeam', 'MCTSTeam']

class PositionalEncoding2D(nn.Module):
    def __init__(self, d_model, max_h=50, max_w=50):
        super().__init__()
        d_model_half = d_model // 2
        
        y_pos = torch.arange(max_h).unsqueeze(1)
        y_den = torch.exp(torch.arange(0, d_model_half, 2) * -(math.log(10000.0) / d_model_half))
        y_enc = torch.zeros(max_h, d_model_half)
        y_enc[:, 0::2] = torch.sin(y_pos * y_den)
        y_enc[:, 1::2] = torch.cos(y_pos * y_den)
        
        x_pos = torch.arange(max_w).unsqueeze(1)
        x_den = torch.exp(torch.arange(0, d_model_half, 2) * -(math.log(10000.0) / d_model_half))
        x_enc = torch.zeros(max_w, d_model_half)
        x_enc[:, 0::2] = torch.sin(x_pos * x_den)
        x_enc[:, 1::2] = torch.cos(x_pos * x_den)
        
        self.register_buffer('y_enc', y_enc)
        self.register_buffer('x_enc', x_enc)

    def forward(self, x):
        B, C, H, W = x.shape
        y_emb = self.y_enc[:H, :].unsqueeze(1).repeat(1, W, 1)
        x_emb = self.x_enc[:W, :].unsqueeze(0).repeat(H, 1, 1)
        pos = torch.cat([y_emb, x_emb], dim=2)
        pos = pos.permute(2, 0, 1).unsqueeze(0).repeat(B, 1, 1, 1)
        return x + pos

class MAPPOAgent(nn.Module):
    def __init__(self, obs_shape, action_dim, num_agents=2):
        super().__init__()
        self.obs_shape = obs_shape
        
        # --- HYPERPARAMETERS ---
        self.d_model = 64     # Increased from 128
        nhead = 4
        num_layers = 2
        dim_ff = 256          # Increased from 512
        
        # Shared Positional Encoder
        self.pos_encoder = PositionalEncoding2D(self.d_model)

        # =====================================================================
        # ACTOR (Transformer)
        # =====================================================================
        self.actor_projector = nn.Sequential(
            nn.Conv2d(obs_shape[0], 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, self.d_model, kernel_size=1)
        )
        
        actor_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, nhead=nhead, dim_feedforward=dim_ff, 
            dropout=0.0, batch_first=False, norm_first=True
        )
        self.actor_transformer = nn.TransformerEncoder(actor_layer, num_layers=num_layers)
        
        self.actor_head = nn.Sequential(
            nn.Linear(self.d_model, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Linear(1024, action_dim)
        )

        # =====================================================================
        # CRITIC (Transformer)
        # =====================================================================
        # Note: We use obs_shape[0] assuming channels are consistent or handled via merging
        self.critic_projector = nn.Sequential(
            nn.Conv2d(obs_shape[0], 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, self.d_model, kernel_size=1)
        )
        
        critic_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, nhead=nhead, dim_feedforward=dim_ff, 
            dropout=0.0, batch_first=False, norm_first=True
        )
        self.critic_transformer = nn.TransformerEncoder(critic_layer, num_layers=num_layers)
        
        self.critic_head = nn.Sequential(
            nn.Linear(self.d_model, 1024),
            nn.GELU(),
            nn.Linear(1024, 1)
        )
        
        self.apply(self._init_weights)
        nn.init.orthogonal_(self.actor_head[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.critic_head[-1].weight, gain=1.0)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def _forward_transformer(self, obs, projector, transformer):
        # 1. Project to d_model
        x = projector(obs)
        
        # 2. Add Positional Encoding
        x = self.pos_encoder(x)
        
        # 3. Reshape: [Batch, Dim, H, W] -> [Sequence, Batch, Dim]
        # (batch_first=False in encoder layer, so Sequence is dim 0)
        x = x.flatten(2).permute(2, 0, 1)
        
        # 4. Transformer
        x = transformer(x)
        
        # 5. Global Average Pooling (over sequence dimension 0)
        x = x.mean(dim=0)
        return x

    def get_action_and_value(self, obs, all_obs_list):
        # --- ACTOR ---
        h_actor = self._forward_transformer(obs, self.actor_projector, self.actor_transformer)
        logits = self.actor_head(h_actor)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        # --- CRITIC ---
        if all_obs_list[0].dim() == 4:
            merged = merge_obs_for_critic([o.squeeze(0) for o in all_obs_list]).unsqueeze(0)
        else:
            merged = merge_obs_for_critic(all_obs_list).unsqueeze(0)
            
        h_critic = self._forward_transformer(merged.to(obs.device), self.critic_projector, self.critic_transformer)
        value = self.critic_head(h_critic).squeeze(-1)
        
        return action, log_prob, value, dist.entropy()

    def get_value(self, all_obs_list):
        if all_obs_list[0].dim() == 4:
            merged = merge_obs_for_critic([o.squeeze(0) for o in all_obs_list]).unsqueeze(0)
        else:
            merged = merge_obs_for_critic(all_obs_list).unsqueeze(0)
        
        h_critic = self._forward_transformer(merged.to(all_obs_list[0].device), self.critic_projector, self.critic_transformer)
        return self.critic_head(h_critic).squeeze(-1)

    def evaluate(self, obs, merged_obs, action, num_agents, obs_shape):
        # Actor
        h_actor = self._forward_transformer(obs, self.actor_projector, self.actor_transformer)
        logits = self.actor_head(h_actor)
        dist = Categorical(logits=logits)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        # Critic
        h_critic = self._forward_transformer(merged_obs, self.critic_projector, self.critic_transformer)
        value = self.critic_head(h_critic).squeeze(-1)
        
        return value, log_prob, entropy
    
    def get_deterministic_action(self, obs):
        with torch.no_grad():
            h_actor = self._forward_transformer(obs, self.actor_projector, self.actor_transformer)
            logits = self.actor_head(h_actor)
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

    living_punishment = -0.15 #if u just punish it for being alive this apparently leads to good things, lets try
    #this ends up being 0.01 remember
    
    #if carry_next > carry_curr or (carry_curr > 0 and carry_next == 0):
    #    return living_punishment + 0.0

    # dying is generally bad, especially if carrying stuff
    dist_moved = abs(pos_curr[0] - pos_next[0]) + abs(pos_curr[1] - pos_next[1])
    if dist_moved > 1.5:
        return living_punishment - 0.5 - (0.25 * carry_curr)
    
    if dist_moved < 0.1: #standing still is generally bad
        return living_punishment - 0.075 

    return living_punishment
    
    # if carry_curr > 0:
    #     dist_curr = pos_curr[1]
    #     dist_next = pos_next[1]
    #     return living_punishment - ((dist_curr - dist_next) * (0.8 + (0.1 * carry_curr)))

    # food_ch = obs_curr[7]
    # food_locs = (food_ch > 0).nonzero(as_tuple=False).float()
    # if len(food_locs) == 0:
    #     return 0.0
    # curr_p = torch.tensor(pos_curr).float()
    # next_p = torch.tensor(pos_next).float()
    # dist_curr = (food_locs - curr_p).abs().sum(dim=1).min().item()
    # dist_next = (food_locs - next_p).abs().sum(dim=1).min().item()

    # return living_punishment - (dist_curr - dist_next)


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
    """Evaluate agent ONLY against MCTSTeam as requested."""
    agent.eval()
    returns = []
    wins = 0
    
    learner_ids = [1, 3] # Playing as Blue
    
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
        # Blue wins if score < 0
        if final_score < 0:
            wins += 1
    
    agent.train()
    return np.mean(returns), np.std(returns), wins / num_episodes


def train():
    # Self-play env
    env_selfplay = gymPacMan_parallel_env(
        layout_file='layouts/bloxCapture.lay',
        display=False,
        reward_forLegalAction=True,
        defenceReward=True,
        length=300,
        enemieName='randomTeam',
        self_play=True
    )
    
    # Bot opponent env (we play as blue, bots are red)
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
    
    agent = MAPPOAgent(obs_shape, 5, num_agents).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=LR_START, eps=1e-5)
    
    if LOAD_CHECKPOINT is not None:
        print(f"Loading checkpoint: {LOAD_CHECKPOINT}")
        state_dict = torch.load(LOAD_CHECKPOINT, map_location=device)
        agent.load_state_dict(state_dict)
    
    # Pool for historical self-play
    opponent_pool = deque(maxlen=OPPONENT_POOL_SIZE)
    # Initialize with current agent
    opponent_pool.append(copy.deepcopy(agent.state_dict()))
    
    log = {
        'update': [], 'reward': [], 'ep_return': [],
        'eval_return': [], 'eval_winrate': [],
        'train_winrate': [], # Winrate against current opponent
        'entropy': [], 'clip_frac': [], 'pg_loss': [], 'v_loss': [],
        'lr': [], 'ent_coef': []
    }
    
    for update in range(START_UPDATES, TOTAL_UPDATES + 1):
        # === ANNEAL HYPERPARAMETERS ===
        progress = update / TOTAL_UPDATES
        lr = LR_START - (LR_START - LR_END) * progress
        ent_coef = ENT_COEF_START - (ENT_COEF_START - ENT_COEF_END) * progress
        
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        if update <= 200: #skip this part since checkpoint training run
            # teach it the game
            use_bot_opponent = True
            play_as_red = False
            opp_name = np.random.choice(EASY_TEAMS)
            env = env_bot
            env.reset(enemieName=opp_name)
            
        elif update <= 1000:

            use_bot_opponent = True
            play_as_red = False
            opp_name = np.random.choice(EASY_TEAMS + MEDIUM_TEAMS + (HARD_TEAMS * 5)) #overrepresent hard teams
            env = env_bot
            env.reset(enemieName=opp_name)
            
        else:
            # PHASE 3: MIXED REGIME
            rand_val = np.random.rand()
            
            if rand_val < 0.50:
                # 40% Self-Play (Current Version)
                use_bot_opponent = False
                play_as_red = np.random.rand() > 0.5 
                opp_name = "Self(Curr)"
                env = env_selfplay
                
                opponent = MAPPOAgent(obs_shape, 5, num_agents).to(device)
                opponent.load_state_dict(agent.state_dict())
                opponent.eval()
                
            elif rand_val < 0.70:
                # 20% Self-Play (Old Version)
                use_bot_opponent = False
                play_as_red = np.random.rand() > 0.5
                opp_name = "Self(Old)"
                env = env_selfplay
                
                opponent = MAPPOAgent(obs_shape, 5, num_agents).to(device)
                opponent.load_state_dict(opponent_pool[np.random.randint(len(opponent_pool))])
                opponent.eval()
                
            else:
                # 30% face against a hard team weighted towards hard team
                use_bot_opponent = True
                play_as_red = False
                #make hard teams very overrepresented
                opp_name = np.random.choice((HARD_TEAMS * 5) + MEDIUM_TEAMS + EASY_TEAMS)
                env = env_bot
                env.reset(enemieName=opp_name)
        
        if play_as_red:
            learner_ids = [0, 2]
            opponent_ids = [1, 3]
        else:
            learner_ids = [1, 3]
            opponent_ids = [0, 2]
        
        # Init win tracking for this update
        batch_wins = []

        # Buffers
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
            
            # === BUILD ENV ACTIONS ===
            env_actions = {}
            for i, aid in enumerate(learner_ids):
                env_actions[env.agents[aid]] = canonicalize_action(actions[i], play_as_red).item()
            
            if use_bot_opponent:
                pass 
            else:
                # Self-play opponent
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
                # === WIN LOGIC ===
                final_score = env.game.state.data.score
                if play_as_red:
                    is_win = 1 if final_score > 0 else 0
                else:
                    is_win = 1 if final_score < 0 else 0
                batch_wins.append(is_win)
                # =================

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
                
                vals, lps, ent = agent.evaluate(
                    b_obs[mb].to(device), 
                    b_merged[mb].to(device), 
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
                loss = pg + VF_COEF*vl - ent_coef*ent.mean()
                
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), MAX_GRAD_NORM)
                optimizer.step()
                
                pg_l.append(pg.item())
                v_l.append(vl.item())
                ent_l.append(ent.mean().item())
                clip_l.append(((ratio - 1).abs() > CLIP_EPS).float().mean().item())

        # Update opponent pool every 25 updates (as requested)
        if update % OPPONENT_UPDATE_FREQ == 0:
            opponent_pool.append(copy.deepcopy(agent.state_dict()))
        
        # Log Winrate
        if len(batch_wins) > 0:
            current_win_rate = np.mean(batch_wins)
        else:
            current_win_rate = log['train_winrate'][-1] if log['train_winrate'] else 0.0
        log['train_winrate'].append(current_win_rate)

        # Evaluation (MCTS Only)
        if update > 0 and update % EVAL_FREQ == 0:
            eval_ret, eval_std, eval_wr = evaluate_vs_bots(agent, EVAL_EPISODES)
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
    
    # Plotting
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    
    axes[0, 0].plot(log['update'], log['reward'])
    axes[0, 0].set_title('Rollout Reward')
    axes[0, 0].set_xlabel('Update')
    
    axes[0, 1].plot(log['update'], log['train_winrate'], color='green', label='Train Win%')
    axes[0, 1].set_title('Training Win Rate (vs Current Opp)')
    axes[0, 1].set_ylabel('Win Rate (0-1)')
    axes[0, 1].set_xlabel('Update')
    
    axes[0, 2].plot(log['update'], log['eval_return'], label='Return')
    axes[0, 2].set_title('Eval vs MCTS')
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
    
    axes[2, 0].plot(log['update'], log['lr'])
    axes[2, 0].set_title('Learning Rate (annealed)')
    axes[2, 0].set_xlabel('Update')
    
    axes[2, 1].plot(log['update'], log['ent_coef'])
    axes[2, 1].set_title('Entropy Coef (annealed)')
    axes[2, 1].set_xlabel('Update')
    
    axes[2, 2].plot(log['update'], log['pg_loss'])
    axes[2, 2].set_title('Policy Loss')
    axes[2, 2].set_xlabel('Update')
    
    plt.tight_layout()
    plt.savefig("mappo_resnet_training.png", dpi=150)
    plt.close()
    
    print(f"\n{'='*60}")
    print(f"Final eval (MCTS): {log['eval_return'][-1]:.1f} return, {log['eval_winrate'][-1]*100:.1f}% winrate")
    print(f"{'='*60}")


if __name__ == "__main__":
    train()