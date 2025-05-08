import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

# Action definitions (ensure these match your environment's expected format)
ACTION_MOVE_LIST = ['S', 'L', 'R', 'U', 'D']
ACTION_PKG_LIST = ['0', '1', '2']
MOVE_DIM = len(ACTION_MOVE_LIST)
PKG_DIM = len(ACTION_PKG_LIST)

class ActorNetwork(nn.Module):
    """Actor Network for PPO: Outputs logits for move and package actions."""
    def __init__(self, obs_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_move = nn.Linear(hidden_dim, MOVE_DIM)
        self.fc_pkg = nn.Linear(hidden_dim, PKG_DIM)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        move_logits = self.fc_move(x)
        pkg_logits = self.fc_pkg(x)
        return move_logits, pkg_logits

class CriticNetwork(nn.Module):
    """Critic Network for PPO: Outputs the value of a state."""
    def __init__(self, obs_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_val = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        value = self.fc_val(x)
        return value

class PPO_Agent:
    """Proximal Policy Optimization (PPO) Agent."""
    def __init__(self, obs_dim,
                 lr_actor=3e-4, lr_critic=1e-3,
                 gamma=0.99, gae_lambda=0.95, policy_clip=0.2,
                 n_epochs=10, ppo_batch_size=64,
                 entropy_coef=0.01, vf_coef=0.5, max_grad_norm=0.5,
                 device=None):

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.ppo_batch_size = ppo_batch_size # Mini-batch size for PPO updates
        self.entropy_coef = entropy_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor = ActorNetwork(obs_dim).to(self.device)
        self.critic = CriticNetwork(obs_dim).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        # Memory to store one rollout of experiences
        self.memory_states = []
        self.memory_actions_move_idx = []
        self.memory_actions_pkg_idx = []
        self.memory_log_probs_move = []
        self.memory_log_probs_pkg = []
        self.memory_rewards = []
        self.memory_dones = []
        self.memory_values = [] # V(s_t)

    def _clear_memory(self):
        self.memory_states.clear()
        self.memory_actions_move_idx.clear()
        self.memory_actions_pkg_idx.clear()
        self.memory_log_probs_move.clear()
        self.memory_log_probs_pkg.clear()
        self.memory_rewards.clear()
        self.memory_dones.clear()
        self.memory_values.clear()

    def store_transition(self, obs, action_move_idx, action_pkg_idx,
                         log_prob_move, log_prob_pkg, reward, done, value):
        """Store a single transition in memory."""
        self.memory_states.append(obs)
        self.memory_actions_move_idx.append(action_move_idx)
        self.memory_actions_pkg_idx.append(action_pkg_idx)
        self.memory_log_probs_move.append(log_prob_move)
        self.memory_log_probs_pkg.append(log_prob_pkg)
        self.memory_rewards.append(reward)
        self.memory_dones.append(done)
        self.memory_values.append(value)

    def select_action(self, obs_np, explore=True):
        """
        Selects an action for the agent given an observation.
        Returns:
            env_action (tuple): (move_str, pkg_str) for the environment.
            action_move_idx (int): Index of the move action.
            action_pkg_idx (int): Index of the package action.
            log_prob_move (float): Log probability of the move action.
            log_prob_pkg (float): Log probability of the package action.
            value (float): State value V(obs_np) from the critic.
        """
        obs_tensor = torch.tensor(obs_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        with torch.no_grad():
            move_logits, pkg_logits = self.actor(obs_tensor)
            value = self.critic(obs_tensor).item() # Get scalar value

        move_dist = Categorical(logits=move_logits)
        pkg_dist = Categorical(logits=pkg_logits)

        if explore:
            action_move_idx_tensor = move_dist.sample()
            action_pkg_idx_tensor = pkg_dist.sample()
        else:
            action_move_idx_tensor = torch.argmax(move_dist.probs, dim=-1)
            action_pkg_idx_tensor = torch.argmax(pkg_dist.probs, dim=-1)

        log_prob_move = move_dist.log_prob(action_move_idx_tensor).item()
        log_prob_pkg = pkg_dist.log_prob(action_pkg_idx_tensor).item()

        action_move_idx = action_move_idx_tensor.item()
        action_pkg_idx = action_pkg_idx_tensor.item()

        env_action_move = ACTION_MOVE_LIST[action_move_idx]
        env_action_pkg = ACTION_PKG_LIST[action_pkg_idx]

        return (env_action_move, env_action_pkg), \
               action_move_idx, action_pkg_idx, \
               log_prob_move, log_prob_pkg, \
               value

    def learn(self, last_state_value_bootstrap=0.0):
        """
        Update policy and value functions using collected experiences.
        Args:
            last_state_value_bootstrap (float): Value of the state after the last action in the rollout,
                                                 if the episode didn't terminate. Otherwise 0.0.
        """
        if not self.memory_rewards:
            return # Nothing to learn from

        # Convert memory lists to tensors
        states_t = torch.tensor(np.array(self.memory_states), dtype=torch.float32, device=self.device)
        actions_move_idx_t = torch.tensor(self.memory_actions_move_idx, dtype=torch.long, device=self.device)
        actions_pkg_idx_t = torch.tensor(self.memory_actions_pkg_idx, dtype=torch.long, device=self.device)
        old_log_probs_move_t = torch.tensor(self.memory_log_probs_move, dtype=torch.float32, device=self.device)
        old_log_probs_pkg_t = torch.tensor(self.memory_log_probs_pkg, dtype=torch.float32, device=self.device)
        rewards_t = torch.tensor(self.memory_rewards, dtype=torch.float32, device=self.device)
        # dones_t should indicate if s_{t+1} is terminal.
        dones_t = torch.tensor(self.memory_dones, dtype=torch.bool, device=self.device) 
        values_t = torch.tensor(self.memory_values, dtype=torch.float32, device=self.device) # V(s_t)

        # Calculate GAE (Generalized Advantage Estimation)
        advantages = torch.zeros_like(rewards_t, device=self.device)
        gae = 0.0
        for t in reversed(range(len(rewards_t))):
            if dones_t[t]: # If s_{t+1} was terminal
                next_val = 0.0
            else: # s_{t+1} was not terminal
                if t == len(rewards_t) - 1: # If this is the last transition in the rollout
                    next_val = last_state_value_bootstrap
                else:
                    next_val = values_t[t+1] # V(s_{t+1})

            delta = rewards_t[t] + self.gamma * next_val - values_t[t]
            gae = delta + self.gamma * self.gae_lambda * (1.0 - dones_t[t].float()) * gae
            advantages[t] = gae
        
        returns = advantages + values_t # Target for value function V_target = A_GAE + V(s_t)

        # Normalize advantages (optional but often beneficial)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Perform PPO updates for n_epochs
        num_samples = len(states_t)
        indices = np.arange(num_samples)

        for _ in range(self.n_epochs):
            np.random.shuffle(indices)
            for start in range(0, num_samples, self.ppo_batch_size):
                end = start + self.ppo_batch_size
                batch_indices = indices[start:end]

                batch_states = states_t[batch_indices]
                batch_actions_move_idx = actions_move_idx_t[batch_indices]
                batch_actions_pkg_idx = actions_pkg_idx_t[batch_indices]
                batch_old_log_probs_move = old_log_probs_move_t[batch_indices]
                batch_old_log_probs_pkg = old_log_probs_pkg_t[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                # Get new log_probs, values, and entropy from current policy
                move_logits, pkg_logits = self.actor(batch_states)
                current_values_squeezed = self.critic(batch_states).squeeze()

                move_dist = Categorical(logits=move_logits)
                pkg_dist = Categorical(logits=pkg_logits)

                new_log_probs_move = move_dist.log_prob(batch_actions_move_idx)
                new_log_probs_pkg = pkg_dist.log_prob(batch_actions_pkg_idx)
                
                # Combine log_probs for move and package actions (assuming independence)
                batch_old_total_log_probs = batch_old_log_probs_move + batch_old_log_probs_pkg
                batch_new_total_log_probs = new_log_probs_move + new_log_probs_pkg

                # Policy ratio r_t(theta) = pi_theta(a_t|s_t) / pi_theta_old(a_t|s_t)
                ratio = torch.exp(batch_new_total_log_probs - batch_old_total_log_probs)

                # Clipped surrogate objective (Actor loss)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.policy_clip, 1.0 + self.policy_clip) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                # Value loss (Critic loss)
                critic_loss = F.mse_loss(current_values_squeezed, batch_returns)

                # Entropy bonus (to encourage exploration)
                entropy_move = move_dist.entropy().mean()
                entropy_pkg = pkg_dist.entropy().mean()
                entropy_bonus = entropy_move + entropy_pkg 

                # Total Actor loss
                total_actor_loss = actor_loss - self.entropy_coef * entropy_bonus # Maximize entropy -> subtract

                # Update Actor
                self.actor_optimizer.zero_grad()
                total_actor_loss.backward()
                if self.max_grad_norm:
                    nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                # Update Critic
                self.critic_optimizer.zero_grad()
                scaled_critic_loss = self.vf_coef * critic_loss
                scaled_critic_loss.backward()
                if self.max_grad_norm:
                    nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()
        
        self._clear_memory() # Clear memory after updates 