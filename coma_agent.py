import numpy as np
import random
from collections import deque, namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# ----------------------------- CONSTANTS -----------------------------------
action_move = ['S', 'L', 'R', 'U', 'D']
action_pkg = ['0', '1', '2']
MOVE_DIM = len(action_move)
PKG_DIM = len(action_pkg)


def one_hot(index, size):
    x = np.zeros(size, dtype=np.float32)
    x[index] = 1.0
    return x


class Actor(nn.Module):
    """Simple two-head network returning logits for (move, pkg)"""

    def __init__(self, obs_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.out_move = nn.Linear(hidden_dim, MOVE_DIM)
        self.out_pkg = nn.Linear(hidden_dim, PKG_DIM)

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        move_logits = self.out_move(x)
        pkg_logits = self.out_pkg(x)
        return move_logits, pkg_logits


class CentralCritic(nn.Module):
    """Critic estimates Q for global state and joint action"""

    def __init__(self, global_state_dim, joint_action_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(global_state_dim + joint_action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, sa):
        x = F.relu(self.fc1(sa))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.out(x)


class ReplayBuffer:
    def __init__(self, buffer_size=int(5e4), batch_size=512):
        self.memory = deque(maxlen=buffer_size)
        self.experience = namedtuple('Experience', field_names=[
            'state', 'joint_action', 'rewards', 'next_state', 'dones'])
        self.batch_size = batch_size

    def push(self, *args):
        self.memory.append(self.experience(*args))

    def sample(self):
        batch = random.sample(self.memory, k=self.batch_size)
        return batch

    def __len__(self):
        return len(self.memory)


class COMAAgents:
    """Counterfactual Multi-Agent Policy-Gradient (COMA) implementation for the delivery env."""

    def __init__(self, lr_actor=1e-3, lr_critic=1e-3, gamma=0.95, tau=0.01,
                 buffer_size=int(5e4), batch_size=512, device=None):
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.memory = ReplayBuffer(buffer_size, batch_size)

        self.num_agents = 0
        self.obs_dim = None
        self.state_dim = None
        self.joint_action_dim = None
        self.actors = []
        self.target_critic = None
        self.critic = None
        self.actor_opts = []
        self.critic_opt = None

    # ----------------------------------------------------------------------
    @staticmethod
    def encode_action(m_idx, p_idx):
        return np.concatenate([one_hot(m_idx, MOVE_DIM), one_hot(p_idx, PKG_DIM)])

    @staticmethod
    def tuple_env_action(m_idx, p_idx):
        return (action_move[m_idx], action_pkg[p_idx])

    @staticmethod
    def decode_logits(move_logits, pkg_logits, explore=True):
        move_prob = F.softmax(move_logits, dim=-1)
        pkg_prob = F.softmax(pkg_logits, dim=-1)
        if explore:
            m = torch.distributions.Categorical(move_prob).sample()
            p = torch.distributions.Categorical(pkg_prob).sample()
        else:
            m = move_prob.argmax(dim=-1)
            p = pkg_prob.argmax(dim=-1)
        return m.item(), p.item(), move_prob.squeeze(0), pkg_prob.squeeze(0)

    # very simple encoding of global state: concat each robot small obs
    def _agent_obs(self, state, agent_id):
        robot = state['robots'][agent_id]
        r = robot[0] / len(state['map'])
        c = robot[1] / len(state['map'][0])
        carrying = robot[2]
        t = state['time_step'] / 1000.0
        return np.array([r, c, carrying, t], dtype=np.float32)

    def _global_state(self, state):
        obs_all = []
        for i in range(self.num_agents):
            obs_all.append(self._agent_obs(state, i))
        return np.concatenate(obs_all)

    # ----------------------------------------------------------------------
    def init_agents(self, state):
        self.num_agents = len(state['robots'])
        self.obs_dim = 4
        self.state_dim = self.obs_dim * self.num_agents
        self.joint_action_dim = (MOVE_DIM + PKG_DIM) * self.num_agents

        for _ in range(self.num_agents):
            act = Actor(self.obs_dim).to(self.device)
            self.actors.append(act)
            self.actor_opts.append(optim.Adam(act.parameters(), lr=self.lr_actor))

        self.critic = CentralCritic(self.state_dim, self.joint_action_dim).to(self.device)
        self.target_critic = CentralCritic(self.state_dim, self.joint_action_dim).to(self.device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=self.lr_critic)

    # ----------------------------------------------------------------------
    def get_actions(self, state, explore=True):
        self.last_state = state
        self.last_agent_obs = []
        self.last_action_onehot = []
        actions_env = []
        self.last_action_log_probs = []

        for i in range(self.num_agents):
            obs = torch.tensor(self._agent_obs(state, i), dtype=torch.float32, device=self.device).unsqueeze(0)
            move_logits, pkg_logits = self.actors[i](obs)
            m_idx, p_idx, move_prob, pkg_prob = self.decode_logits(move_logits, pkg_logits, explore)
            log_prob = torch.log(move_prob[m_idx] + 1e-8) + torch.log(pkg_prob[p_idx] + 1e-8)

            self.last_agent_obs.append(obs.squeeze(0).cpu().numpy())
            self.last_action_onehot.append(self.encode_action(m_idx, p_idx))
            self.last_action_log_probs.append(log_prob)
            actions_env.append(self.tuple_env_action(m_idx, p_idx))

        return actions_env

    def remember(self, next_state, rewards, done):
        # store experience in buffer
        state_vec = self._global_state(self.last_state)
        next_state_vec = self._global_state(next_state)
        joint_action_vec = np.concatenate(self.last_action_onehot)
        self.memory.push(state_vec, joint_action_vec, np.array(rewards, dtype=np.float32), next_state_vec, np.array(done, dtype=np.float32))
        self._learn()

    # ----------------------------------------------------------------------
    def _learn(self):
        if len(self.memory) < self.batch_size:
            return
        batch = self.memory.sample()

        # prepare tensors
        state_batch = torch.tensor(np.vstack([e.state for e in batch]), dtype=torch.float32, device=self.device)
        action_batch = torch.tensor(np.vstack([e.joint_action for e in batch]), dtype=torch.float32, device=self.device)
        reward_batch = torch.tensor(np.vstack([e.rewards for e in batch]), dtype=torch.float32, device=self.device)
        next_state_batch = torch.tensor(np.vstack([e.next_state for e in batch]), dtype=torch.float32, device=self.device)
        done_batch = torch.tensor(np.vstack([e.dones for e in batch]), dtype=torch.float32, device=self.device)

        # ---------------- Critic update --------------------
        with torch.no_grad():
            q_next = self.target_critic(torch.cat([next_state_batch, action_batch], dim=-1))
            y = reward_batch.sum(dim=1, keepdim=True) + self.gamma * q_next * (1 - done_batch)
        critic_input = torch.cat([state_batch, action_batch], dim=-1)
        q = self.critic(critic_input)
        critic_loss = F.mse_loss(q, y)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # ---------------- Actor update --------------------
        # For each agent compute counterfactual advantage
        for agent_idx in range(self.num_agents):
            # Current policy
            obs_i_batch = state_batch[:, agent_idx * self.obs_dim:(agent_idx + 1) * self.obs_dim]
            obs_i_batch_t = obs_i_batch.detach()
            actor = self.actors[agent_idx]
            logits_move, logits_pkg = actor(obs_i_batch_t)
            prob_move = F.softmax(logits_move, dim=-1)
            prob_pkg = F.softmax(logits_pkg, dim=-1)

            # Sample greedy for baseline expectation
            q_baseline = 0.0
            for m in range(MOVE_DIM):
                for p in range(PKG_DIM):
                    act_onehot = self.encode_action(m, p)
                    act_repeated = np.repeat(act_onehot[np.newaxis, :], self.batch_size, axis=0)
                    # Build joint action where only agent's part varies, others remain from batch
                    joint_actions = action_batch.clone().cpu().numpy()
                    start = agent_idx * (MOVE_DIM + PKG_DIM)
                    end = start + MOVE_DIM + PKG_DIM
                    joint_actions[:, start:end] = act_repeated
                    joint_actions_t = torch.tensor(joint_actions, dtype=torch.float32, device=self.device)
                    sa = torch.cat([state_batch, joint_actions_t], dim=-1)
                    with torch.no_grad():
                        q_val = self.critic(sa).squeeze(-1)
                    q_baseline += prob_move[:, m] * prob_pkg[:, p] * q_val
            # Actual Q
            sa_actual = torch.cat([state_batch, action_batch], dim=-1)
            q_actual = self.critic(sa_actual).squeeze(-1)
            advantage = (q_actual - q_baseline).detach()

            # Log prob of taken action (need from memory). For simplicity we recompute using actions batch
            taken_onehot = action_batch[:, agent_idx * (MOVE_DIM + PKG_DIM):(agent_idx + 1) * (MOVE_DIM + PKG_DIM)]
            # separate move and pkg indices from onehot
            move_taken = taken_onehot[:, :MOVE_DIM]
            pkg_taken = taken_onehot[:, MOVE_DIM:]
            move_idx_taken = move_taken.argmax(dim=-1)
            pkg_idx_taken = pkg_taken.argmax(dim=-1)

            log_prob_taken = torch.log(prob_move.gather(1, move_idx_taken.unsqueeze(1)).squeeze(1) + 1e-8) + \
                             torch.log(prob_pkg.gather(1, pkg_idx_taken.unsqueeze(1)).squeeze(1) + 1e-8)

            loss_actor = -(log_prob_taken * advantage).mean()
            self.actor_opts[agent_idx].zero_grad()
            loss_actor.backward()
            self.actor_opts[agent_idx].step()

        # ---------------- target critic soft update -----------------
        self._soft_update(self.critic, self.target_critic)

    def _soft_update(self, local, target):
        for t_param, l_param in zip(target.parameters(), local.parameters()):
            t_param.data.copy_(self.tau * l_param.data + (1 - self.tau) * t_param.data) 