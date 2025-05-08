import numpy as np
import random
from collections import deque, namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

action_move = ['S', 'L', 'R', 'U', 'D']
action_pkg = ['0', '1', '2']
MOVE_DIM = len(action_move)
PKG_DIM = len(action_pkg)
TOTAL_DISCRETE_ACTIONS = MOVE_DIM + PKG_DIM  # We will output two distributions


def one_hot(index, size):
    x = np.zeros(size, dtype=np.float32)
    x[index] = 1.0
    return x


class Actor(nn.Module):
    """Simple MLP Actor that outputs two softmax distributions (move and package)."""

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


class Critic(nn.Module):
    """Centralised critic that takes all observations and all actions as inputs."""

    def __init__(self, total_obs_dim, total_action_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(total_obs_dim + total_action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, obs_act):
        x = F.relu(self.fc1(obs_act))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.out(x)


class ReplayBuffer:
    def __init__(self, buffer_size=int(1e6), batch_size=1024):
        self.memory = deque(maxlen=buffer_size)
        self.experience = namedtuple("Experience", field_names=[
            "obs", "actions", "rewards", "next_obs", "dones"])
        self.batch_size = batch_size

    def push(self, obs, actions, rewards, next_obs, dones):
        e = self.experience(obs, actions, rewards, next_obs, dones)
        self.memory.append(e)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)
        # Convert to tensors later in agent
        return experiences

    def __len__(self):
        return len(self.memory)


class MADDPGAgents:
    """A wrapper that manages multiple agents each with own actor but centralised critics."""

    def __init__(self, lr_actor=1e-3, lr_critic=1e-3, gamma=0.95, tau=0.01, batch_size=1024,
                 buffer_size=int(1e6), device=None):
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.num_agents = 0
        self.obs_dim = None
        self.total_obs_dim = None
        self.total_action_dim = None
        self.actors = []
        self.critics = []
        self.target_actors = []
        self.target_critics = []
        self.actor_optimisers = []
        self.critic_optimisers = []

        self.memory = ReplayBuffer(buffer_size, batch_size)
        self.time_step = 0  # for updating every some steps

    # -----------------------------------------------------------------------
    # Helper functions for encoding/decoding actions and observations
    # -----------------------------------------------------------------------
    @staticmethod
    def encode_action_tuple(move_idx, pkg_idx):
        """Return concatenated one-hot representation."""
        return np.concatenate([one_hot(move_idx, MOVE_DIM), one_hot(pkg_idx, PKG_DIM)])

    @staticmethod
    def decode_action(move_logits, pkg_logits, explore=True):
        move_probs = F.softmax(move_logits, dim=-1)
        pkg_probs = F.softmax(pkg_logits, dim=-1)
        if explore:
            move_idx = torch.distributions.Categorical(move_probs).sample()
            pkg_idx = torch.distributions.Categorical(pkg_probs).sample()
        else:
            move_idx = move_probs.argmax(dim=-1)
            pkg_idx = pkg_probs.argmax(dim=-1)
        return move_idx.item(), pkg_idx.item()

    @staticmethod
    def tuple_to_env_action(move_idx, pkg_idx):
        return (action_move[move_idx], action_pkg[pkg_idx])

    def _extract_agent_obs(self, state, agent_id):
        """Very naive encoding: agent (row, col, carrying) & time step."""
        robot = state['robots'][agent_id]
        robot_row = robot[0] / len(state['map'])  # normalised 0-1
        robot_col = robot[1] / len(state['map'][0])
        carrying = robot[2]
        time_step = state['time_step'] / 1000.0  # assume max 1000
        obs = np.array([robot_row, robot_col, carrying, time_step], dtype=np.float32)
        return obs

    def init_agents(self, state):
        self.num_agents = len(state['robots'])
        # Use simple obs dim as above
        self.obs_dim = 4
        self.total_obs_dim = self.obs_dim * self.num_agents
        # Action dim is one-hot MOVE+PKG for each agent
        self.total_action_dim = (MOVE_DIM + PKG_DIM) * self.num_agents

        for _ in range(self.num_agents):
            actor = Actor(self.obs_dim).to(self.device)
            target_actor = Actor(self.obs_dim).to(self.device)
            target_actor.load_state_dict(actor.state_dict())

            critic = Critic(self.total_obs_dim, self.total_action_dim).to(self.device)
            target_critic = Critic(self.total_obs_dim, self.total_action_dim).to(self.device)
            target_critic.load_state_dict(critic.state_dict())

            self.actors.append(actor)
            self.target_actors.append(target_actor)
            self.critics.append(critic)
            self.target_critics.append(target_critic)

            self.actor_optimisers.append(optim.Adam(actor.parameters(), lr=self.lr_actor))
            self.critic_optimisers.append(optim.Adam(critic.parameters(), lr=self.lr_critic))

    # -----------------------------------------------------------------------
    def get_actions(self, state, explore=True):
        # Build per-agent obs tensors
        obs_n = [self._extract_agent_obs(state, i) for i in range(self.num_agents)]
        obs_tensor_n = [torch.tensor(o, dtype=torch.float32, device=self.device).unsqueeze(0)
                        for o in obs_n]

        moves = []
        pkgs = []
        actions_for_env = []
        with torch.no_grad():
            for i in range(self.num_agents):
                move_logits, pkg_logits = self.actors[i](obs_tensor_n[i])
                move_idx, pkg_idx = self.decode_action(move_logits, pkg_logits, explore)
                moves.append(move_idx)
                pkgs.append(pkg_idx)
                actions_for_env.append(self.tuple_to_env_action(move_idx, pkg_idx))
        self.latest_obs = obs_n  # store for memory
        self.latest_actions = [self.encode_action_tuple(m, p) for m, p in zip(moves, pkgs)]
        return actions_for_env

    def remember(self, next_state, rewards, done):
        # Build next obs list
        next_obs_n = [self._extract_agent_obs(next_state, i) for i in range(self.num_agents)]
        # Flatten lists for storage
        self.memory.push(np.concatenate(self.latest_obs),
                         np.concatenate(self.latest_actions),
                         np.array(rewards, dtype=np.float32),
                         np.concatenate(next_obs_n),
                         np.array(done, dtype=np.float32))

        self.latest_obs = None
        self.latest_actions = None

        # Learn every timestep
        self.time_step += 1
        #Tạm thời comment tự động học mỗi bước để tránh tốn thời gian, chỉ học thủ công sau khi train xong
        if len(self.memory) >= self.batch_size:
            self.learn()
            print("is learning")

    def learn(self):
        experiences = self.memory.sample()
        # Convert to torch tensors
        obs_batch = torch.tensor(np.vstack([e.obs for e in experiences]), dtype=torch.float32, device=self.device)
        actions_batch = torch.tensor(np.vstack([e.actions for e in experiences]), dtype=torch.float32, device=self.device)
        rewards_batch = torch.tensor(np.vstack([e.rewards for e in experiences]), dtype=torch.float32, device=self.device)
        next_obs_batch = torch.tensor(np.vstack([e.next_obs for e in experiences]), dtype=torch.float32, device=self.device)
        dones_batch = torch.tensor(np.vstack([e.dones for e in experiences]), dtype=torch.float32, device=self.device)

        # For each agent compute targets and update actor/critic
        for agent_idx in range(self.num_agents):
            # ---------------- Critic update --------------------
            critic = self.critics[agent_idx]
            target_critic = self.target_critics[agent_idx]
            critic_optimizer = self.critic_optimisers[agent_idx]

            # Current Q
            critic_input = torch.cat([obs_batch, actions_batch], dim=-1)
            q_values = critic(critic_input)

            # Next actions from target actors
            next_actions = []
            for i in range(self.num_agents):
                actor_i = self.target_actors[i]
                obs_i = next_obs_batch[:, i * self.obs_dim:(i + 1) * self.obs_dim]
                move_logits, pkg_logits = actor_i(obs_i)
                move_idx = F.softmax(move_logits, dim=-1).argmax(dim=-1)
                pkg_idx = F.softmax(pkg_logits, dim=-1).argmax(dim=-1)
                one_hot_actions = []
                for b in range(move_idx.shape[0]):
                    one_hot_actions.append(self.encode_action_tuple(move_idx[b].item(), pkg_idx[b].item()))
                next_actions.append(torch.tensor(np.stack(one_hot_actions), dtype=torch.float32, device=self.device))
            next_actions_batch = torch.cat(next_actions, dim=-1)
            next_critic_input = torch.cat([next_obs_batch, next_actions_batch], dim=-1)
            with torch.no_grad():
                q_next = target_critic(next_critic_input)
                target_q = rewards_batch[:, agent_idx:agent_idx+1] + self.gamma * q_next * (1 - dones_batch)

            critic_loss = F.mse_loss(q_values, target_q)

            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            # ---------------- Actor update --------------------
            actor = self.actors[agent_idx]
            actor_optimizer = self.actor_optimisers[agent_idx]
            obs_i = obs_batch[:, agent_idx * self.obs_dim:(agent_idx + 1) * self.obs_dim]
            move_logits, pkg_logits = actor(obs_i)
            move_idx = F.softmax(move_logits, dim=-1).argmax(dim=-1)
            pkg_idx = F.softmax(pkg_logits, dim=-1).argmax(dim=-1)
            # Build continuous one-hot for actor actions
            actions_i = []
            for b in range(move_idx.shape[0]):
                actions_i.append(self.encode_action_tuple(move_idx[b].item(), pkg_idx[b].item()))
            actions_i = torch.tensor(np.stack(actions_i), dtype=torch.float32, device=self.device)

            all_actions_list = list(torch.split(actions_batch, (MOVE_DIM + PKG_DIM), dim=-1))
            all_actions_list[agent_idx] = actions_i  # replace for policy gradient
            actions_for_actor = torch.cat(all_actions_list, dim=-1)
            actor_input = torch.cat([obs_batch, actions_for_actor], dim=-1)
            actor_loss = -critic(actor_input).mean()

            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            # ---------------- Target networks soft update -------------
            self.soft_update(critic, target_critic)
            self.soft_update(actor, self.target_actors[agent_idx])

    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data) 