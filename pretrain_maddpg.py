import argparse
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sa_utils import ACTION_SPACE
from maddpg_agent import Actor, Critic

# Hỗ trợ mapping action index <-> one-hot
def action_onehot(a_idx):
    vec = np.zeros(len(ACTION_SPACE), dtype=np.float32)
    vec[a_idx] = 1.0
    return vec

class CriticDataset(Dataset):
    """Dataset cho pre-train Critic: input = [obs; action_onehot], target = Q(s,a)"""
    def __init__(self, encoder, Q):
        self.encoder = encoder          # state_encoder chứa idx2state: comp tuples
        self.Q = Q                      # numpy array shape (n_states, n_actions)
        self.inputs = []
        self.targets = []
        n_states, n_actions = Q.shape
        for s_idx in range(n_states):
            obs = np.array(self.encoder.idx2state[s_idx], dtype=np.float32)
            for a_idx in range(n_actions):
                act_oh = action_onehot(a_idx)
                inp = np.concatenate([obs, act_oh])
                self.inputs.append(inp)
                self.targets.append(Q[s_idx, a_idx])
        self.inputs = torch.tensor(self.inputs, dtype=torch.float32)
        self.targets = torch.tensor(self.targets, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

class ActorDataset(Dataset):
    """Dataset cho pre-train Actor: input = obs, labels = move_idx, pkg_idx."""
    def __init__(self, encoder, policy):
        self.encoder = encoder  # idx2state maps to comp tuples
        self.policy = policy    # dict state_idx -> action_idx
        self.obs = []
        self.labels_move = []
        self.labels_pkg = []
        for s_idx, a_idx in policy.items():
            obs = np.array(self.encoder.idx2state[s_idx], dtype=np.float32)
            self.obs.append(obs)
            move_idx, pkg_idx = ACTION_SPACE[a_idx]
            # move_idx and pkg_idx currently are characters; convert to indices
            # map char->index
            m = ['S','L','R','U','D'].index(move_idx)
            p = ['0','1','2'].index(pkg_idx)
            self.labels_move.append(m)
            self.labels_pkg.append(p)
        self.obs = torch.tensor(self.obs, dtype=torch.float32)
        self.labels_move = torch.tensor(self.labels_move, dtype=torch.long)
        self.labels_pkg = torch.tensor(self.labels_pkg, dtype=torch.long)

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        return self.obs[idx], self.labels_move[idx], self.labels_pkg[idx]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pre-train MADDPG Actor & Critic')
    parser.add_argument('--in_q', type=str, default='q_values.pkl', help='Pickle Q(s,a) and encoder')
    parser.add_argument('--in_policy', type=str, default='policy_init.pkl', help='Pickle policy and encoder')
    parser.add_argument('--critic_epochs', type=int, default=100, help='Epochs to train critic')
    parser.add_argument('--actor_epochs', type=int, default=100, help='Epochs to train actor')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--lr_critic', type=float, default=1e-3, help='Learning rate critic')
    parser.add_argument('--lr_actor', type=float, default=1e-3, help='Learning rate actor')
    parser.add_argument('--save_dir', type=str, default='pretrained_maddpg', help='Directory to save models')
    args = parser.parse_args()

    # Load Q and policy pickles
    with open(args.in_q, 'rb') as f:
        data_q = pickle.load(f)
    Q = data_q['Q']
    encoder_q = data_q['state_encoder']

    with open(args.in_policy, 'rb') as f:
        data_p = pickle.load(f)
    policy = data_p['policy']

    # Create save dir
    os.makedirs(args.save_dir, exist_ok=True)

    # Critic: input dim = obs_dim + action_dim
    obs_dim = len(encoder_q.idx2state[0])
    action_dim = len(ACTION_SPACE)
    critic_input_dim = obs_dim + action_dim

    # Instantiate Critic network
    critic = Critic(obs_dim, action_dim, hidden_dim=256)
    optimizer_c = optim.Adam(critic.parameters(), lr=args.lr_critic)
    criterion = nn.MSELoss()

    # Prepare dataset and loader for Critic
    critic_dataset = CriticDataset(encoder_q, Q)
    critic_loader = DataLoader(critic_dataset, batch_size=args.batch_size, shuffle=True)

    # Train Critic
    critic.train()
    for epoch in range(1, args.critic_epochs+1):
        total_loss = 0
        for x, y in critic_loader:
            optimizer_c.zero_grad()
            pred = critic(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer_c.step()
            total_loss += loss.item()
        print(f'[Critic] Epoch {epoch}/{args.critic_epochs}, Loss: {total_loss/len(critic_loader):.6f}')
    torch.save(critic.state_dict(), os.path.join(args.save_dir, 'critic.pth'))

    # Actor: input dim = obs_dim, output heads MOVE_DIM and PKG_DIM
    from maddpg_agent import MOVE_DIM, PKG_DIM
    actor = Actor(obs_dim, hidden_dim=256)
    optimizer_a = optim.Adam(actor.parameters(), lr=args.lr_actor)
    criterion_ce = nn.CrossEntropyLoss()

    # Prepare dataset for Actor
    actor_dataset = ActorDataset(encoder_q, policy)
    actor_loader = DataLoader(actor_dataset, batch_size=args.batch_size, shuffle=True)

    # Train Actor
    actor.train()
    for epoch in range(1, args.actor_epochs+1):
        total_loss = 0
        for obs, move_lbl, pkg_lbl in actor_loader:
            optimizer_a.zero_grad()
            move_logits, pkg_logits = actor(obs)
            loss_move = criterion_ce(move_logits, move_lbl)
            loss_pkg = criterion_ce(pkg_logits, pkg_lbl)
            loss = loss_move + loss_pkg
            loss.backward()
            optimizer_a.step()
            total_loss += loss.item()
        print(f'[Actor]  Epoch {epoch}/{args.actor_epochs}, Loss: {total_loss/len(actor_loader):.6f}')
    torch.save(actor.state_dict(), os.path.join(args.save_dir, 'actor.pth'))

    print('Pre-training completed. Models saved in', args.save_dir) 