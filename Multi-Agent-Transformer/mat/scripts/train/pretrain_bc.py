#!/usr/bin/env python
import sys, os
# Thêm project root và MAT code vào đường dẫn
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from mat.config import get_config
from mat.envs.delivery_env import DeliveryEnv
from mat.algorithms.mat.algorithm.transformer_policy import TransformerPolicy

class GreedyDataset(Dataset):
    """
    Dataset chứa các quan sát và hành động từ greedyagent.
    Dữ liệu lưu trong file npz với keys 'obs' và 'acts'.
    obs: shape (N, n_agents, obs_dim)
    acts: shape (N, n_agents)
    """
    def __init__(self, data_file):
        data = np.load(data_file)
        self.obs = data['obs']
        self.acts = data['acts']
    def __len__(self):
        return len(self.obs)
    def __getitem__(self, idx):
        return self.obs[idx], self.acts[idx]


def main():
    # Parser chung từ MAT
    parser = get_config()
    # Thêm tham số cho BC
    parser.add_argument('--data_file', type=str, required=True,
                        help='File npz chứa obs và acts do greedyagent thu thập')
    parser.add_argument('--map', type=str, default='map1.txt', help='File bản đồ')
    parser.add_argument('--n_packages', type=int, default=5)
    parser.add_argument('--num_agents', type=int, default=1)
    parser.add_argument('--max_time_steps', type=int, default=100)
    parser.add_argument('--pretrain_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()

    # Tạo env để lấy không gian obs/act
    env = DeliveryEnv(args)
    obs_space = env.observation_space[0]
    cent_space = env.share_observation_space[0]
    act_space = env.action_space[0]

    # Thiết bị
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')

    # Dataset và DataLoader
    dataset = GreedyDataset(args.data_file)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Tạo policy
    policy = TransformerPolicy(args, obs_space, cent_space, act_space, args.num_agents, device)
    policy.train()
    optimizer = policy.optimizer

    # Khởi rnn_states và masks mặc định
    # Lấy kích thước từ policy
    rnn_shape = (args.batch_size, args.num_agents, args.recurrent_N, args.hidden_size) if hasattr(args, 'recurrent_N') else (args.batch_size, args.num_agents, 1, policy.obs_dim)
    mask_shape = (args.batch_size, args.num_agents, 1)

    for epoch in range(args.pretrain_epochs):
        total_loss = 0.0
        count = 0
        for obs_np, acts_np in loader:
            # obs_np: shape (B, n_agents, obs_dim)
            # acts_np: shape (B, n_agents)
            B = obs_np.shape[0]
            # điều chỉnh nếu batch cuối nhỏ hơn batch_size
            batch_size = B
            obs = torch.tensor(obs_np, dtype=torch.float32, device=device)
            # Dùng cùng obs cho share_obs
            cent_obs = obs
            # reshape về đúng kích thước [B*n_agent, ...] sẽ được policy xử lý
            # Khởi initial rnn_states và masks
            rnn_states_actor = torch.zeros((batch_size, args.num_agents, args.recurrent_N, args.hidden_size), dtype=torch.float32, device=device)
            rnn_states_critic = torch.zeros_like(rnn_states_actor)
            masks = torch.ones((batch_size, args.num_agents, 1), dtype=torch.float32, device=device)
            # Actions tensor (B, n_agents, 1)
            actions = torch.tensor(acts_np, dtype=torch.long, device=device).unsqueeze(-1)

            # Tính log-probs của greedy actions
            values, log_probs, _ = policy.evaluate_actions(
                cent_obs.reshape(-1, args.num_agents, cent_space.shape[0]),
                obs.reshape(-1, args.num_agents, obs_space.shape[0]),
                rnn_states_actor, rnn_states_critic,
                actions, masks,
                available_actions=None, active_masks=None
            )
            # log_probs shape (B*n_agents, 1)
            bc_loss = -log_probs.mean()

            optimizer.zero_grad()
            bc_loss.backward()
            optimizer.step()

            total_loss += bc_loss.item()
            count += 1

        print(f'Epoch {epoch+1}/{args.pretrain_epochs}, BC loss = {total_loss/count:.6f}')

    # Lưu pretrained model
    save_dir = os.path.join(os.getcwd(), 'pretrained_models')
    os.makedirs(save_dir, exist_ok=True)
    torch.save(policy.transformer.state_dict(), os.path.join(save_dir, 'greedy_pretrained.pt'))
    print('Pretrain hoàn tất, model lưu tại pretrained_models/greedy_pretrained.pt')

if __name__ == '__main__':
    main() 