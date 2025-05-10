import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from env import Environment
from sa_utils import ACTION_SPACE, IDX2ACTION

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, n_robots, action_dim):
        super(ActorCritic, self).__init__()
        self.n_robots = n_robots
        self.action_dim = action_dim
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        self.actor = nn.Linear(128, n_robots * action_dim)
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        x = self.shared(x)
        logits = self.actor(x)
        logits = logits.view(-1, self.n_robots, self.action_dim)
        value = self.critic(x)
        return logits, value


def state_to_vector(state, map_size, max_pkgs):
    vec = np.zeros(4 + 7 * max_pkgs, dtype=np.float32)
    r, c, carrying = state['robots'][0]
    r0, c0 = r - 1, c - 1
    vec[0] = r0 / map_size[0]
    vec[1] = c0 / map_size[1]
    vec[2] = 1.0 if carrying > 0 else 0.0
    vec[3] = carrying / max_pkgs if carrying > 0 else 0.0
    for i, pkg in enumerate(state['packages']):
        if i >= max_pkgs:
            break
        base = 4 + i * 7
        _, sr, sc, tr, tc, *_ = pkg
        vec[base] = 1.0
        vec[base+1] = 0.0
        vec[base+2] = 0.0
        vec[base+3] = (sr - 1) / map_size[0]
        vec[base+4] = (sc - 1) / map_size[1]
        vec[base+5] = (tr - 1) / map_size[0]
        vec[base+6] = (tc - 1) / map_size[1]
    return vec


def collect_trajectories(env, model, args):
    obs_list, actions_list, logprobs_list, values_list, rewards_list, dones_list = [], [], [], [], [], []
    state = env.reset()
    obs_vec = state_to_vector(state, (env.n_rows, env.n_cols), env.n_packages)
    obs = torch.tensor(obs_vec, dtype=torch.float32).unsqueeze(0)
    for _ in range(args.rollout_length):
        logits, value = model(obs)
        dist = torch.distributions.Categorical(logits=logits)
        actions = dist.sample()  # [1, n_robots]
        logprobs = dist.log_prob(actions)
        action_tuples = [IDX2ACTION[a.item()] for a in actions[0]]
        state_next, reward, done, _ = env.step(action_tuples)
        obs_next_vec = state_to_vector(state_next, (env.n_rows, env.n_cols), env.n_packages)
        obs_next = torch.tensor(obs_next_vec, dtype=torch.float32).unsqueeze(0)
        obs_list.append(obs)
        actions_list.append(actions)
        logprobs_list.append(logprobs)
        values_list.append(value)
        rewards_list.append(torch.tensor([reward], dtype=torch.float32))
        dones_list.append(done)
        obs = obs_next
        if done:
            break
    returns = []
    R = torch.zeros(1)
    for r, d in zip(reversed(rewards_list), reversed(dones_list)):
        R = r + args.gamma * R * (1 - d)
        returns.insert(0, R)
    returns = torch.cat(returns)
    values = torch.cat(values_list).squeeze(-1)
    advantages = returns - values
    return obs_list, actions_list, logprobs_list, values_list, returns, advantages


def train(args):
    env = Environment(map_file=args.map_file,
                      max_time_steps=args.max_time_steps,
                      n_robots=args.n_robots,
                      n_packages=args.n_packages)
    obs_dim = 4 + 7 * args.n_packages
    n_robots = args.n_robots
    action_dim = len(ACTION_SPACE)
    model = ActorCritic(obs_dim, n_robots, action_dim)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    for iter in range(args.num_updates):
        obs_list, actions_list, old_logprobs_list, values_list, returns, advantages = collect_trajectories(env, model, args)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        for _ in range(args.ppo_epochs):
            for obs, actions, old_logprobs, ret, adv in zip(obs_list, actions_list, old_logprobs_list, returns, advantages):
                logits, value = model(obs)
                dist = torch.distributions.Categorical(logits=logits)
                logprobs = dist.log_prob(actions)
                ratio = (logprobs - old_logprobs).exp()
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1 - args.clip_eps, 1 + args.clip_eps) * adv
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = (ret - value.squeeze(-1)).pow(2).mean()
                entropy = dist.entropy().mean()
                loss = actor_loss + args.vf_coeff * critic_loss - args.ent_coeff * entropy
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    print("Training complete")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HAPPO training in PyTorch")
    parser.add_argument('--map_file', type=str, default='map1.txt')
    parser.add_argument('--n_robots', type=int, default=2)
    parser.add_argument('--n_packages', type=int, default=5)
    parser.add_argument('--max_time_steps', type=int, default=100)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--ppo_epochs', type=int, default=4)
    parser.add_argument('--clip_eps', type=float, default=0.2)
    parser.add_argument('--rollout_length', type=int, default=128)
    parser.add_argument('--num_updates', type=int, default=1000)
    parser.add_argument('--ent_coeff', type=float, default=0.01)
    parser.add_argument('--vf_coeff', type=float, default=0.5)
    args = parser.parse_args()
    train(args) 