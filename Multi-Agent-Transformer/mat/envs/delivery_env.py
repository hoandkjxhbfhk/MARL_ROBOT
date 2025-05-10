import gym
import numpy as np
from gym import spaces
from env import Environment as BaseEnv
from sa_utils import MOVE_ACTIONS, PKG_ACTIONS

class DeliveryEnv(gym.Env):
    def __init__(self, args):
        super(DeliveryEnv, self).__init__()
        # Khởi tạo environment gốc
        self.base_env = BaseEnv(map_file=args.map,
                                max_time_steps=args.max_time_steps,
                                n_robots=args.num_agents,
                                n_packages=args.n_packages,
                                seed=args.seed)
        # Số agent
        self.n = self.base_env.n_robots
        # Định nghĩa không gian quan sát: (row_norm, col_norm, carrying_flag) + one-hot agent id
        obs_dim = 3 + self.n
        # Local observation space per agent
        self.observation_space = [spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
                                  for _ in range(self.n)]
        # Shared observation space for critic (concatenation of all local obs)
        self.share_observation_space = [spaces.Box(low=0.0, high=1.0, shape=(obs_dim * self.n,), dtype=np.float32)
                                        for _ in range(self.n)]
        # Action space: treat joint (move,pkg) as a single discrete action
        self.num_moves = len(MOVE_ACTIONS)
        self.num_pkgs = len(PKG_ACTIONS)
        self.num_actions = self.num_moves * self.num_pkgs
        self.action_space = [spaces.Discrete(self.num_actions) for _ in range(self.n)]

    def seed(self, seed=None):
        if seed is not None:
            self.base_env.rng = np.random.RandomState(seed)

    def reset(self):
        # Lấy state dict từ env.py
        state_dict = self.base_env.reset()
        robots = state_dict['robots']  # list of tuples (r, c, carrying)
        n_rows, n_cols = self.base_env.n_rows, self.base_env.n_cols
        obs_n = []
        for i in range(self.n):
            r, c, carrying = robots[i]
            # Chuyển về 0-index
            r0, c0 = r - 1, c - 1
            row_norm = r0 / max(1, n_rows - 1)
            col_norm = c0 / max(1, n_cols - 1)
            carry_flag = 1.0 if carrying > 0 else 0.0
            # agent id one-hot
            agent_id_vec = np.zeros(self.n, dtype=np.float32)
            agent_id_vec[i] = 1.0
            obs_i = np.concatenate(([row_norm, col_norm, carry_flag], agent_id_vec), axis=0)
            obs_n.append(obs_i.astype(np.float32))
        return obs_n

    def step(self, actions):
        # Chuyển action (one-hot vector hoặc int) thành tuple (move, pkg)
        action_list = []
        for a in actions:
            # a có thể là one-hot vector hoặc số nguyên
            if isinstance(a, (list, tuple, np.ndarray)):
                idx = int(np.argmax(a))
            else:
                idx = int(a)
            move_idx = idx // self.num_pkgs
            pkg_idx = idx % self.num_pkgs
            action_list.append((MOVE_ACTIONS[move_idx], PKG_ACTIONS[pkg_idx]))
        next_state_dict, reward, done, info = self.base_env.step(action_list)
        robots = next_state_dict['robots']
        n_rows, n_cols = self.base_env.n_rows, self.base_env.n_cols
        obs_n = []
        for i in range(self.n):
            r, c, carrying = robots[i]
            r0, c0 = r - 1, c - 1
            row_norm = r0 / max(1, n_rows - 1)
            col_norm = c0 / max(1, n_cols - 1)
            carry_flag = 1.0 if carrying > 0 else 0.0
            agent_id_vec = np.zeros(self.n, dtype=np.float32)
            agent_id_vec[i] = 1.0
            obs_i = np.concatenate(([row_norm, col_norm, carry_flag], agent_id_vec), axis=0)
            obs_n.append(obs_i.astype(np.float32))
        # Reward và done đồng nhất cho tất cả agents
        reward_n = [[reward] for _ in range(self.n)]
        done_n = [done for _ in range(self.n)]
        info_n = [info for _ in range(self.n)]
        return obs_n, reward_n, done_n, info_n

    def render(self, mode='human'):
        return self.base_env.render() 