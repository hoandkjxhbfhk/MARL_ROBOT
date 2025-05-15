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
        # Tính kích thước và định nghĩa không gian quan sát mở rộng
        self.grid = self.base_env.grid
        self.n_rows = self.base_env.n_rows
        self.n_cols = self.base_env.n_cols
        self.n_packages = args.n_packages
        orig_feat_len = 3 + self.n
        map_feat_len = self.n_rows * self.n_cols
        pkg_feat_len = 2 + 2 + 1 + 4  # start(x,y), target(x,y), time_remain, status one-hot(4)
        total_pkg_feat = pkg_feat_len * self.n_packages
        dist_feat_len = 1
        rel_feat_len = 2 * (self.n - 1)
        obs_dim = orig_feat_len + map_feat_len + total_pkg_feat + dist_feat_len + rel_feat_len
        # Local observation space per agent (giá trị trong [-1,1])
        self.observation_space = [spaces.Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
                                  for _ in range(self.n)]
        # Shared observation space cho critic
        self.share_observation_space = [spaces.Box(low=-1.0, high=1.0, shape=(obs_dim * self.n,), dtype=np.float32)
                                        for _ in range(self.n)]
        # Action space: treat joint (move,pkg) as a single discrete action
        self.num_moves = len(MOVE_ACTIONS)
        self.num_pkgs = len(PKG_ACTIONS)
        self.num_actions = self.num_moves * self.num_pkgs
        self.action_space = [spaces.Discrete(self.num_actions) for _ in range(self.n)]

    def seed(self, seed=None):
        if seed is not None:
            self.base_env.rng = np.random.RandomState(seed)

    def _build_obs(self):
        # Bản đồ flatten
        map_flat = np.array(self.base_env.grid, dtype=np.float32).flatten()
        # Thông tin packages
        pkg_feats = []
        status_map = {'None': 0, 'waiting': 1, 'in_transit': 2, 'delivered': 3}
        for pkg in self.base_env.packages:
            sr, sc = pkg.start
            tr, tc = pkg.target
            sr_norm = sr / max(1, self.n_rows - 1)
            sc_norm = sc / max(1, self.n_cols - 1)
            tr_norm = tr / max(1, self.n_rows - 1)
            tc_norm = tc / max(1, self.n_cols - 1)
            time_remain = (pkg.deadline - self.base_env.t) / self.base_env.max_time_steps
            status_vec = np.zeros(4, dtype=np.float32)
            status_vec[status_map.get(pkg.status, 0)] = 1.0
            pkg_feats.extend([sr_norm, sc_norm, tr_norm, tc_norm, time_remain])
            pkg_feats.extend(status_vec.tolist())
        pkg_feats = np.array(pkg_feats, dtype=np.float32)
        # Xây quan sát cho từng agent
        obs_n = []
        robots = self.base_env.robots
        for i, robot in enumerate(robots):
            r, c = robot.position
            row_norm = r / max(1, self.n_rows - 1)
            col_norm = c / max(1, self.n_cols - 1)
            carry_flag = 1.0 if robot.carrying > 0 else 0.0
            id_vec = np.zeros(self.n, dtype=np.float32)
            id_vec[i] = 1.0
            # Khoảng cách đến package chờ gần nhất
            waiting = [p for p in self.base_env.packages if p.status == 'waiting']
            if waiting:
                dists = [abs(r - p.start[0]) + abs(c - p.start[1]) for p in waiting]
                dist_norm = min(dists) / max(self.n_rows, self.n_cols)
            else:
                dist_norm = 0.0
            # Vị trí tương đối robot khác
            rel_pos = []
            for j, other in enumerate(robots):
                if j == i: continue
                orow, ocol = other.position
                rel_r = (orow - r) / max(1, self.n_rows - 1)
                rel_c = (ocol - c) / max(1, self.n_cols - 1)
                rel_pos.extend([rel_r, rel_c])
            obs_i = np.concatenate((
                [row_norm, col_norm, carry_flag],
                id_vec,
                map_flat,
                pkg_feats,
                [dist_norm],
                rel_pos
            ), axis=0)
            obs_n.append(obs_i)
        return obs_n

    def reset(self):
        # Reset môi trường gốc và trả về quan sát mới
        self.base_env.reset()
        return self._build_obs()

    def step(self, actions):
        # Decode và thực thi hành động, sau đó trả về quan sát mới
        action_list = []
        for a in actions:
            if isinstance(a, (list, tuple, np.ndarray)):
                idx = int(np.argmax(a))
            else:
                idx = int(a)
            move_idx = idx // self.num_pkgs
            pkg_idx = idx % self.num_pkgs
            action_list.append((MOVE_ACTIONS[move_idx], PKG_ACTIONS[pkg_idx]))
        _, reward, done, info = self.base_env.step(action_list)
        obs_n = self._build_obs()
        reward_n = [[reward] for _ in range(self.n)]
        done_n = [done for _ in range(self.n)]
        info_n = [info for _ in range(self.n)]
        return obs_n, reward_n, done_n, info_n

    def render(self, mode='human'):
        return self.base_env.render() 