from typing import List, Tuple
import os, sys
import numpy as np
import gym
from gym import spaces

# Bảo đảm path tới thư mục gốc nằm trong sys.path để import mappo và DeliveryEnv
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if (ROOT_DIR not in sys.path):
    sys.path.append(ROOT_DIR)

# Import DeliveryEnv và các hàm util đã có trong mappo
try:
    from mappo import (
        convert_observation,
        generate_vector_features,
        convert_global_state,
        compute_shaped_rewards,
    )
except ImportError:
    # Trường hợp không import được do mappo không phải package, thêm path hiện tại
    pass

from env import Environment  # DeliveryEnv ở thư mục gốc

from mat.envs.ma_mujoco.multiagent_mujoco.multiagentenv import MultiAgentEnv

# Hằng số hành động – giống trong mappo
MOVE_ACTIONS = ["S", "L", "R", "U", "D"]
PKG_ACTIONS = ["0", "1", "2"]
ACTION_DIM = len(MOVE_ACTIONS) * len(PKG_ACTIONS)  # 15


class DeliveryMATEnv(MultiAgentEnv):
    """Wrapper chuyển DeliveryEnv sang interface MultiAgentEnv cho MAT."""

    def __init__(self, env_args: dict):
        # Khởi tạo MultiAgentEnv (không cần batch_size)
        super().__init__(batch_size=None, env_args=env_args)

        # Lưu trữ tham số
        self.n_agents: int = env_args.get("n_robots", 5)
        self.map_file: str = env_args.get("map_file", "map1.txt")
        self.n_packages: int = env_args.get("n_packages", 50)
        self.max_time_steps: int = env_args.get("max_time_steps", 500)

        # Khởi tạo DeliveryEnv gốc
        self.base_env = Environment(
            map_file=self.map_file,
            n_robots=self.n_agents,
            n_packages=self.n_packages,
            max_time_steps=self.max_time_steps,
            move_cost=env_args.get("move_cost", -0.01),
            delivery_reward=env_args.get("delivery_reward", 10),
            delay_reward=env_args.get("delay_reward", 1),
            seed=env_args.get("seed", 42),
        )

        # Kích thước map
        self.n_rows = self.base_env.n_rows
        self.n_cols = self.base_env.n_cols

        # Xây dựng observation & state dimensions
        dummy_state = self.base_env.get_state()
        self.persistent_packages = {}
        self.update_persistent_packages(dummy_state)

        obs_vector_dim = generate_vector_features(
            dummy_state, self.persistent_packages, 0, self.max_time_steps, self.n_agents - 1, 5
        ).shape[0]
        obs_spatial_dim = 6 * self.n_rows * self.n_cols
        self.obs_dim = obs_spatial_dim + obs_vector_dim

        state_vec_dim = convert_global_state(
            dummy_state, self.persistent_packages, self.max_time_steps
        )[1].shape[0]
        state_map_flat_dim = 4 * self.n_rows * self.n_cols
        self.state_dim = state_map_flat_dim + state_vec_dim

        # Xây dựng Gym spaces
        high_obs = np.ones(self.obs_dim, dtype=np.float32)
        self.observation_space: List[spaces.Box] = [
            spaces.Box(low=-high_obs, high=high_obs, dtype=np.float32) for _ in range(self.n_agents)
        ]
        high_state = np.ones(self.state_dim, dtype=np.float32)
        self.share_observation_space: List[spaces.Box] = [
            spaces.Box(low=-high_state, high=high_state, dtype=np.float32) for _ in range(self.n_agents)
        ]
        self.action_space = tuple([spaces.Discrete(ACTION_DIM) for _ in range(self.n_agents)])

        # Các biến thời gian
        self.steps = 0
        self.episode_limit = self.max_time_steps

    # ------------------------------------------------------------------
    # Helper – cập nhật persistent_packages từ trạng thái env.
    def update_persistent_packages(self, env_state_dict):
        """Theo dõi trạng thái package theo thời gian để phục vụ reward shaping."""
        t = env_state_dict["time_step"]
        for pkg_tuple in env_state_dict.get("packages", []):
            pkg_id = pkg_tuple[0]
            if pkg_id not in self.persistent_packages:
                self.persistent_packages[pkg_id] = {
                    "id": pkg_id,
                    "start_pos": (pkg_tuple[1] - 1, pkg_tuple[2] - 1),
                    "target_pos": (pkg_tuple[3] - 1, pkg_tuple[4] - 1),
                    "start_time": pkg_tuple[5],
                    "deadline": pkg_tuple[6],
                    "status": "waiting",
                }

        # Cập nhật trạng thái package dựa vào robots
        carried_ids = {int(r[2]) for r in env_state_dict["robots"] if int(r[2]) != 0}
        for pkg_id, pkg_data in list(self.persistent_packages.items()):
            if pkg_id in carried_ids:
                pkg_data["status"] = "in_transit"
            else:
                # Nếu đã delivered, xoá khỏi tracking
                if pkg_data["status"] == "in_transit":
                    del self.persistent_packages[pkg_id]
        # No return

    # ------------------------------------------------------------------
    # Các hàm yêu cầu bởi MultiAgentEnv
    def get_obs_agent(self, agent_id: int):
        state = self.base_env.get_state()
        return self._build_agent_obs(state, agent_id)

    def get_obs(self):
        state = self.base_env.get_state()
        return [self._build_agent_obs(state, idx) for idx in range(self.n_agents)]

    def get_obs_size(self):
        return self.obs_dim

    def get_state(self):
        # share state (centralized) – giống cho mọi agent
        return self._build_global_state(self.base_env.get_state())

    def get_state_size(self):
        return self.state_dim

    def get_avail_actions(self):
        return np.ones((self.n_agents, ACTION_DIM), dtype=np.float32)

    def get_avail_agent_actions(self, agent_id):
        return np.ones(ACTION_DIM, dtype=np.float32)

    def get_total_actions(self):
        return ACTION_DIM

    def reset(self):
        self.steps = 0
        state_dict = self.base_env.reset()
        self.persistent_packages = {}
        self.update_persistent_packages(state_dict)

        obs_n = self.get_obs()
        share_state_vec = self._build_global_state(state_dict)
        share_state_n = [share_state_vec for _ in range(self.n_agents)]
        avail_actions = self.get_avail_actions()
        return obs_n, share_state_n, avail_actions

    def step(self, actions: List[int]):
        """Thực hiện 1 bước với list[int] hành động cho từng agent."""
        prev_state_dict = self.base_env.get_state()
        env_actions = []
        for int_act in actions:
            move_idx = int(int_act) % len(MOVE_ACTIONS)
            pkg_idx = int(int_act) // len(MOVE_ACTIONS)
            move_str = MOVE_ACTIONS[move_idx]
            pkg_op_str = PKG_ACTIONS[pkg_idx] if pkg_idx < len(PKG_ACTIONS) else PKG_ACTIONS[0]
            env_actions.append((move_str, pkg_op_str))

        next_state_dict, global_reward, done, infos = self.base_env.step(env_actions)

        # Reward shaping
        shaped_global_reward = compute_shaped_rewards(
            global_reward,
            prev_state_dict,
            next_state_dict,
            env_actions,
            self.persistent_packages,
            self.n_agents,
        )

        # Cập nhật packages tracking
        self.update_persistent_packages(next_state_dict)

        # Chuẩn bị output
        obs_n = [self._build_agent_obs(next_state_dict, idx) for idx in range(self.n_agents)]
        share_state_vec = self._build_global_state(next_state_dict)
        share_state_n = [share_state_vec for _ in range(self.n_agents)]
        rewards = [[shaped_global_reward / self.n_agents] for _ in range(self.n_agents)]
        dones = [done for _ in range(self.n_agents)]
        infos_list = [infos for _ in range(self.n_agents)]
        avail_actions = self.get_avail_actions()

        self.steps += 1
        return obs_n, share_state_n, rewards, dones, infos_list, avail_actions

    def render(self, **kwargs):
        self.base_env.render()

    def close(self):
        pass

    def get_total_delivered_packages(self):
        return self.base_env.get_total_delivered_packages()

    # ------------------------------------------------------------------
    # Internal builders
    def _build_agent_obs(self, state_dict, agent_idx):
        spatial = convert_observation(state_dict, self.persistent_packages, agent_idx).flatten()
        vec = generate_vector_features(
            state_dict, self.persistent_packages, agent_idx, self.max_time_steps, self.n_agents - 1, 5
        )
        return np.concatenate([spatial, vec]).astype(np.float32)

    def _build_global_state(self, state_dict):
        g_map, g_vec = convert_global_state(state_dict, self.persistent_packages, self.max_time_steps)
        return np.concatenate([g_map.flatten(), g_vec]).astype(np.float32)