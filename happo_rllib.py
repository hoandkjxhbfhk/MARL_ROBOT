import ray
from ray import tune
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.agents.ppo import PPOTrainer
from gym import spaces
import numpy as np
from env import Environment
from sa_utils import ACTION_SPACE, IDX2ACTION

class MultiRobotEnv(MultiAgentEnv):
    def __init__(self, config):
        self.env = Environment(map_file=config.get("map_file", "map1.txt"),
                               max_time_steps=config.get("max_time_steps", 100),
                               n_robots=config.get("n_robots", 2),
                               n_packages=config.get("n_packages", 5))
        self.n_robots = config.get("n_robots", 2)
        self.map_size = (self.env.n_rows, self.env.n_cols)
        self.max_pkgs = config.get("n_packages", 5)
        obs_dim = 4 + 7 * self.max_pkgs
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(len(ACTION_SPACE))

    def reset(self):
        state = self.env.reset()
        obs = {i: state_to_vector(state, self.map_size, self.max_pkgs) for i in range(self.n_robots)}
        return obs

    def step(self, action_dict):
        # map actions
        actions = [IDX2ACTION[action_dict[i]] for i in range(self.n_robots)]
        state, reward, done, info = self.env.step(actions)
        obs = {i: state_to_vector(state, self.map_size, self.max_pkgs) for i in range(self.n_robots)}
        rewards = {i: reward for i in range(self.n_robots)}
        dones = {i: done for i in range(self.n_robots)}
        dones["__all__"] = done
        infos = {i: {} for i in range(self.n_robots)}
        return obs, rewards, dones, infos

# --- State encoder instead of autoencoder ---
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

if __name__ == "__main__":
    ray.init()
    obs_dim = 4 + 7 * 5
    obs_space = spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
    act_space = spaces.Discrete(len(ACTION_SPACE))

    tune.run(
        PPOTrainer,
        stop={"training_iteration": 1000},
        config={
            "env": MultiRobotEnv,
            "env_config": {
                "map_file": "map1.txt",
                "n_robots": 2,
                "n_packages": 5,
                "max_time_steps": 100
            },
            "multiagent": {
                "policies": {
                    "shared_policy": (None, obs_space, act_space, {})
                },
                "policy_mapping_fn": lambda agent_id, episode=None: "shared_policy",
                "policies_to_train": ["shared_policy"],
            },
            "framework": "torch",
            "num_workers": 1,
            "train_batch_size": 4000,
            "rollout_fragment_length": 200,
            "sgd_minibatch_size": 64,
            "num_sgd_iter": 10,
            "model": {
                "fcnet_hiddens": [256, 256],
                "vf_share_layers": False
            }
        }
    ) 