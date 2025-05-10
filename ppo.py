import argparse
import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from gym import spaces
from env import Environment
from sa_utils import ACTION_SPACE, IDX2ACTION
from autoencoder import state_to_vector
from stable_baselines3.common.callbacks import EvalCallback

class PPOEnv(gym.Env):
    def __init__(self, map_file, n_robots, n_packages, max_time_steps):
        super(PPOEnv, self).__init__()
        # Tạo environment chính
        self.env = Environment(map_file=map_file,
                               max_time_steps=max_time_steps,
                               n_robots=n_robots,
                               n_packages=n_packages)
        # Kích thước map và số packages
        self.map_size = (self.env.n_rows, self.env.n_cols)
        self.max_pkgs = n_packages
        self.n_robots = n_robots
        # Định nghĩa observation space: vector độ dài 4 + 7*max_pkgs
        obs_dim = 4 + 7 * self.max_pkgs
        self.observation_space = spaces.Box(low=0, high=1, shape=(obs_dim,), dtype=np.float32)
        # Định nghĩa action space: mỗi robot chọn 1 action trong ACTION_SPACE
        self.action_list = ACTION_SPACE
        self.idx2action = IDX2ACTION
        self.action_space = spaces.MultiDiscrete([len(self.action_list)] * self.n_robots)

    def reset(self):
        state_dict = self.env.reset()
        vec = state_to_vector(state_dict, self.map_size, self.max_pkgs)
        return vec

    def step(self, actions):
        # Chuyển chỉ số thành action tuple
        action_tuples = [self.idx2action[int(a)] for a in actions]
        next_state_dict, reward, done, info = self.env.step(action_tuples)
        vec = state_to_vector(next_state_dict, self.map_size, self.max_pkgs)
        return vec, reward, done, info

    def render(self, mode='human'):
        return self.env.render()


def main():
    parser = argparse.ArgumentParser(description="PPO training on custom env")
    parser.add_argument('--map_file', type=str, default='map1.txt')
    parser.add_argument('--n_robots', type=int, default=1)
    parser.add_argument('--n_packages', type=int, default=5)
    parser.add_argument('--max_time_steps', type=int, default=100)
    parser.add_argument('--total_timesteps', type=int, default=20000)
    parser.add_argument('--num_envs', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--n_steps', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--n_epochs', type=int, default=10)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--gae_lambda', type=float, default=0.95)
    parser.add_argument('--clip_range', type=float, default=0.2)
    parser.add_argument('--ent_coef', type=float, default=0.01)
    parser.add_argument('--vf_coef', type=float, default=0.5)
    parser.add_argument('--tensorboard_log', type=str, default='./ppo_tensorboard/')
    args = parser.parse_args()

    # Tạo vectorized environment với Subproc và chuẩn hóa
    env_fns = [lambda: PPOEnv(map_file=args.map_file,
                              n_robots=args.n_robots,
                              n_packages=args.n_packages,
                              max_time_steps=args.max_time_steps)
               for _ in range(args.num_envs)]
    env = SubprocVecEnv(env_fns)
    env = VecNormalize(env, norm_obs=True, norm_reward=True)
    # Tạo env cho đánh giá
    eval_env = DummyVecEnv(env_fns[:1])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True, training=False)
    eval_callback = EvalCallback(eval_env,
                                 best_model_save_path='./logs/best_model',
                                 log_path='./logs/eval_logs',
                                 eval_freq=10000,
                                 deterministic=True,
                                 render=False)

    # Khởi tạo PPO với siêu tham số cải tiến và mạng mạnh hơn
    model = PPO("MlpPolicy", env, verbose=1,
                learning_rate=args.learning_rate,
                n_steps=args.n_steps,
                batch_size=args.batch_size,
                n_epochs=args.n_epochs,
                gamma=args.gamma,
                gae_lambda=args.gae_lambda,
                clip_range=args.clip_range,
                ent_coef=args.ent_coef,
                vf_coef=args.vf_coef,
                policy_kwargs={'net_arch': [dict(pi=[256,256], vf=[256,256])]},
                tensorboard_log=args.tensorboard_log)
    # Huấn luyện với callback đánh giá
    model.learn(total_timesteps=args.total_timesteps, callback=eval_callback)
    model.save("ppo_model_normalized")

    # Đánh giá nhanh
    obs = env.reset()
    done = False
    average_reward = 0
    
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        average_reward += reward
        
    print(f"Average reward: {average_reward}")

if __name__ == '__main__':
    main()
