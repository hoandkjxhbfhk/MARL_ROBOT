import sys, os
# Thêm path tới thư mục MAT envs
sys.path.append(os.path.join(os.path.dirname(__file__), 'Multi-Agent-Transformer', 'mat'))
from envs.delivery_env import DeliveryEnv
from maddpg_agent import MADDPGAgents
from greedyagent import GreedyAgents as Agents
from coma_agent import COMAAgents

import numpy as np
import csv
import time

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Multi-Agent Reinforcement Learning for Delivery")
    parser.add_argument("--num_agents", type=int, default=5, help="Number of agents")
    parser.add_argument("--n_packages", type=int, default=10, help="Number of packages")
    parser.add_argument("--max_steps", type=int, default=100, help="Maximum number of steps per episode")
    parser.add_argument("--seed", type=int, default=2025, help="Random seed for reproducibility")
    parser.add_argument("--max_time_steps", type=int, default=100, help="Maximum time steps for the environment")
    parser.add_argument("--map", type=str, default="map.txt", help="Map name")
    parser.add_argument("--num_episodes", type=int, default=100, help="Number of episodes to train")
    
    # Chuẩn bị tên file log mặc định để tránh lỗi phân tích f-string với strftime
    current_time_for_log = time.strftime("%Y%m%d-%H%M%S")
    default_log_file_name = f"training_log_{current_time_for_log}.csv"
    parser.add_argument("--log_file", type=str, default=default_log_file_name, help="CSV file to log training data")

    args = parser.parse_args()
    np.random.seed(args.seed)

    # Sử dụng DeliveryEnv để nhận quan sát đầy đủ
    env = DeliveryEnv(args)
    
    agents = MADDPGAgents(lr_actor=1e-4, lr_critic=1e-3)  # For MADDPG
    
    with open(args.log_file, 'w', newline='') as csvfile:
        log_writer = csv.writer(csvfile)
        log_writer.writerow(['episode', 'total_reward', 'total_time_steps', 
                             'num_agents', 'n_packages', 'max_steps', 'map', 
                             'lr_actor', 'lr_critic'])

        for episode in range(args.num_episodes):
            # Thông báo bắt đầu episode để biết tiến độ
            print(f"Starting Episode {episode + 1}/{args.num_episodes}", flush=True)
            # Khởi tạo với quan sát đầy đủ cho mỗi agent
            obs_n = env.reset()
            agents.init_agents(obs_n)
            done = False
            episode_reward = 0

            while not done:
                # Lấy hành động từ danh sách quan sát
                actions = agents.get_actions(obs_n)
                # Thực thi hành động và nhận quan sát mới
                next_obs_n, reward_n, done_n, infos_n = env.step(actions)
                # Flatten reward list-of-lists thành list các float
                rewards = [r[0] if isinstance(r, (list, tuple, np.ndarray)) else r for r in reward_n]
                # Lấy reward chung (giả sử giống nhau cho tất cả agents)
                reward_scalar = rewards[0]
                # Xác định điều kiện kết thúc khi tất cả agents done
                done = all(done_n)
                # Lấy info từ agent đầu tiên
                infos = infos_n[0]
                # Ghi nhớ trải nghiệm với danh sách rewards
                agents.remember(next_obs_n, rewards, done)
                obs_n = next_obs_n
                episode_reward += reward_scalar
            
            # Cập nhật mạng một lần sau mỗi episode
            

            log_writer.writerow([episode + 1,
                                 infos.get('total_reward', episode_reward),
                                 infos.get('total_time_steps', env.base_env.t),
                                 args.num_agents, args.n_packages, args.max_steps, args.map,
                                 agents.lr_actor, agents.lr_critic])

            # Thông báo kết thúc episode
            print(f"Episode {episode + 1}/{args.num_episodes} finished", flush=True)
            print(f"  Total reward: {infos.get('total_reward', episode_reward)}", flush=True)
            print(f"  Total time steps: {infos.get('total_time_steps', env.base_env.t)}", flush=True)

    print(f"Training finished. Log saved to {args.log_file}")
