from env import Environment
#from agent import Agents
from greedyagent import GreedyAgents as Agents

from env import Environment
from coma_agent import COMAAgents
from maddpg_agent import MADDPGAgents

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

    env = Environment(map_file=args.map, max_time_steps=args.max_time_steps,
                      n_robots=args.num_agents, n_packages=args.n_packages,
                      seed = args.seed)
    
    agents = COMAAgents(lr_actor=1e-4, lr_critic=1e-3)  # For COMA
    
    with open(args.log_file, 'w', newline='') as csvfile:
        log_writer = csv.writer(csvfile)
        log_writer.writerow(['episode', 'total_reward', 'total_time_steps', 
                             'num_agents', 'n_packages', 'max_steps', 'map', 
                             'lr_actor', 'lr_critic'])

        for episode in range(args.num_episodes):
            # Thông báo bắt đầu episode để biết tiến độ
            print(f"Starting Episode {episode + 1}/{args.num_episodes}", flush=True)
            state = env.reset()
            agents.init_agents(state)
            done = False
            episode_reward = 0

            while not done:
                actions = agents.get_actions(state)
                next_state, reward, done, infos = env.step(actions)
                agents.remember(next_state, [reward]*args.num_agents, done)
                state = next_state
                episode_reward += reward
            
            # Cập nhật mạng một lần sau mỗi episode
            

            log_writer.writerow([episode + 1, infos.get('total_reward', episode_reward), infos.get('total_time_steps', env.t),
                                 args.num_agents, args.n_packages, args.max_steps, args.map,
                                 agents.lr_actor, agents.lr_critic])

            # Thông báo kết thúc episode
            print(f"Episode {episode + 1}/{args.num_episodes} finished", flush=True)
            print(f"  Total reward: {infos.get('total_reward', episode_reward)}", flush=True)
            print(f"  Total time steps: {infos.get('total_time_steps', env.t)}", flush=True)

    print(f"Training finished. Log saved to {args.log_file}")
