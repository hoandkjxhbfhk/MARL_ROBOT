import argparse
import numpy as np
import csv
import time

from env import Environment
#from agent import Agents
from greedyagent import GreedyAgents as Agents

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark GreedyAgents on multiple maps")
    parser.add_argument("--num_agents", type=int, default=5, help="Number of agents")
    parser.add_argument("--n_packages", type=int, default=10, help="Number of packages")
    parser.add_argument("--max_time_steps", type=int, default=1000, help="Maximum time steps per episode")
    parser.add_argument("--maps", type=str, default="map1.txt,map2.txt,map3.txt,map4.txt,map5.txt", help="Comma-separated list of map files")
    parser.add_argument("--num_episodes", type=int, default=50, help="Number of episodes per map")
    parser.add_argument("--seed", type=int, default=2025, help="Random seed for reproducibility")
    default_log = f"greedy_benchmark_{time.strftime('%Y%m%d-%H%M%S')}.csv"
    parser.add_argument("--log_file", type=str, default=default_log, help="CSV file to log results")

    args = parser.parse_args()
    np.random.seed(args.seed)

    map_list = [m.strip() for m in args.maps.split(',') if m.strip()]

    with open(args.log_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['map', 'episode', 'total_reward', 'total_time_steps'])

        for map_file in map_list:
            print(f"Running benchmark on map: {map_file}")
            for ep in range(1, args.num_episodes + 1):
                env = Environment(map_file=map_file, max_time_steps=args.max_time_steps,
                                  n_robots=args.num_agents, n_packages=args.n_packages,
                                  seed=args.seed)
                state = env.reset()
                agents = Agents()
                agents.init_agents(state)
                done = False
                while not done:
                    actions = agents.get_actions(state)
                    state, reward, done, infos = env.step(actions)
                total_r = infos.get('total_reward', 0)
                total_steps = infos.get('total_time_steps', env.t)
                writer.writerow([map_file, ep, total_r, total_steps])
                print(f" Map {map_file}, Episode {ep}/{args.num_episodes}, Reward: {total_r:.2f}, Steps: {total_steps}")
    print(f"Benchmark completed. Results saved to {args.log_file}")