import argparse
import numpy as np
import csv
import time

from ppo import PPOEnv
from stable_baselines3 import PPO

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark trained PPO model")
    parser.add_argument("--model_path", type=str, default="ppo_model", help="Path to trained PPO model")
    parser.add_argument("--num_agents", type=int, default=1, help="Number of agents")
    parser.add_argument("--n_packages", type=int, default=5, help="Number of packages")
    parser.add_argument("--max_time_steps", type=int, default=100, help="Maximum time steps per episode")
    parser.add_argument("--maps", type=str, default="map1.txt,map2.txt,map3.txt,map4.txt,map5.txt", help="Comma-separated list of map files")
    parser.add_argument("--num_episodes", type=int, default=10, help="Number of episodes per map")
    parser.add_argument("--seed", type=int, default=2025, help="Random seed")
    default_log = f"ppo_benchmark_{time.strftime('%Y%m%d-%H%M%S')}.csv"
    parser.add_argument("--log_file", type=str, default=default_log, help="CSV file to log results")

    args = parser.parse_args()
    np.random.seed(args.seed)

    map_list = [m.strip() for m in args.maps.split(',') if m.strip()]

    # Load trained PPO model
    model = PPO.load(args.model_path)

    # Benchmark
    with open(args.log_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['map', 'episode', 'total_reward', 'total_time_steps'])

        for map_file in map_list:
            print(f"Evaluating on map: {map_file}")
            for ep in range(1, args.num_episodes + 1):
                env = PPOEnv(map_file=map_file,
                             n_robots=args.num_agents,
                             n_packages=args.n_packages,
                             max_time_steps=args.max_time_steps)
                obs = env.reset()
                done = False
                total_reward = 0.0
                total_steps = 0

                while not done:
                    action, _states = model.predict(obs, deterministic=True)
                    obs, reward, done, info = env.step(action)
                    total_reward += reward
                    total_steps += 1

                # Collect final metrics from info if available
                final_reward = info.get('total_reward', total_reward)
                final_steps = info.get('total_time_steps', total_steps)
                writer.writerow([map_file, ep, final_reward, final_steps])
                print(f" Episode {ep}/{args.num_episodes}: Reward={final_reward:.2f}, Steps={final_steps}")

    print(f"Benchmark completed. Results saved to {args.log_file}") 