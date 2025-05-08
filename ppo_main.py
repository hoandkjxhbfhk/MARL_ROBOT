import torch
import numpy as np
import csv
import time
import argparse

from env import Environment # Assuming your Environment class is in env.py
from ppo import PPO_Agent   # Assuming your PPO_Agent class is in ppo.py

# --- Helper function to extract observation for a single agent ---
# This needs to match how you define observations for your PPO agents.
# This is a placeholder based on previous agent implementations.
def _extract_agent_obs(env_state, agent_id, num_map_rows, num_map_cols):
    """Extracts a simple observation for a single agent.
    Args:
        env_state (dict): The current state from the Environment.
        agent_id (int): The ID of the agent for whom to extract the observation.
        num_map_rows (int): Number of rows in the map for normalization.
        num_map_cols (int): Number of columns in the map for normalization.
    Returns:
        np.array: The observation vector for the agent.
    """
    robot_info = env_state['robots'][agent_id]
    robot_row = robot_info[0] / num_map_rows  # Normalize row
    robot_col = robot_info[1] / num_map_cols  # Normalize col
    carrying_status = float(robot_info[2] > 0) # 1 if carrying, 0 otherwise
    # Consider adding more features: package locations, other robot locations (relative?)
    # For simplicity, let's use a basic observation like in MADDPG/COMA
    # time_step_norm = env_state['time_step'] / env_state.get('max_time_steps', 1000.0) # Assuming max_time_steps in state

    # For PPO, often a fixed-size observation is easier. 
    # Let's stick to simple (row, col, carrying) for now.
    # The obs_dim in PPO_Agent constructor must match this.
    # If you add time, ensure your env.reset() also gives this structure initially.
    # obs = np.array([robot_row, robot_col, carrying_status, time_step_norm], dtype=np.float32)
    obs = np.array([robot_row, robot_col, carrying_status], dtype=np.float32)
    return obs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Independent PPO for Multi-Agent Delivery")
    # Environment Args
    parser.add_argument("--num_agents", type=int, default=2, help="Number of agents")
    parser.add_argument("--n_packages", type=int, default=5, help="Number of packages")
    parser.add_argument("--map", type=str, default="map1.txt", help="Map name")
    parser.add_argument("--seed", type=int, default=2024, help="Random seed")
    parser.add_argument("--max_episode_steps", type=int, default=200, help="Maximum steps per episode in env")
    
    # PPO Agent Args
    parser.add_argument("--obs_dim_ppo", type=int, default=3, help="Observation dimension for PPO agent (must match _extract_agent_obs)")
    parser.add_argument("--lr_actor", type=float, default=3e-4, help="Actor learning rate")
    parser.add_argument("--lr_critic", type=float, default=1e-3, help="Critic learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae_lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--policy_clip", type=float, default=0.2, help="PPO policy clip range")
    parser.add_argument("--n_epochs", type=int, default=10, help="PPO update epochs")
    parser.add_argument("--ppo_batch_size", type=int, default=64, help="PPO mini-batch size")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Entropy coefficient")
    parser.add_argument("--vf_coef", type=float, default=0.5, help="Value function loss coefficient")
    parser.add_argument("--max_grad_norm", type=float, default=0.5, help="Max gradient norm for clipping")
    parser.add_argument("--rollout_steps", type=int, default=2048, help="Steps to collect per rollout before PPO update")

    # Training Args
    parser.add_argument("--num_train_episodes", type=int, default=1000, help="Number of episodes to train")
    current_time_str = time.strftime("%Y%m%d-%H%M%S")
    default_log_filename = f"ppo_training_log_{current_time_str}.csv"
    parser.add_argument("--log_file", type=str, default=default_log_filename, help="CSV file to log training data")
    parser.add_argument("--save_model_interval", type=int, default=100, help="Interval to save models (episodes)")
    parser.add_argument("--model_save_dir", type=str, default="ppo_models", help="Directory to save models")

    args = parser.parse_args()

    # Seed for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Initialize Environment
    # Note: env.max_time_steps is used by the Environment class itself for termination.
    # args.max_episode_steps is used here in the main loop to control episode length for training rollouts.
    env = Environment(map_file=args.map, max_time_steps=args.max_episode_steps, 
                      n_robots=args.num_agents, n_packages=args.n_packages,
                      seed=args.seed)
    
    # For observation normalization if needed later
    num_map_rows = env.n_rows
    num_map_cols = env.n_cols

    # Initialize PPO Agents (one for each robot)
    agents = []
    for i in range(args.num_agents):
        agent = PPO_Agent(obs_dim=args.obs_dim_ppo, # Ensure this matches _extract_agent_obs output
                          lr_actor=args.lr_actor, lr_critic=args.lr_critic,
                          gamma=args.gamma, gae_lambda=args.gae_lambda, policy_clip=args.policy_clip,
                          n_epochs=args.n_epochs, ppo_batch_size=args.ppo_batch_size,
                          entropy_coef=args.entropy_coef, vf_coef=args.vf_coef, 
                          max_grad_norm=args.max_grad_norm)
        agents.append(agent)
    
    print(f"Starting PPO training with {args.num_agents} agents on map '{args.map}'. Logging to '{args.log_file}'")

    # Prepare CSV logging
    with open(args.log_file, 'w', newline='') as csvfile:
        log_writer = csv.writer(csvfile)
        header = ['episode', 'total_steps_this_episode', 'cumulative_reward']
        for i in range(args.num_agents):
            header.append(f'agent_{i}_reward')
        log_writer.writerow(header)

        global_step_counter = 0

        for episode_num in range(1, args.num_train_episodes + 1):
            current_env_state = env.reset()
            episode_done = False
            current_episode_steps = 0
            # Dùng reward tổng thể (scalar)
            episode_cumulative_reward = 0.0

            # Collect one rollout or until episode ends
            # The PPO agent's memory is cleared after each learn() call.
            # So, we collect one full rollout for each agent before learning.
            
            # PPO typically collects a fixed number of steps (rollout_steps) 
            # or until the episode ends, across all agents if run in parallel.
            # For simplicity here, we will run sequentially for rollout_steps OR until done.
            # More advanced: use a shared buffer or collect T steps per agent in parallel.

            for t_rollout in range(args.rollout_steps):
                if episode_done:
                    break

                joint_env_action = [] # List of (move_str, pkg_str) for env.step()
                
                # Store details for each agent for this step
                # These will be fed into store_transition later
                step_observations = []
                step_action_move_indices = []
                step_action_pkg_indices = []
                step_log_probs_move = []
                step_log_probs_pkg = []
                step_values = [] 

                for agent_id in range(args.num_agents):
                    agent_obs = _extract_agent_obs(current_env_state, agent_id, num_map_rows, num_map_cols)
                    
                    env_action, move_idx, pkg_idx, log_p_move, log_p_pkg, value = agents[agent_id].select_action(agent_obs)
                    
                    joint_env_action.append(env_action)
                    step_observations.append(agent_obs)
                    step_action_move_indices.append(move_idx)
                    step_action_pkg_indices.append(pkg_idx)
                    step_log_probs_move.append(log_p_move)
                    step_log_probs_pkg.append(log_p_pkg)
                    step_values.append(value)
                
                # Execute joint action in the environment, nhận reward scalar
                next_env_state, reward, episode_done, infos = env.step(joint_env_action)

                # Store transition cho từng agent, dùng reward chung
                for agent_id in range(args.num_agents):
                    agents[agent_id].store_transition(
                        obs=step_observations[agent_id],
                        action_move_idx=step_action_move_indices[agent_id],
                        action_pkg_idx=step_action_pkg_indices[agent_id],
                        log_prob_move=step_log_probs_move[agent_id],
                        log_prob_pkg=step_log_probs_pkg[agent_id],
                        reward=reward,  # Global scalar reward
                        done=episode_done,
                        value=step_values[agent_id]
                    )
                # Cộng dồn tổng reward của episode
                episode_cumulative_reward += reward
                
                current_env_state = next_env_state
                current_episode_steps += 1
                global_step_counter +=1

            # After collecting rollout or if episode ended, update agents
            for agent_id in range(args.num_agents):
                last_val_bootstrap = 0.0
                if not episode_done: # If rollout ended but episode continues
                    # Get obs for the agent at the step AFTER the rollout ended
                    agent_final_obs = _extract_agent_obs(current_env_state, agent_id, num_map_rows, num_map_cols)
                    agent_final_obs_t = torch.tensor(agent_final_obs, dtype=torch.float32, device=agents[agent_id].device).unsqueeze(0)
                    with torch.no_grad():
                        last_val_bootstrap = agents[agent_id].critic(agent_final_obs_t).item()
                
                agents[agent_id].learn(last_state_value_bootstrap=last_val_bootstrap)
            
            # Logging (ghi reward tổng thể)
            log_row = [episode_num, current_episode_steps, episode_cumulative_reward]
            log_writer.writerow(log_row)
            csvfile.flush() # Ensure data is written

            print(f"Episode: {episode_num}, Steps: {current_episode_steps}, Total Reward: {episode_cumulative_reward:.2f}")

            # Save models periodically
            # (Add model saving logic here if needed, e.g., torch.save(agent.actor.state_dict(), ...))

    print("PPO Training finished.") 