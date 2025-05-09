import argparse
import numpy as np
import pickle
from collections import deque

from env import Environment
from sa_utils import StateEncoder, ACTION_SPACE, ACTION2IDX
import torch
from autoencoder import Autoencoder, VAE, state_to_vector
from greedyagent import GreedyAgents


def compress_snapshot(snapshot, n_rows, n_cols):
    """Trích xuất state được nén: (robot_row, robot_col, carrying_flag, dist_to_goal_or_nearest_waiting)"""
    # Robot position (0-indexed)
    robot = snapshot['robots'][0]
    r, c = robot.position
    carrying = robot.carrying
    if carrying == 0:
        # Tìm khoảng cách nhỏ nhất tới các gói đang chờ
        dists = []
        for pkg in snapshot['packages']:
            if pkg.status == 'waiting':
                sr, sc = pkg.start
                dists.append(abs(sr - r) + abs(sc - c))
        dist = min(dists) if dists else 0
        flag = 0
    else:
        # Đang mang, tính khoảng cách tới target
        pkg = snapshot['packages'][carrying - 1]
        tr, tc = pkg.target
        dist = abs(tr - r) + abs(tc - c)
        flag = 1
    return (r, c, flag, dist)


def encode_latent(ae_model, snapshot, map_size, max_pkgs, use_vae, device):
    vec = state_to_vector(snapshot, map_size, max_pkgs)
    x = torch.tensor(vec, dtype=torch.float32).unsqueeze(0).to(device)
    ae_model.eval()
    with torch.no_grad():
        if use_vae:
            mu, _ = ae_model.encode(x)
            z = mu.squeeze(0).cpu().numpy()
        else:
            z = ae_model.encode(x).squeeze(0).cpu().numpy()
    return tuple(z.tolist())


def compute_greedy_q_values(
    env, ae_model, use_vae, device, map_size, max_pkgs,
    num_greedy_episodes, max_steps_per_episode,
    gamma=0.99, alpha=0.1, q_iterations=100
):
    """
    Tính toán Q-values cho policy của GreedyAgent bằng cách mô phỏng và policy evaluation.
    """
    greedy_agent = GreedyAgents()
    
    transitions = []
    print(f"Collecting transitions using GreedyAgent for {num_greedy_episodes} episodes...")
    for i_episode in range(num_greedy_episodes):
        current_env_state = env.reset()
        greedy_agent.init_agents(current_env_state.copy())
        
        current_full_snap = env.get_full_state()

        for i_step in range(max_steps_per_episode):
            action_list = greedy_agent.get_actions(current_env_state.copy()) 
            action_tuple = action_list[0]

            next_env_state, reward, done, _ = env.step([action_tuple])
            next_full_snap = env.get_full_state()

            transitions.append({
                "s_full": current_full_snap,
                "a_tuple": action_tuple,
                "r": reward,
                "s_prime_full": next_full_snap,
                "done": done,
                "s_env": current_env_state,
                "s_prime_env": next_env_state
            })
            
            current_full_snap = next_full_snap
            current_env_state = next_env_state
            
            if done:
                break
        if (i_episode + 1) % 10 == 0 or (i_episode + 1) == num_greedy_episodes:
            print(f"  Episode {i_episode + 1}/{num_greedy_episodes} finished. Total transitions: {len(transitions)}")

    print(f"Collected {len(transitions)} transitions.")

    if not transitions:
        print("No transitions collected. Exiting.")
        return np.array([]), {}, StateEncoder()

    state_encoder = StateEncoder()
    all_full_snaps = set()
    for t in transitions:
        all_full_snaps.add(pickle.dumps(t["s_full"]))
        all_full_snaps.add(pickle.dumps(t["s_prime_full"]))

    print(f"Encoding {len(all_full_snaps)} unique full snapshots...")
    for pickled_snap in all_full_snaps:
        snap = pickle.loads(pickled_snap)
        comp_snap = encode_latent(ae_model, snap, map_size, max_pkgs, use_vae, device)
        state_encoder.encode(comp_snap)
    
    num_unique_compressed_states = len(state_encoder.state2idx)
    num_actions_env = len(ACTION_SPACE)
    Q_table = np.zeros((num_unique_compressed_states, num_actions_env))
    print(f"Initialized Q-table with shape: ({num_unique_compressed_states}, {num_actions_env})")

    print(f"Performing {q_iterations} Q-iterations for policy evaluation...")
    for i_q_iter in range(q_iterations):
        total_q_change = 0
        
        temp_greedy_agent_for_s_prime = GreedyAgents()

        for t in transitions:
            s_full = t["s_full"]
            a_tuple = t["a_tuple"]
            r = t["r"]
            s_prime_full = t["s_prime_full"]
            done = t["done"]
            s_prime_env = t["s_prime_env"]

            comp_s = encode_latent(ae_model, s_full, map_size, max_pkgs, use_vae, device)
            s_idx = state_encoder.encode(comp_s)
            
            a_idx = ACTION2IDX.get(a_tuple) 
            if a_idx is None:
                print(f"Warning: Action {a_tuple} not in ACTION2IDX. Skipping transition.")
                continue

            old_q_value = Q_table[s_idx, a_idx]
            
            target_q_s_prime = 0
            if not done:
                temp_greedy_agent_for_s_prime.init_agents(s_prime_env.copy())
                action_list_prime = temp_greedy_agent_for_s_prime.get_actions(s_prime_env.copy())
                a_prime_tuple = action_list_prime[0]
                
                a_prime_idx = ACTION2IDX.get(a_prime_tuple)
                if a_prime_idx is None:
                    print(f"Warning: Greedy's next action {a_prime_tuple} not in ACTION2IDX. Assuming 0 Q-value for next state.")
                else:
                    comp_s_prime = encode_latent(ae_model, s_prime_full, map_size, max_pkgs, use_vae, device)
                    s_prime_idx = state_encoder.encode(comp_s_prime)
                    target_q_s_prime = Q_table[s_prime_idx, a_prime_idx]
            
            target = r + gamma * target_q_s_prime
            Q_table[s_idx, a_idx] = (1 - alpha) * old_q_value + alpha * target
            total_q_change += abs(Q_table[s_idx, a_idx] - old_q_value)

        print(f"  Q-iteration {i_q_iter + 1}/{q_iterations}, Total Q change: {total_q_change:.4f}")
        if total_q_change < 1e-5 and i_q_iter > 10:
             print("  Q-values converged.")
             break
             
    policy = {s_idx: int(np.argmax(Q_table[s_idx])) for s_idx in range(num_unique_compressed_states)}
    
    return Q_table, policy, state_encoder


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Greedy Agent Q-Value Estimation for Warm Starting RL')
    parser.add_argument('--map', type=str, default='map1.txt', help='Map file')
    parser.add_argument('--max_time_steps', type=int, default=100, help='Max time steps in env for greedy episode collection')
    parser.add_argument('--n_packages', type=int, default=5, help='Number of packages')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--ae_model', type=str, required=True, help='Path to autoencoder/vae model .pth')
    parser.add_argument('--use_vae', action='store_true', help='Set nếu model là VAE (mặc định False nếu không có)')
    parser.add_argument('--latent_dim', type=int, default=8, help='Latent dimension của AE/VAE')
    parser.add_argument('--max_pkgs', type=int, default=None, help='Max packages to encode for AE (default = n_packages in env)')
    parser.add_argument('--num_greedy_episodes', type=int, default=100, help='Number of episodes to run GreedyAgent for data collection')
    parser.add_argument('--q_gamma', type=float, default=0.99, help='Discount factor for Q-value calculation')
    parser.add_argument('--q_alpha', type=float, default=0.1, help='Learning rate for Q-value updates')
    parser.add_argument('--q_iterations', type=int, default=1000, help='Number of iterations for Q-value policy evaluation')
    parser.add_argument('--out_q', type=str, default='q_values_greedy.pkl', help='Output pickle for Q-table of GreedyAgent')
    parser.add_argument('--out_policy', type=str, default='policy_greedy_init.pkl', help='Output pickle for policy derived from Q-table')
    args = parser.parse_args()

    if args.max_pkgs is None:
        args.max_pkgs = args.n_packages

    env = Environment(map_file=args.map,
                      max_time_steps=args.max_time_steps,
                      n_robots=1,
                      n_packages=args.n_packages,
                      seed=args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_dim = 4 + 7 * args.max_pkgs
    
    if args.use_vae:
        print("Loading VAE model...")
        ae_model = VAE(input_dim, args.latent_dim).to(device)
    else:
        print("Loading Autoencoder model...")
        ae_model = Autoencoder(input_dim, args.latent_dim).to(device)
    
    state_dict = torch.load(args.ae_model, map_location=device)
    ae_model.load_state_dict(state_dict, strict=False)
    print(f"Loaded {'VAE' if args.use_vae else 'Autoencoder'} model from {args.ae_model}")

    map_size_env = (env.n_rows, env.n_cols)
    
    Q_table, policy, state_encoder_obj = compute_greedy_q_values(
        env, ae_model, args.use_vae, device, map_size_env, args.max_pkgs,
        args.num_greedy_episodes, args.max_time_steps,
        gamma=args.q_gamma, alpha=args.q_alpha, q_iterations=args.q_iterations
    )

    if Q_table.size > 0:
        with open(args.out_q, 'wb') as f:
            pickle.dump({'Q_table': Q_table, 'state_encoder': state_encoder_obj}, f)
        with open(args.out_policy, 'wb') as f:
            pickle.dump({'policy_map': policy, 'state_encoder': state_encoder_obj}, f)

        print(f'Greedy Q-value estimation finished: n_compressed_states={len(state_encoder_obj.state2idx)}, n_actions={Q_table.shape[1] if Q_table.ndim == 2 else 0}')
        print(f'Q-table for GreedyAgent saved at {args.out_q}')
        print(f'Policy derived from Q-table saved at {args.out_policy}')
    else:
        print("Q-table computation resulted in an empty table. Nothing saved.") 