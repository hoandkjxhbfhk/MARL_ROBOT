import argparse
import numpy as np
import pickle
from collections import deque

from env import Environment
from sa_utils import StateEncoder, ACTION_SPACE, encode_action_tuple, decode_action_idx


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


def value_iteration(env, gamma=0.99, eps=1e-4):
    """
    Thực hiện Value Iteration cho env với một robot.
    Returns:
        Q: np.array shape (n_states, n_actions)
        policy: dict mapping state_idx -> action_idx
        encoder: StateEncoder instance chứa mapping state<->idx
    """
    encoder = StateEncoder()
    # Khởi tạo trạng thái ban đầu
    init_snapshot = env.get_full_state()
    init_comp = compress_snapshot(init_snapshot, env.n_rows, env.n_cols)
    encoder.encode(init_comp)

    # Thu thập tập trạng thái khả dĩ qua BFS
    queue = deque([init_snapshot])
    while queue:
        snapshot = queue.popleft()
        env.set_full_state(snapshot)
        # nén snapshot
        comp = compress_snapshot(snapshot, env.n_rows, env.n_cols)
        for action in ACTION_SPACE:
            next_state_dict, reward, done, _ = env.step([action])
            next_snapshot = env.get_full_state()
            comp_next = compress_snapshot(next_snapshot, env.n_rows, env.n_cols)
            idx = encoder.encode(comp_next)
            # Nếu là state mới (theo compressed key), thêm vào queue
            if idx == len(encoder.idx2state) - 1:
                queue.append(next_snapshot)
    
    n_states = len(encoder.state2idx)
    n_actions = len(ACTION_SPACE)
    Q = np.zeros((n_states, n_actions), dtype=np.float64)

    # Value Iteration
    delta = eps + 1.0
    while delta > eps:
        delta = 0.0
        for comp_key, state_idx in encoder.state2idx.items():
            # lấy snapshot từ idx
            state = encoder.idx2state[state_idx]
            # state là compressed tuple, không phải full snapshot, nên ta cần decode không full
            # Thay vì restore full snapshot, ta chỉ cần mô phỏng từng snapshot ban đầu dùng BFS thu thập
            # Do tính chất approximation, ta bỏ set_full_state ở đây (bắt buộc phải dùng full state env)
            # TODO: nếu cần, lưu mapping idx->full snapshot trong encoder
            env.set_full_state(state)
            for a_idx, action in enumerate(ACTION_SPACE):
                next_state_dict, reward, done, _ = env.step([action])
                next_snapshot = env.get_full_state()
                comp_next = compress_snapshot(next_snapshot, env.n_rows, env.n_cols)
                next_idx = encoder.encode(comp_next)
                # Q update
                q_val = reward + (0.0 if done else gamma * np.max(Q[next_idx]))
                diff = abs(q_val - Q[state_idx, a_idx])
                delta = max(delta, diff)
                Q[state_idx, a_idx] = q_val
    
    # Derive greedy policy
    policy = {s_idx: int(np.argmax(Q[s_idx])) for s_idx in range(n_states)}
    return Q, policy, encoder


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Value Iteration for 1-robot env')
    parser.add_argument('--map', type=str, default='map1.txt', help='Map file')
    parser.add_argument('--max_time_steps', type=int, default=100, help='Max time steps in env')
    parser.add_argument('--n_packages', type=int, default=5, help='Number of packages')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--eps', type=float, default=1e-4, help='Convergence threshold')
    parser.add_argument('--out_q', type=str, default='q_values.pkl', help='Output pickle for Q')
    parser.add_argument('--out_policy', type=str, default='policy_init.pkl', help='Output pickle for policy')
    args = parser.parse_args()

    env = Environment(map_file=args.map,
                      max_time_steps=args.max_time_steps,
                      n_robots=1,
                      n_packages=args.n_packages,
                      seed=args.seed)
    # Reset để khởi đầu tại state 0
    env.reset()

    Q, policy, encoder = value_iteration(env, gamma=args.gamma, eps=args.eps)

    # Lưu ra file
    with open(args.out_q, 'wb') as f:
        pickle.dump({'Q': Q, 'state_encoder': encoder}, f)
    with open(args.out_policy, 'wb') as f:
        pickle.dump({'policy': policy, 'state_encoder': encoder}, f)

    print(f'Value Iteration xong: n_states={len(encoder.state2idx)}, n_actions={len(ACTION_SPACE)}')
    print(f'Q lưu tại {args.out_q}, policy lưu tại {args.out_policy}') 