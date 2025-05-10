#!/usr/bin/env python
import sys, os
# Thêm project root và MAT code vào đường dẫn
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
# Đưa thư mục gốc dự án (baaa) vào sys.path để import env.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))

import socket
import setproctitle
import numpy as np
import torch
from pathlib import Path

from mat.config import get_config
from mat.envs.delivery_env import DeliveryEnv
from mat.runner.shared.mpe_runner import MPERunner as Runner  # Runner sẽ ghi log vào SummaryWriter
from mat.envs.env_wrappers import SubprocVecEnv, DummyVecEnv


def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            env = DeliveryEnv(all_args)
            env.seed(all_args.seed + rank * 1000)
            return env
        return init_env
    if all_args.n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            env = DeliveryEnv(all_args)
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env
        return init_env
    if all_args.n_eval_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])


def parse_args(args, parser):
    parser.add_argument('--map', type=str, default='map1.txt', help='Map file')
    parser.add_argument('--n_packages', type=int, default=5)
    parser.add_argument('--num_agents', type=int, default=1)
    parser.add_argument('--max_time_steps', type=int, default=100)
    all_args = parser.parse_known_args(args)[0]
    # Gán tên env cho MAT
    all_args.env_name = 'Delivery'
    # Gán scenario_name để Runner và mpe_runner sử dụng
    all_args.scenario_name = os.path.basename(all_args.map)
    return all_args


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    # Thiết lập device
    if all_args.cuda and torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    # Thư mục lưu kết quả
    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + "/results") / all_args.env_name / all_args.algorithm_name / all_args.experiment_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Tạo thư mục run_dir cho logging file
    run_dir = run_dir / 'run1'
    run_dir.mkdir(parents=True, exist_ok=True)

    setproctitle.setproctitle(f"{all_args.algorithm_name}-{all_args.env_name}-{all_args.experiment_name}@{all_args.user_name}")

    # Đặt seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # Tạo env
    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None
    num_agents = all_args.num_agents

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir
    }

    runner = Runner(config)
    runner.run()

    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()

    # Ghi log kết quả training ra file JSON
    log_path = run_dir / 'delivery_summary.json'
    runner.writter.export_scalars_to_json(str(log_path))
    runner.writter.close()


if __name__ == "__main__":
    main(sys.argv[1:]) 