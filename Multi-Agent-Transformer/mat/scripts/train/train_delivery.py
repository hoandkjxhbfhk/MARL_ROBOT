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
from mat.envs.env_mat import DeliveryMATEnv
from mat.runner.shared.delivery_runner import DeliveryRunner as Runner
from mat.envs.env_wrappers import ShareSubprocVecEnv, ShareDummyVecEnv

"""Train script for Delivery environment using MAT."""

def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "delivery":
                env_args = {
                    "map_file": all_args.map_file,
                    "n_robots": all_args.n_agents,
                    "n_packages": all_args.n_packages,
                    "max_time_steps": all_args.max_time_steps,
                    "move_cost": all_args.move_cost,
                    "delivery_reward": all_args.delivery_reward,
                    "delay_reward": all_args.delay_reward,
                    "seed": all_args.seed + rank * 1000,
                }
                env = DeliveryMATEnv(env_args=env_args)
            else:
                raise NotImplementedError(f"Unsupported env {all_args.env_name}")
            return env
        return init_env

    if all_args.n_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            env = DeliveryMATEnv(all_args)
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env
        return init_env
    if all_args.n_eval_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])


def parse_args(args, parser):
    parser.add_argument("--map_file", type=str, default="map1.txt")
    parser.add_argument("--n_agents", type=int, default=5)
    parser.add_argument("--n_packages", type=int, default=50)
    parser.add_argument("--max_time_steps", type=int, default=500)
    parser.add_argument("--move_cost", type=float, default=-0.01)
    parser.add_argument("--delivery_reward", type=float, default=10.0)
    parser.add_argument("--delay_reward", type=float, default=1.0)
    all_args = parser.parse_known_args(args)[0]
    return all_args


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    # Override env_name and algorithm check
    all_args.env_name = "delivery"
    if all_args.algorithm_name == "mat_dec":
        all_args.dec_actor = True
        all_args.share_actor = True
    elif all_args.algorithm_name not in ["mat", "mat_gru", "mat_encoder", "mat_decoder", "mat_dec"]:
        raise NotImplementedError("Unsupported algorithm for delivery")

    # device
    if all_args.cuda and torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
    else:
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    run_dir = Path(
        os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
        + "/results"
    ) / all_args.env_name / all_args.algorithm_name / all_args.experiment_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # choose run id
    if not all_args.use_wandb:
        if not run_dir.exists():
            curr_run = "run1"
        else:
            exst = [int(p.name.split("run")[-1]) for p in run_dir.iterdir() if p.name.startswith("run")]
            curr_run = f"run{max(exst)+1}" if exst else "run1"
        run_dir = run_dir / curr_run
        run_dir.mkdir(parents=True, exist_ok=True)

    setproctitle.setproctitle(
        f"{all_args.algorithm_name}-{all_args.env_name}-{all_args.experiment_name}@{all_args.user_name}"
    )

    # seeds
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    envs = make_train_env(all_args)
    eval_envs = None
    num_agents = envs.n_agents

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir,
    }

    runner = Runner(config)
    runner.run()

    envs.close()
    if eval_envs is not None and eval_envs is not envs:
        eval_envs.close()

    runner.writter.export_scalars_to_json(str(runner.log_dir + "/summary.json"))
    runner.writter.close()


if __name__ == "__main__":
    main(sys.argv[1:]) 