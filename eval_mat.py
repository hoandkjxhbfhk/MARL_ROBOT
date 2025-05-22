import sys, os
# Thêm project root và MAT code vào đường dẫn
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'Multi-Agent-Transformer')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'Multi-Agent-Transformer/mat')))

from mappo import generate_vector_features, convert_observation
from env import Environment
from mappo_agent import Agents
from greedyagent import GreedyAgents
#from randomagent import RandomAgents
import numpy as np
import argparse
import torch
from pathlib import Path

from mat.config import get_config as get_mat_config
from mat.envs.env_mat import DeliveryMATEnv
from mat.algorithms.mat.algorithm.transformer_policy import TransformerPolicy as Policy # Kiểm tra lại đường dẫn này

def _t2n(x):
    """Chuyển tensor sang numpy array."""
    return x.detach().cpu().numpy()

class MATAgent:
    def __init__(self, agent_params):
        self.args = agent_params['args']
        self.device = agent_params['device']

        # MAT policy expects args.n_agents, not num_agents from the main eval script
        # Ensure self.args.n_agents is set correctly before this point (done in __main__)
        
        # Tạo observation space và action space giả lập cho policy của MAT
        # DeliveryMATEnv sẽ cung cấp observation_space và action_space thực tế khi được khởi tạo
        # Tuy nhiên, Policy.__init__ cần chúng. Chúng ta có thể tạo chúng dựa trên args.
        # Hoặc, nếu DeliveryMATEnv có thể được khởi tạo nhẹ nhàng để lấy space, thì tốt hơn.
        # Tạm thời, chúng ta sẽ tạo placeholder dựa trên hiểu biết về DeliveryMATEnv.
        # Kích thước obs_space và act_space của DeliveryMATEnv phụ thuộc vào map, n_agents, v.v.
        # Cần đảm bảo các tham số này (self.args.map_file, self.args.n_agents, ...) đã được set.
        
        # Khởi tạo một env tạm thời để lấy spaces
        env_args_for_spaces = {
            "map_file": self.args.map_file,
            "n_robots": self.args.n_agents,
            "n_packages": self.args.n_packages,
            "max_time_steps": self.args.episode_length, # MAT dùng episode_length
            "move_cost": self.args.move_cost,
            "delivery_reward": self.args.delivery_reward,
            "delay_reward": self.args.delay_reward,
            "seed": self.args.seed, # Seed này chỉ để khởi tạo, không ảnh hưởng đến eval episodes
        }
        temp_env_for_spaces = DeliveryMATEnv(env_args=env_args_for_spaces)
        observation_space = temp_env_for_spaces.observation_space
        action_space = temp_env_for_spaces.action_space
        # centroid_obs_space = temp_env_for_spaces.centroid_obs_space # Nếu MAT model dùng (cho centralized critic)
        # share_observation_space = temp_env_for_spaces.share_observation_space # Nếu MAT model dùng
        del temp_env_for_spaces


        # Load policy
        # model_dir phải là đường dẫn tới thư mục models (vd: run1/models)
        # Policy sẽ tự tìm file actor.pt hoặc policy.pt trong đó
        self.policy = Policy(self.args,
                             obs_space=observation_space[0], # Lấy space của một agent
                             # cent_obs_space=share_observation_space[0] if self.args.use_centralized_V else None,
                             cent_obs_space=observation_space[0], # Tạm thời dùng obs_space cho cent_obs_space nếu use_centralized_V=False, cần kiểm tra lại
                             act_space=action_space[0], # Lấy space của một agent
                             device=self.device)
        
        if self.args.model_dir is not None:
            print(f"Đang tải model MAT từ: {self.args.model_dir}")
            self.policy.load_models(self.args.model_dir)
        else:
            print("Cảnh báo: không có --mat_model_dir được cung cấp, MAT agent sẽ dùng policy khởi tạo ngẫu nhiên.")

        self.rnn_states = np.zeros((1, self.args.n_agents, self.args.recurrent_N, self.args.hidden_size), dtype=np.float32)
        if self.args.use_centralized_V: # Chỉ khởi tạo rnn_states_critic nếu cần
            # Kích thước của rnn_states_critic có thể khác, phụ thuộc vào cấu hình policy
            # Thông thường là (n_rollout_threads, num_agents, recurrent_N, hidden_size) hoặc (n_rollout_threads, num_agents, hidden_size)
            # Với eval, n_rollout_threads = 1
             self.rnn_states_critic = np.zeros((1, self.args.n_agents, self.args.recurrent_N, self.args.hidden_size), dtype=np.float32)
        else:
            self.rnn_states_critic = self.rnn_states # Nếu không dùng V tập trung, critic rnn state có thể giống actor

    def init_agents(self, initial_obs=None): # initial_obs có thể không cần cho MAT
        self.rnn_states = np.zeros((1, self.args.n_agents, self.args.recurrent_N, self.args.hidden_size), dtype=np.float32)
        if self.args.use_centralized_V:
            self.rnn_states_critic = np.zeros((1, self.args.n_agents, self.args.recurrent_N, self.args.hidden_size), dtype=np.float32)
        else:
             self.rnn_states_critic = self.rnn_states


    @torch.no_grad()
    def get_actions(self, obs, share_obs, available_actions, deterministic=True):
        # obs, share_obs, available_actions đã có dạng (n_agents, feature_dim) hoặc tương tự
        # Chúng cần được reshape thành (1, n_agents, feature_dim) cho policy
        
        obs_reshaped = np.expand_dims(obs, axis=0)
        share_obs_reshaped = np.expand_dims(share_obs, axis=0)
        available_actions_reshaped = np.expand_dims(available_actions, axis=0)

        # Masks for recurrent layers (1 = not done, 0 = done)
        # Trong eval, mask luôn là 1 trừ khi episode kết thúc, nhưng get_actions được gọi mỗi step
        masks = np.ones((1, self.args.n_agents, 1), dtype=np.float32)

        actions, rnn_states_actor, rnn_states_critic_ = self.policy.get_actions(
            share_obs=share_obs_reshaped,
            obs=obs_reshaped,
            rnn_states_actor=self.rnn_states,
            rnn_states_critic=self.rnn_states_critic,
            masks=masks,
            available_actions=available_actions_reshaped,
            deterministic=deterministic
        )
        
        self.rnn_states = _t2n(rnn_states_actor)
        self.rnn_states_critic = _t2n(rnn_states_critic_)
        
        # actions là tensor, cần chuyển sang numpy và squeeze batch_dim (là 1)
        return _t2n(actions).squeeze(0)


def run_eval(agent_type, agent_params, num_episodes, env_config):
    rewards, delivered, delivery_rate = [], [], []
    base_seed = env_config['seed']

    # Initialize agent once before the episode loop
    if agent_type == "mappo":
        agent = Agents(
            agent_params['observation_shape'],
            agent_params['vector_obs_dim'],
            env_config['max_time_steps'],
            agent_params['model_path'],
            agent_params['device']
        )
    elif agent_type == "greedy":
        agent = GreedyAgents()
    elif agent_type == "mat":
        agent = MATAgent(agent_params)
    # elif agent_type == "random":
    #     agent = RandomAgents() 
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

    for ep in range(num_episodes):
        current_seed = base_seed + ep
        if agent_type == "mat":
            # MAT agent sử dụng DeliveryMATEnv
            # Các tham số env cho DeliveryMATEnv lấy từ agent_params['args'] đã được chuẩn bị
            mat_args = agent_params['args']
            env_args_for_mat = {
                "map_file": mat_args.map_file,
                "n_robots": mat_args.n_agents,
                "n_packages": mat_args.n_packages,
                "max_time_steps": mat_args.episode_length,
                "move_cost": mat_args.move_cost,
                "delivery_reward": mat_args.delivery_reward,
                "delay_reward": mat_args.delay_reward,
                "seed": current_seed, # Sử dụng seed riêng cho mỗi episode
            }
            env = DeliveryMATEnv(env_args=env_args_for_mat)
            # Reset env và lấy obs, share_obs, available_actions
            # Output của env.reset() là: obs, share_obs, available_actions
            obs, share_obs, available_actions = env.reset()
            agent.init_agents() # Reset RNN states nếu cần
        else:
            # Các agent khác (MAPPO, Greedy) dùng Environment hiện tại
            env = Environment(
                map_file=env_config['map'],
                max_time_steps=env_config['max_time_steps'],
                n_robots=env_config['n_agents'],
                n_packages=env_config['n_packages'],
                seed=current_seed
            )
            state = env.reset() # state của Environment hiện tại
            agent.init_agents(state)

        done = False
        infos = {}
        current_episode_reward = 0 # Theo dõi reward cho MAT agent

        while not done:
            if agent_type == "mappo":
                actions = agent.get_actions(state, deterministic=False) 
            elif agent_type == "greedy": # Greedy không có deterministic param
                actions = agent.get_actions(state)
            elif agent_type == "mat":
                # MAT agent cần obs, share_obs, available_actions
                actions = agent.get_actions(obs, share_obs, available_actions, deterministic=True) # MAT thường dùng deterministic=True cho eval
            else: # Random hoặc các agent khác
                actions = agent.get_actions(state)
            
            if agent_type == "mat":
                # env.step() của DeliveryMATEnv trả về:
                # obs, share_obs, rewards, dones, infos, available_actions
                next_obs, next_share_obs, step_rewards, step_dones, step_infos, next_available_actions = env.step(actions)
                obs, share_obs, available_actions = next_obs, next_share_obs, next_available_actions
                
                # rewards là (n_agents, 1), dones là (n_agents, 1)
                # Ta cần tổng reward của tất cả agents tại step này
                current_episode_reward += np.sum(step_rewards) 
                infos = step_infos # infos từ DeliveryMATEnv có thể chứa 'num_delivered_packages'
                # done trong MAT là khi all(dones) hoặc max_steps
                # DeliveryMATEnv trả về dones dưới dạng mảng (n_agents, 1)
                # done_env = np.all(step_dones) # Nếu muốn kiểm tra từng agent
                if "bad_transition" in infos and infos["bad_transition"]: # Đây là cách một số env của MAT báo hiệu kết thúc episode
                    done = True
                elif env.base_env.t >= env.max_time_steps: # Kiểm tra thủ công max_time_steps
                     done = True
                # Hoặc nếu tất cả agents đều done (tùy theo logic của DeliveryMATEnv)
                # if np.all(step_dones):
                #    done = True

            else: # MAPPO, Greedy
                state, reward, current_done, current_infos = env.step(actions)
                infos = current_infos
                if current_done:
                    done = True
        
        # Lấy final metrics
        if agent_type == "mat":
            final_reward = current_episode_reward
            # DeliveryMATEnv thường lưu số gói hàng đã giao trong infos hoặc trực tiếp trong env
            # Ví dụ, nếu nó được lưu trong `env.num_delivered_packages_total`
            n_delivered = env.get_total_delivered_packages() # Always use the method for MAT agent
            
        else: # MAPPO, Greedy
            final_reward = infos.get('total_reward', env.total_reward)
            n_delivered = sum(1 for p in env.packages if p.status == 'delivered')

        rewards.append(final_reward)
        delivered.append(n_delivered)
        current_delivery_rate = 0
        
        n_total_packages_for_rate = env_config['n_packages']
        if agent_type == "mat": # MAT env có thể có số package khác được định nghĩa trong mat_args
             n_total_packages_for_rate = agent_params['args'].n_packages

        if n_total_packages_for_rate > 0:
            current_delivery_rate = (n_delivered / n_total_packages_for_rate) * 100
        delivery_rate.append(current_delivery_rate)

    return {
        "rewards": rewards,
        "delivered": delivered,
        "delivery_rate": delivery_rate,
        "mean_reward": np.mean(rewards) if rewards else 0,
        "std_reward": np.std(rewards) if rewards else 0,
        "mean_delivered": np.mean(delivered) if delivered else 0,
        "std_delivered": np.std(delivered) if delivered else 0,
        "mean_delivery_rate": np.mean(delivery_rate) if delivery_rate else 0,
        "std_delivery_rate": np.std(delivery_rate) if delivery_rate else 0,
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_agents", type=int, default=5)
    parser.add_argument("--n_packages", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--max_time_steps", type=int, default=1000)
    parser.add_argument("--map", type=str, default="map3.txt")
    parser.add_argument("--num_test_episodes", type=int, default=1, 
                        help="Số episode để chạy cho mỗi agent. Mỗi episode sẽ dùng cùng seed môi trường.")
    parser.add_argument("--mappo_model_path", type=str, default="models_newcnn_newcnn_map2_map2/mappo_final_actor.pt")
    parser.add_argument("--device", type=str, default="cuda:3")

    # Thêm các tham số cho MAT
    parser.add_argument("--mat_model_dir", type=str, default="Multi-Agent-Transformer/mat/scripts/results/delivery/mat/map3_5agents_50pkgs/check/run2/models/transformer_624.pt", 
                        help="Đường dẫn đến thư mục chứa model MAT đã huấn luyện.")
    parser.add_argument("--mat_algorithm_name", type=str, default="mat", choices=['mat', 'mat_dec'], help="Tên thuật toán MAT.")

    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Device
    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda:3")
    else:
        device = torch.device("cpu")
    args.device = device # Cập nhật lại args.device với torch.device object

    temp_env = Environment(map_file=args.map, max_time_steps=args.max_time_steps,
                           n_robots=args.num_agents, n_packages=args.n_packages,
                           seed=args.seed)
    temp_state = temp_env.reset()
    obs_shape = (6, temp_env.n_rows, temp_env.n_cols)
    vec_obs_dim = 0
    try:
        # Ensure num_agents is at least 1 for max_other_robots_to_observe
        max_other_robots = max(0, args.num_agents - 1)
        max_pkgs_to_obs = 5 # Consistent with mappo.py usage
        vec_obs_dim = generate_vector_features(temp_state, {}, 0, args.max_time_steps, max_other_robots, max_pkgs_to_obs).shape[0]
    except Exception as e:
        print(f"Lỗi khi tạo vector_obs_dim: {e}. Đặt tạm vec_obs_dim = 0.")
    del temp_env

    env_config = {
        'max_time_steps': args.max_time_steps,
        'n_packages': args.n_packages,
        'n_agents': args.num_agents,
        'seed': args.seed,
        'map': args.map,
    }
    mappo_params = {
        'observation_shape': obs_shape,
        'vector_obs_dim': vec_obs_dim,
        'model_path': args.mappo_model_path,
        'device': args.device
    }

    # Chuẩn bị tham số cho MAT Agent
    # Lấy config mặc định của MAT và override nếu cần
    mat_parser = get_mat_config()
    mat_default_args = mat_parser.parse_args([]) # Parse với list rỗng để lấy default

    # Override các giá trị từ args của eval.py nếu chúng có ý nghĩa tương đương
    mat_default_args.n_agents = args.num_agents # Quan trọng: MAT gọi là n_agents
    mat_default_args.episode_length = args.max_time_steps # MAT gọi là episode_length
    mat_default_args.seed = args.seed
    mat_default_args.cuda = args.device.type == 'cuda'
    mat_default_args.model_dir = args.mat_model_dir # Đường dẫn đến model đã train
    mat_default_args.algorithm_name = args.mat_algorithm_name
    
    # Các tham số đặc thù cho DeliveryMATEnv (giống train_delivery.py)
    mat_default_args.map_file = args.map
    mat_default_args.n_packages = args.n_packages
    # Các giá trị mặc định khác từ train_delivery.py nếu không có trong get_mat_config
    # Ví dụ: move_cost, delivery_reward, delay_reward. Nếu MATPolicy/DeliveryMATEnv cần thì phải thêm vào đây
    # Hiện tại, DeliveryMATEnv trong MAT repo có vẻ lấy các giá trị này từ all_args.xxx
    # nên chúng ta cần đảm bảo chúng được set trong mat_default_args
    if not hasattr(mat_default_args, 'move_cost'): mat_default_args.move_cost = -0.01
    if not hasattr(mat_default_args, 'delivery_reward'): mat_default_args.delivery_reward = 10.0
    if not hasattr(mat_default_args, 'delay_reward'): mat_default_args.delay_reward = 1.0


    mat_params = {
        'args': mat_default_args, # Truyền toàn bộ args của MAT
        'device': args.device
    }
    # Kết thúc chuẩn bị tham số cho MAT Agent

    print(f"Testing each agent on {args.num_test_episodes} episodes.")
    print(f"Environment seed for each episode: {args.seed}")
    print(f"MAPPO agent will use deterministic=False for actions.")
    print(f"Greedy and Random agents will use deterministic=True for actions.")

    # print("\nEvaluating MAPPO agent...")
    # mappo_metrics = run_eval("mappo", mappo_params, args.num_test_episodes, env_config)
    # print("Evaluating Greedy agent...")
    # greedy_metrics = run_eval("greedy", mappo_params, args.num_test_episodes, env_config)
    print("Evaluating MAT agent...")
    mat_metrics = run_eval("mat", mat_params, args.num_test_episodes, env_config)

    print("\n=== Evaluation Results ===")
    # print(f"MAPPO: mean_reward={mappo_metrics['mean_reward']:.2f}±{mappo_metrics['std_reward']:.2f}, "
    #       f"mean_delivered={mappo_metrics['mean_delivered']:.2f}±{mappo_metrics['std_delivered']:.2f}, "
    #       f"mean_delivery_rate={mappo_metrics['mean_delivery_rate']:.2f}%±{mappo_metrics['std_delivery_rate']:.2f}%")
    # print(f"Greedy: mean_reward={greedy_metrics['mean_reward']:.2f}±{greedy_metrics['std_reward']:.2f}, "
    #       f"mean_delivered={greedy_metrics['mean_delivered']:.2f}±{greedy_metrics['std_delivered']:.2f}, "
    #       f"mean_delivery_rate={greedy_metrics['mean_delivery_rate']:.2f}%±{greedy_metrics['std_delivery_rate']:.2f}%")
    print(f"MAT: mean_reward={mat_metrics['mean_reward']:.2f}±{mat_metrics['std_reward']:.2f}, "
          f"mean_delivered={mat_metrics['mean_delivered']:.2f}±{mat_metrics['std_delivered']:.2f}, "
          f"mean_delivery_rate={mat_metrics['mean_delivery_rate']:.2f}%±{mat_metrics['std_delivery_rate']:.2f}%")

    # # Plot nhiều metric hơn
    # try:
    #     import matplotlib.pyplot as plt
    #     labels = ['Reward', 'Delivered', 'Delivery Rate (%)']
        
    #     mappo_means = [mappo_metrics['mean_reward'], mappo_metrics['mean_delivered'], mappo_metrics['mean_delivery_rate']]
    #     mappo_stds = [mappo_metrics['std_reward'], mappo_metrics['std_delivered'], mappo_metrics['std_delivery_rate']]
        
    #     greedy_means = [greedy_metrics['mean_reward'], greedy_metrics['mean_delivered'], greedy_metrics['mean_delivery_rate']]
    #     greedy_stds = [greedy_metrics['std_reward'], greedy_metrics['std_delivered'], greedy_metrics['std_delivery_rate']]
        
    #     mat_means = [mat_metrics['mean_reward'], mat_metrics['mean_delivered'], mat_metrics['mean_delivery_rate']]
    #     mat_stds = [mat_metrics['std_reward'], mat_metrics['std_delivered'], mat_metrics['std_delivery_rate']]
        
    #     x = np.arange(len(labels))
    #     width = 0.25 
        
    #     fig, ax = plt.subplots(figsize=(10, 6))
    #     rects1 = ax.bar(x - width, mappo_means, width, yerr=mappo_stds, label='MAPPO', color='skyblue', capsize=5)
    #     rects2 = ax.bar(x, greedy_means, width, yerr=greedy_stds, label='Greedy', color='salmon', capsize=5)
    #     rects_mat = ax.bar(x + width, mat_means, width, yerr=mat_stds, label='MAT', color='lightgreen', capsize=5)
        
    #     ax.set_xticks(x)
    #     ax.set_xticklabels(labels)
    #     ax.set_ylabel('Score')
    #     ax.set_title(
    #         f'MAPPO vs Greedy vs MAT Agent Evaluation ({args.num_test_episodes} episodes each)\n'
    #         f'Map: {args.map} | Agents: {args.num_agents} | Packages: {args.n_packages} | Seed: {args.seed}',
    #         fontsize=12
    #     )
    #     ax.legend()
        
    #     # Thêm số và sai số lên trên cột
    #     def autolabel(rects, stds):
    #         for rect, std in zip(rects, stds):
    #             height = rect.get_height()
    #             ax.annotate(f'{height:.2f}\n±{std:.2f}',
    #                         xy=(rect.get_x() + rect.get_width() / 2, height),
    #                         xytext=(0, 3),  # 3 points vertical offset
    #                         textcoords="offset points",
    #                         ha='center', va='bottom', fontsize=9, fontweight='bold')

    #     autolabel(rects1, mappo_stds)
    #     autolabel(rects2, greedy_stds)
    #     autolabel(rects_mat, mat_stds)
        
    #     plt.tight_layout()
    #     plt.show()
    #     plt.savefig(f"plots/eval_results_{args.map}_{args.num_agents}_{args.n_packages}_{args.seed}.png")
    # except ImportError:
    #     print("matplotlib not installed, skipping plot.")
    # except Exception as e:
    #     print(f"Error during plotting: {e}")