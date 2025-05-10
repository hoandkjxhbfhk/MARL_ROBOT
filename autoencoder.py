import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pickle
import os
from tqdm import tqdm

# ---- State Representation ----
def state_to_vector(state, map_size=(10, 10), max_pkgs=10):
    """Chuyển trạng thái môi trường thành feature vector cố định cho Autoencoder."""
    # Thông tin robot: hỗ trợ cả tuple và object
    robot = state['robots'][0]
    if isinstance(robot, tuple):
        r, c, carrying = robot
        # env.get_state trả về vị trí 1-indexed
        r0 = r - 1
        c0 = c - 1
    else:
        r0, c0 = robot.position
    carrying = robot.carrying

    # Khởi tạo vector đặc trưng
    vec = np.zeros(4 + 7*max_pkgs, dtype=np.float32)
    
    # Đặc trưng robot: (r, c, carrying_flag, carrying_id)
    vec[0] = r0 / map_size[0]
    vec[1] = c0 / map_size[1]
    vec[2] = 1.0 if carrying > 0 else 0.0
    vec[3] = carrying / max_pkgs if carrying > 0 else 0.0
    
    # Đặc trưng cho mỗi package (tối đa max_pkgs gói), hỗ trợ tuple và object
    for i, pkg in enumerate(state['packages']):
        if i >= max_pkgs:
            break
        
        base = 4 + i*7
        
        # Xử lý theo dạng tuple hoặc object
        if isinstance(pkg, tuple):
            # tuple: (id, start_r, start_c, target_r, target_c, start_time, deadline)
            vec[base] = 1.0  # waiting
            vec[base+1] = 0.0
            vec[base+2] = 0.0
            _, sr, sc, tr, tc, *rest = pkg
            vec[base+3] = (sr - 1) / map_size[0]
            vec[base+4] = (sc - 1) / map_size[1]
            vec[base+5] = (tr - 1) / map_size[0]
            vec[base+6] = (tc - 1) / map_size[1]
        else:
            # object có thuộc tính status, start, target
        if pkg.status == 'waiting':
            vec[base] = 1.0
        elif pkg.status == 'in_transit':
            vec[base+1] = 1.0
        elif pkg.status == 'delivered':
            vec[base+2] = 1.0
        if hasattr(pkg, 'start') and pkg.start:
            vec[base+3] = pkg.start[0] / map_size[0]
            vec[base+4] = pkg.start[1] / map_size[1]
        if hasattr(pkg, 'target') and pkg.target:
            vec[base+5] = pkg.target[0] / map_size[0]
            vec[base+6] = pkg.target[1] / map_size[1]
            
    return vec

# ---- Dataset ----
class StateDataset(Dataset):
    """Dataset chứa các trạng thái để huấn luyện Autoencoder."""
    def __init__(self, states, map_size=(10,10), max_pkgs=10):
        self.states = states
        self.map_size = map_size
        self.max_pkgs = max_pkgs
        
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        state = self.states[idx]
        vec = state_to_vector(state, self.map_size, self.max_pkgs)
        return torch.tensor(vec, dtype=torch.float32)

# ---- Simple Autoencoder ----
class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=8):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        # Decoder 
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Sigmoid() # Để output nằm trong [0,1] - cân nhắc nếu có đặc trưng cần khác
        )
        
    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)

# ---- Variational Autoencoder ----
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=8):
        super().__init__()
        # Encoder
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc31 = nn.Linear(64, latent_dim)  # mu
        self.fc32 = nn.Linear(64, latent_dim)  # logvar
        
        # Decoder
        self.fc4 = nn.Linear(latent_dim, 64)
        self.fc5 = nn.Linear(64, 128)
        self.fc6 = nn.Linear(128, input_dim)
        
    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        mu = self.fc31(h2)
        logvar = self.fc32(h2)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h4 = F.relu(self.fc4(z))
        h5 = F.relu(self.fc5(h4))
        return torch.sigmoid(self.fc6(h5))
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

# ---- Loss Functions ----
def vae_loss(recon_x, x, mu, logvar):
    """VAE loss = Reconstruction loss + KL divergence"""
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# ---- Training ----
def train_autoencoder(states, input_dim, latent_dim=8, epochs=100, batch_size=64, 
                      map_size=(10,10), max_pkgs=10, save_path="autoencoder.pth", vae=False):
    """Huấn luyện Autoencoder hoặc VAE với tập states."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Tạo dataset và dataloader
    dataset = StateDataset(states, map_size, max_pkgs)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Khởi tạo model
    if vae:
        model = VAE(input_dim, latent_dim).to(device)
    else:
        model = Autoencoder(input_dim, latent_dim).to(device)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_idx, data in enumerate(dataloader):
            data = data.to(device)
            optimizer.zero_grad()
            
            if vae:
                recon_batch, mu, logvar = model(data)
                loss = vae_loss(recon_batch, data, mu, logvar)
            else:
                recon_batch = model(data)
                loss = F.mse_loss(recon_batch, data)
                
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            
        print(f'Epoch: {epoch}, Avg loss: {train_loss / len(dataloader.dataset):.6f}')
    
    # Lưu model
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
    return model

# ---- Quantization và State Compression ----
class QuantizedEncoder:
    """Encoder kết hợp với lượng tử hóa để tạo state representation gọn."""
    def __init__(self, model, latent_dim=8, n_bins=10, device=None):
        self.model = model
        self.latent_dim = latent_dim
        self.n_bins = n_bins
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        
    def encode_state(self, state, map_size=(10,10), max_pkgs=10):
        """Mã hóa state thành vector latent."""
        vec = state_to_vector(state, map_size, max_pkgs)
        x = torch.tensor(vec, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if isinstance(self.model, VAE):
                mu, _ = self.model.encode(x)
                z = mu.squeeze(0).cpu().numpy()
            else:
                z = self.model.encode(x).squeeze(0).cpu().numpy()
        return z
    
    def quantize(self, z):
        """Lượng tử hóa vector z thành tuple discrete."""
        quantized = []
        for dim in range(self.latent_dim):
            # Scale to [0, n_bins-1]
            val = min(max(0, int((z[dim] + 1) * self.n_bins / 2)), self.n_bins - 1)
            quantized.append(val)
        return tuple(quantized)
    
    def encode_quantized(self, state, map_size=(10,10), max_pkgs=10):
        """Mã hóa state thành latent quantized tuple."""
        z = self.encode_state(state, map_size, max_pkgs)
        return self.quantize(z)

# ---- Collect States for Training ----
def collect_states(env, n_episodes=100, steps_per_episode=100):
    """Thu thập state bằng cách chạy môi trường với random actions."""
    states = []
    for _ in tqdm(range(n_episodes)):
        env.reset()
        state = env.get_full_state()
        states.append(state)
        for _ in range(steps_per_episode):
            # Random action
            action = np.random.choice(['S', 'L', 'R', 'U', 'D']), np.random.choice(['0', '1', '2'])
            _, _, done, _ = env.step([action])
            state = env.get_full_state()
            states.append(state)
            if done:
                break
    return states

# ---- Main Training Script ----
if __name__ == "__main__":
    import argparse
    from env import Environment
    
    parser = argparse.ArgumentParser(description="Train Autoencoder for State Compression")
    parser.add_argument("--map", type=str, default="map1.txt", help="Map file")
    parser.add_argument("--n_episodes", type=int, default=100, help="Episodes to collect data")
    parser.add_argument("--steps", type=int, default=100, help="Steps per episode for collection")
    parser.add_argument("--latent_dim", type=int, default=8, help="Latent dimension for AE/VAE")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--use_vae", action="store_true", help="Use VAE instead of AE")
    parser.add_argument("--max_pkgs", type=int, default=10, help="Max packages to encode")
    parser.add_argument("--save_path", type=str, default=None, help="Path to save model")
    
    args = parser.parse_args()
    
    # Set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create environment
    env = Environment(map_file=args.map, max_time_steps=args.steps, n_robots=1, seed=args.seed)
    map_rows, map_cols = env.n_rows, env.n_cols
    map_size = (map_rows, map_cols)
    
    # Collect states
    print(f"Collecting states from {args.n_episodes} episodes...")
    states = collect_states(env, args.n_episodes, args.steps)
    print(f"Collected {len(states)} states")
    
    # Compute input_dim based on max_pkgs
    input_dim = 4 + 7 * args.max_pkgs
    
    # Choose save path
    if args.save_path is None:
        args.save_path = "vae.pth" if args.use_vae else "autoencoder.pth"
    
    # Train autoencoder
    print(f"Training {'VAE' if args.use_vae else 'Autoencoder'} with latent dim {args.latent_dim}...")
    train_autoencoder(
        states, input_dim, args.latent_dim, args.epochs, args.batch_size,
        map_size, args.max_pkgs, args.save_path, args.use_vae
    )
    
    print("Done! The model can now be used with solve_vi.py to compress states.") 