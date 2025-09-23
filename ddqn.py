from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNet(nn.Module):
    def __init__(self, in_channels: int, num_actions: int):
        super().__init__()
        # 按论文表1: Conv(32, 8x8, stride 4) -> Conv(64, 4x4, stride 2) -> Conv(64, 3x3, stride 1)
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, num_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x / 1.0  # 输入已在 buffer 归一化到[0,1]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


@dataclass
class DDQNConfig:
    num_actions: int
    in_channels: int = 4
    gamma: float = 0.99
    lr: float = 1e-4
    target_sync_interval: int = 1000


class DDQNAgent:
    def __init__(self, cfg: DDQNConfig, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.online = ConvNet(cfg.in_channels, cfg.num_actions).to(device)
        self.target = ConvNet(cfg.in_channels, cfg.num_actions).to(device)
        # 多 GPU: 若可用且设备为 CUDA，则使用 DataParallel
        if device.type == 'cuda' and torch.cuda.device_count() > 1:
            self.online = nn.DataParallel(self.online)
            self.target = nn.DataParallel(self.target)
        self.target.load_state_dict(self.online.state_dict())
        # 注意：优化器需在（可能的）并行包装之后创建
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=cfg.lr)
        self.learn_steps = 0

    @torch.no_grad()
    def act(self, obs: torch.Tensor, epsilon: float) -> int:
        if torch.rand(1).item() < epsilon:
            return int(torch.randint(0, self.cfg.num_actions, (1,)).item())
        # 使用主卡模型进行推理，避免 DataParallel 在 batch=1 时的额外开销
        model = self._unwrap(self.online)
        q_values = model(obs.unsqueeze(0))
        return int(q_values.argmax(dim=1).item())

    def learn(self, batch, gamma: float = None):
        obs, actions, rewards, next_obs, dones = batch
        # 非阻塞拷贝到目标设备（配合 pinned memory）
        obs = obs.to(self.device, non_blocking=True)
        next_obs = next_obs.to(self.device, non_blocking=True)
        actions = actions.to(self.device, non_blocking=True)
        rewards = rewards.to(self.device, non_blocking=True)
        dones = dones.to(self.device, non_blocking=True)
        q_values = self.online(obs).gather(1, actions.view(-1, 1)).squeeze(1)
        with torch.no_grad():
            next_actions = self.online(next_obs).argmax(dim=1, keepdim=True)
            next_q = self.target(next_obs).gather(1, next_actions).squeeze(1)
            target_q = rewards + (1.0 - dones) * (self.cfg.gamma if gamma is None else gamma) * next_q
        loss = F.smooth_l1_loss(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), max_norm=10.0)
        self.optimizer.step()
        self.learn_steps += 1
        if self.learn_steps % self.cfg.target_sync_interval == 0:
            self.target.load_state_dict(self.online.state_dict())
        return loss.item()

    # --- 辅助：保存/加载在 DataParallel 下的权重 ---
    def _unwrap(self, model: nn.Module) -> nn.Module:
        return model.module if isinstance(model, nn.DataParallel) else model

    def state_dicts(self):
        return {
            'online': self._unwrap(self.online).state_dict(),
            'target': self._unwrap(self.target).state_dict(),
        }

    def load_state_dicts(self, state):
        self._unwrap(self.online).load_state_dict(state['online'])
        self._unwrap(self.target).load_state_dict(state['target'])
