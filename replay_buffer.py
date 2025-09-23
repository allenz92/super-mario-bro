from typing import Deque, Tuple
from collections import deque
import random
import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity: int, obs_shape: Tuple[int, int, int], device: torch.device):
        self.capacity = capacity
        self.device = device
        self.obs_shape = obs_shape
        self.buffer: Deque = deque(maxlen=capacity)

    def push(self, obs: np.ndarray, action: int, reward: float, next_obs: np.ndarray, done: bool):
        self.buffer.append((obs, action, reward, next_obs, done))

    def __len__(self):
        return len(self.buffer)

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        obs, actions, rewards, next_obs, dones = zip(*batch)
        # 返回位于 CPU 的张量，并使用 pinned memory，以便后续 .to(device, non_blocking=True)
        obs = torch.from_numpy(np.stack(obs)).float().div(255.0).pin_memory()
        next_obs = torch.from_numpy(np.stack(next_obs)).float().div(255.0).pin_memory()
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)
        return obs, actions, rewards, next_obs, dones
