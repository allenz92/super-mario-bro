import os
import time
import math
import argparse
import random
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from mario_wrappers import make_mario_env
from replay_buffer import ReplayBuffer
from ddqn import DDQNAgent, DDQNConfig


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def linear_decay(start: float, end: float, duration: int, step: int):
    if step >= duration:
        return end
    return start + (end - start) * (step / duration)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-id', type=str, default='SuperMarioBros-1-1-v0')
    parser.add_argument('--total-steps', type=int, default=2_000_000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--buffer-size', type=int, default=100_000)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--start-learning', type=int, default=20_000)
    parser.add_argument('--target-sync', type=int, default=1_000)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--eps-start', type=float, default=1.0)
    parser.add_argument('--eps-end', type=float, default=0.1)
    parser.add_argument('--eps-decay-steps', type=int, default=1_000_000)
    parser.add_argument('--save-dir', type=str, default='runs/mario_ddqn')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=args.save_dir)

    set_seed(args.seed)
    device = torch.device(args.device)

    env = make_mario_env(args.env_id)
    obs = env.reset()
    in_channels = obs.shape[0]
    num_actions = env.action_space.n

    agent = DDQNAgent(DDQNConfig(num_actions=num_actions, in_channels=in_channels, gamma=args.gamma, lr=args.lr, target_sync_interval=args.target_sync), device)
    buffer = ReplayBuffer(args.buffer_size, obs_shape=obs.shape, device=device)

    episode_return = 0.0
    episode_length = 0
    episode_idx = 0

    for global_step in range(1, args.total_steps + 1):
        epsilon = linear_decay(args.eps_start, args.eps_end, args.eps_decay_steps, global_step)
        obs_tensor = torch.from_numpy(obs).float().div(255.0).to(device)
        action = agent.act(obs_tensor, epsilon)
        next_obs, reward, done, info = env.step(action)

        buffer.push(obs, action, reward, next_obs, done)
        obs = next_obs
        episode_return += reward
        episode_length += 1

        # learn
        if global_step > args.start_learning and len(buffer) >= args.batch_size:
            batch = buffer.sample(args.batch_size)
            loss = agent.learn(batch)
            if global_step % 100 == 0:
                writer.add_scalar('loss/td', loss, global_step)

        # logging & reset
        if done:
            writer.add_scalar('charts/episode_return', episode_return, global_step)
            writer.add_scalar('charts/episode_length', episode_length, global_step)
            writer.add_scalar('charts/epsilon', epsilon, global_step)
            obs = env.reset()
            episode_return = 0.0
            episode_length = 0
            episode_idx += 1

        if global_step % 10_000 == 0:
            save_path = os.path.join(args.save_dir, f'model_{global_step}.pt')
            torch.save({'online': agent.online.state_dict(), 'target': agent.target.state_dict(), 'step': global_step}, save_path)

    # final save
    save_path = os.path.join(args.save_dir, f'model_final.pt')
    torch.save({'online': agent.online.state_dict(), 'target': agent.target.state_dict(), 'step': args.total_steps}, save_path)
    env.close()
    writer.close()


if __name__ == '__main__':
    main()
