import os
import time
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
import argparse
import random
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from mario_wrappers import make_mario_env
from replay_buffer import ReplayBuffer
from ddqn import DDQNAgent, DDQNConfig


def record_eval_video(env_id: str, frame_skip: int, stack: int, agent: DDQNAgent, device: torch.device, save_path: str, max_steps: int = 10000, epsilon: float = 0.0, fps: int = 30, prefer_fourcc: str = 'avc1'):
    def to_bgr(img):
        if img is None:
            return None
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        if img.ndim == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if img.shape[2] == 3:
            return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if img.shape[2] == 4:
            return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        return img[:, :, :3]

    def make_writer(path, w, h, fps_val, preferred_list):
        base, ext = os.path.splitext(path)
        for code in preferred_list:
            target_path = path
            if code == 'MJPG' and ext.lower() != '.avi':
                target_path = base + '.avi'
            fourcc = cv2.VideoWriter_fourcc(*code)
            writer_local = cv2.VideoWriter(target_path, fourcc, fps_val, (w, h))
            if writer_local.isOpened():
                return writer_local, target_path, code
            writer_local.release()
        return None, None, None

    env = make_mario_env(env_id, frame_skip=frame_skip, stack=stack)
    obs = env.reset()
    first = env.render('rgb_array')
    if first is None:
        print("[Warn] render('rgb_array') 返回 None，跳过视频保存")
        env.close()
        return
    bgr = to_bgr(first)
    h, w = bgr.shape[:2]
    if (w % 2) != 0 or (h % 2) != 0:
        w = w - (w % 2)
        h = h - (h % 2)
        bgr = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)

    writer, final_path, used_codec = make_writer(
        save_path,
        w,
        h,
        fps,
        [prefer_fourcc, 'avc1', 'mp4v', 'H264', 'MJPG']
    )
    if writer is None:
        print("[Warn] 无法打开 VideoWriter，跳过视频保存")
        env.close()
        return

    steps = 0
    done = False
    while not done and steps < max_steps:
        frame = env.render('rgb_array')
        if frame is not None:
            bgr = to_bgr(frame)
            if bgr.shape[1] != w or bgr.shape[0] != h:
                bgr = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)
            writer.write(bgr)

        obs_tensor = torch.from_numpy(obs).float().div(255.0).to(device)
        action = agent.act(obs_tensor, epsilon)
        obs, _, done, _ = env.step(action)
        steps += 1

    writer.release()
    env.close()
    print(f"[Video] 保存完成: {final_path} (codec={used_codec}, {w}x{h}@{fps})")


def select_device(preferred: str = None) -> torch.device:
    if preferred:
        return torch.device(preferred)
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def linear_decay(start: float, end: float, duration: int, step: int):
    if step >= duration:
        return end
    return start + (end - start) * (step / duration)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-id', type=str, default='SuperMarioBros-1-1-v0')
    parser.add_argument('--total-steps', type=int, default=1_000_000)
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
    parser.add_argument('--save-dir', type=str, default='runs/mario_ddqn_mac')
    parser.add_argument('--log-interval', type=int, default=1000)
    parser.add_argument('--updates-per-step', type=int, default=1)
    parser.add_argument('--frame-skip', type=int, default=4)
    parser.add_argument('--stack-frames', type=int, default=4)
    parser.add_argument('--device', type=str, default=None, help='优先设备：mps/cuda/cpu')
    parser.add_argument('--video-every-episodes', type=int, default=1000, help='每多少回合导出一次评估视频，<=0 关闭')
    parser.add_argument('--video-max-steps', type=int, default=10000, help='评估视频最大步数')
    parser.add_argument('--video-fps', type=int, default=30, help='评估视频帧率')
    parser.add_argument('--plot-ep-group', type=int, default=10, help='绘图聚合粒度：每多少个episode累计一次回报')
    args = parser.parse_args()

    device = select_device(args.device)
    os.makedirs(args.save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=args.save_dir)

    set_seed(args.seed)

    print(f"[Init] Device: {device}, CUDA GPUs: {torch.cuda.device_count() if torch.cuda.is_available() else 0}, MPS: {torch.backends.mps.is_available()}")
    env = make_mario_env(args.env_id, frame_skip=args.frame_skip, stack=args.stack_frames)
    obs = env.reset()
    in_channels = obs.shape[0]
    num_actions = env.action_space.n

    agent = DDQNAgent(DDQNConfig(num_actions=num_actions, in_channels=in_channels, gamma=args.gamma, lr=args.lr, target_sync_interval=args.target_sync), device)
    buffer = ReplayBuffer(args.buffer_size, obs_shape=obs.shape, device=device)

    episode_return = 0.0
    episode_length = 0
    episode_idx = 0
    last_log_time = time.time()
    env_time_accum = 0.0
    learn_time_accum = 0.0

    pbar = tqdm(total=args.total_steps, dynamic_ncols=True)
    episode_returns = []
    episode_lengths = []
    epsilons = []
    loss_steps = []
    loss_values = []
    for global_step in range(1, args.total_steps + 1):
        epsilon = linear_decay(args.eps_start, args.eps_end, args.eps_decay_steps, global_step)
        obs_tensor = torch.from_numpy(obs).float().div(255.0).to(device)
        action = agent.act(obs_tensor, epsilon)
        t0 = time.time()
        next_obs, reward, done, info = env.step(action)
        env_time_accum += time.time() - t0

        buffer.push(obs, action, reward, next_obs, done)
        obs = next_obs
        episode_return += reward
        episode_length += 1

        if global_step > args.start_learning and len(buffer) >= args.batch_size:
            last_loss = None
            for _ in range(args.updates_per_step):
                t1 = time.time()
                batch = buffer.sample(args.batch_size)
                last_loss = agent.learn(batch)
                learn_time_accum += time.time() - t1
            if global_step % 100 == 0 and last_loss is not None:
                writer.add_scalar('loss/td', last_loss, global_step)
                loss_steps.append(global_step)
                loss_values.append(last_loss)

        if done:
            writer.add_scalar('charts/episode_return', episode_return, global_step)
            writer.add_scalar('charts/episode_length', episode_length, global_step)
            writer.add_scalar('charts/epsilon', epsilon, global_step)
            episode_returns.append(episode_return)
            episode_lengths.append(episode_length)
            epsilons.append(epsilon)
            obs = env.reset()
            episode_return = 0.0
            episode_length = 0
            episode_idx += 1
            if args.video_every_episodes > 0 and (episode_idx % args.video_every_episodes == 0):
                try:
                    vid_path = os.path.join(args.save_dir, f"eval_ep{episode_idx}.mp4")
                    print(f"[Video] 录制评估视频: {vid_path}")
                    record_eval_video(
                        env_id=args.env_id,
                        frame_skip=args.frame_skip,
                        stack=args.stack_frames,
                        agent=agent,
                        device=device,
                        save_path=vid_path,
                        max_steps=args.video_max_steps,
                        epsilon=0.0,
                        fps=args.video_fps,
                    )
                except Exception as e:
                    print(f"[Warn] 录制视频失败: {e}")

        if global_step % args.log_interval == 0:
            now = time.time()
            elapsed = max(1e-6, now - last_log_time)
            sps = args.log_interval / elapsed
            total_comp = env_time_accum + learn_time_accum
            env_share = (env_time_accum / total_comp) if total_comp > 0 else 0.0
            learn_share = (learn_time_accum / total_comp) if total_comp > 0 else 0.0
            mem_gb = 0.0
            try:
                if device.type == 'cuda':
                    mem_gb = torch.cuda.memory_allocated() / 1e9
                elif device.type == 'mps':
                    mem_gb = 0.0
            except Exception:
                mem_gb = 0.0

            pbar.update(args.log_interval)
            pbar.set_postfix({
                'eps': f"{epsilon:.3f}",
                'ep': episode_idx,
                'len': episode_length,
                'ret': f"{episode_return:.1f}",
                'sps': f"{sps:.1f}",
                'env%': f"{env_share*100:.0f}",
                'learn%': f"{learn_share*100:.0f}",
                'mem(GB)': f"{mem_gb:.2f}"
            })
            last_log_time = now
            env_time_accum = 0.0
            learn_time_accum = 0.0

        if global_step % 10_000 == 0:
            save_path = os.path.join(args.save_dir, f'model_{global_step}.pt')
            torch.save({**agent.state_dicts(), 'step': global_step}, save_path)

    save_path = os.path.join(args.save_dir, f'model_final.pt')
    torch.save({**agent.state_dicts(), 'step': args.total_steps}, save_path)
    try:
        if len(episode_returns) > 0:
            group = max(1, int(args.plot_ep_group))
            grouped_returns = [sum(episode_returns[i:i+group]) for i in range(0, len(episode_returns), group)]
            plt.figure(figsize=(8, 4))
            plt.plot(grouped_returns)
            plt.title(f'Cumulative Return per {group} Episodes')
            plt.xlabel(f'Block (each {group} episodes)')
            plt.ylabel('Cumulative Return')
            plt.tight_layout()
            plt.savefig(os.path.join(args.save_dir, f'episode_return_per{group}.png'))
            plt.close()
        if len(episode_lengths) > 0:
            plt.figure(figsize=(8, 4))
            plt.plot(episode_lengths)
            plt.title('Episode Length')
            plt.xlabel('Episode')
            plt.ylabel('Length')
            plt.tight_layout()
            plt.savefig(os.path.join(args.save_dir, 'episode_length.png'))
            plt.close()
        if len(epsilons) > 0:
            plt.figure(figsize=(8, 4))
            plt.plot(epsilons)
            plt.title('Epsilon per Episode')
            plt.xlabel('Episode')
            plt.ylabel('Epsilon')
            plt.tight_layout()
            plt.savefig(os.path.join(args.save_dir, 'epsilon.png'))
            plt.close()
        if len(loss_steps) > 0:
            plt.figure(figsize=(8, 4))
            plt.plot(loss_steps, loss_values)
            plt.title('TD Loss')
            plt.xlabel('Global Step')
            plt.ylabel('Loss')
            plt.tight_layout()
            plt.savefig(os.path.join(args.save_dir, 'loss_td.png'))
            plt.close()
    except Exception as e:
        print(f"[Warn] Failed to save plots: {e}")
    env.close()
    writer.close()
    pbar.close()


if __name__ == '__main__':
    main()
