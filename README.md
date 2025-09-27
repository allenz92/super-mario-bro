# Super Mario Bros DDQN (from paper reproduction)

- 算法: DDQN (在线/目标网络 + 经验回放 + ε-greedy)
- 预处理: 帧跳、灰度、缩放到 84x84、堆叠 4 帧、动作空间约简
- 环境: gym-super-mario-bros + nes-py
- 深度学习: PyTorch

## 安装

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

若 macOS Apple Silicon 遇到 torch 安装问题，可参考官方说明或改用 `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`。

## 训练

```bash
python train_ddqn.py \
  --env-id SuperMarioBros-1-1-v0 \
  --total-steps 2000000 \
  --save-dir runs/mario_ddqn
```

### 多 GPU 训练

- 已内置对 PyTorch `DataParallel` 的支持：若检测到多张 CUDA GPU，会自动在 `DDQNAgent` 内并行前向与反向。
- 启动方式与单卡相同；确保可见 GPU 包含两张卡，例如：

```bash
# 例：使用第 0,1 两张卡
CUDA_VISIBLE_DEVICES=0,1 python train_ddqn.py \
  --env-id SuperMarioBros-1-1-v0 \
  --total-steps 2000000 \
  --save-dir runs/mario_ddqn
```

注意：保存时会自动解包 `DataParallel`，因此 checkpoint 可在单卡或多卡环境下互通加载。

### macOS 训练（含 MPS）

```bash
source .venv/bin/activate
python train_ddqn_mac.py \
  --device mps \
  --env-id SuperMarioBros-1-1-v0 \
  --total-steps 1000000 \
  --save-dir runs/mario_ddqn_mac
```

如需关闭视频导出：`--video-every-episodes 0`；按回合导出视频：`--video-every-episodes 1`。

## 主要文件
- `mario_wrappers.py`: 环境包装与图像预处理
- `ddqn.py`: 模型结构与目标网络同步
- `replay_buffer.py`: 经验回放
- `train_ddqn.py`: 训练循环与日志
- `train_ddqn_mac.py`: mac 版训练（自动选择 MPS/CUDA/CPU，训练后绘图、周期性导出视频）

## 参数说明（通用 + mac 版）

通用参数（`train_ddqn.py` 与 `train_ddqn_mac.py` 共享）：

- `--env-id`：环境关卡 ID（如 `SuperMarioBros-1-1-v0`）
- `--total-steps`：训练总步数（环境交互步）
- `--seed`：随机种子
- `--buffer-size`：经验回放容量
- `--batch-size`：一次学习的样本数量
- `--start-learning`：开始学习前需要的最少步数（预热）
- `--target-sync`：目标网络同步间隔（步）
- `--gamma`：折扣因子
- `--lr`：学习率
- `--eps-start` / `--eps-end` / `--eps-decay-steps`：ε-greedy 探索从起始到结束的线性退火配置
- `--save-dir`：日志与模型保存目录（TensorBoard、图表、权重、视频均写入此处）
- `--log-interval`：日志打印与进度条更新的步间隔
- `--updates-per-step`：每个环境步执行的学习次数（>1 可提高设备利用率）
- `--frame-skip`：环境每次动作跳过的帧数（效率/控制的折中）
- `--stack-frames`：堆叠的历史帧数（默认 4）
- `--device`：优先设备（`mps`/`cuda`/`cpu`），缺省时自动选择

mac 版特有（`train_ddqn_mac.py`）：

- `--video-every-episodes`：每多少回合导出一次评估视频（≤0 关闭，默认 1000）
- `--video-max-steps`：评估视频最大步数（防止视频过长）
- `--video-fps`：导出视频帧率（默认 30）

## 进度条字段含义（train_ddqn_mac.py）

- `ep`：已完成回合数（Episode Index）
- `eps`：当前步的探索率 ε（随机动作的概率）
- `len`：当前回合已进行的步数（回合结束归零）
- `ret`：当前回合的累计回报（回合结束归零）
- `sps`：最近一次日志间隔内的 Steps Per Second
- `env%` / `learn%`：环境交互/学习耗时占比（粗略）
- `mem(GB)`：显存占用（CUDA 可用；MPS 无法精确查询时显示 0.00）

## 训练结束图表（train_ddqn_mac.py）

- `episode_return.png`：每个回合的累计回报，反映策略整体表现趋势（越高越好）
- `episode_length.png`：每个回合的步数，间接反映存活时长/通关进度（任务不同含义不同）
- `epsilon.png`：每回合结束时记录的 ε 值（探索率），随训练推进应逐步降低
- `loss_td.png`：随全局步记录的 TD Loss（Huber Loss），用于观测学习稳定性（抖动属正常）

## 视频导出（train_ddqn_mac.py）

- 触发规则：每 `--video-every-episodes` 个回合，在独立评估环境中用当前策略（ε=0）录制 `eval_epXXXX.mp4`
- 写出方式：`render('rgb_array')` + OpenCV `VideoWriter('mp4v')`，若系统编解码器不支持可改用 `MJPG` + `.avi`
- 若渲染返回 `None` 将跳过并打印警告

## 参考
- 论文《基于深度强化学习的Super Mario Bros游戏智能训练》（车景平 等）

---

## Docker (远程服务器)

要求：NVIDIA 驱动 + nvidia-container-toolkit（支持 CUDA 11.8）。

```bash
# 构建
docker build -t mario-ddqn:cu118 .

# 运行（挂载本地 ckpt 输出目录，可选）
docker run --gpus all --rm -it \
  -v $(pwd)/runs:/workspace/runs \
  mario-ddqn:cu118 bash

# 容器内启动训练
python train_ddqn.py --env-id SuperMarioBros-1-1-v0 --total-steps 2000000 --save-dir runs/mario_ddqn
```
