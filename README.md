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

## 主要文件
- `mario_wrappers.py`: 环境包装与图像预处理
- `ddqn.py`: 模型结构与目标网络同步
- `replay_buffer.py`: 经验回放
- `train_ddqn.py`: 训练循环与日志

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
