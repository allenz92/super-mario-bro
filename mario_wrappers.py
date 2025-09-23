import gym
import numpy as np
import cv2
from collections import deque


class FrameSkip(gym.Wrapper):
    def __init__(self, env: gym.Env, skip: int = 4):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = False
        info = {}
        obs = None
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info


class GrayResize(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, width: int = 84, height: int = 84):
        super().__init__(env)
        self.width = width
        self.height = height
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.height, self.width), dtype=np.uint8
        )

    def observation(self, observation):
        # observation is HWC RGB
        gray = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return resized


class FrameStack(gym.Wrapper):
    def __init__(self, env: gym.Env, num_stack: int = 4):
        super().__init__(env)
        self.num_stack = num_stack
        h, w = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(num_stack, h, w), dtype=np.uint8
        )
        self.frames = deque(maxlen=num_stack)

    def reset(self):
        obs = self.env.reset()
        for _ in range(self.num_stack):
            self.frames.append(obs)
        return self._get_ob()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        return np.array(self.frames, dtype=np.uint8)


class RewardShaping(gym.Wrapper):
    def __init__(self, env: gym.Env, time_penalty: float = 0.01, death_penalty: float = 50.0):
        super().__init__(env)
        self.time_penalty = time_penalty
        self.death_penalty = death_penalty
        self._last_x = 0

    def reset(self):
        obs = self.env.reset()
        self._last_x = 0
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        # Mario info: x_pos or score can indicate progress; fallback to reward shaping by time
        x_pos = info.get('x_pos', 0)
        speed_reward = max(0, x_pos - self._last_x) * 0.1
        self._last_x = x_pos
        shaped = reward + speed_reward - self.time_penalty
        if done and info.get('life', 1) <= 0:
            shaped -= self.death_penalty
        return obs, shaped, done, info


def make_mario_env(env_id: str, frame_skip: int = 4, stack: int = 4):
    import gym_super_mario_bros
    from nes_py.wrappers import JoypadSpace
    from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

    env = gym_super_mario_bros.make(env_id)
    env = JoypadSpace(env, SIMPLE_MOVEMENT[:4])  # 限制为 4 种动作
    env = FrameSkip(env, skip=frame_skip)
    env = GrayResize(env, 84, 84)
    env = FrameStack(env, num_stack=stack)
    env = RewardShaping(env)
    return env
