import gym
import numpy as np
from collections import deque
 
class Extractor(gym.ObservationWrapper):
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0, high=1., shape=(2, 120, 160), dtype=np.float32
        )

    def observation(self, observation):
        obs = observation['STEREO_CAMERAS']
        obs = np.swapaxes(obs, 0, 2)
        obs = np.swapaxes(obs, 2, 1)
        obs = obs / 255.
        assert(type(obs) == np.ndarray)
        return obs

class Stacker(gym.Wrapper):
    def __init__(self, env, stack_size=4):
        super().__init__(env)
        self.frames = None
        self.observation_space = gym.spaces.Box(
            low=0., high=1.,
            shape=(stack_size, *self.env.observation_space.shape),
        )
        self.stack_size = stack_size

    def reset(self):
        obs = self.env.reset()
        self.frames = deque([obs] * self.stack_size, maxlen=self.stack_size)
        state = np.stack(self.frames)
        return state

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.frames.append(obs)
        state = np.stack(self.frames)
        return state, reward, done, info

class RewardRework(gym.Wrapper):
    def __init__(self, env, expected_steps=500):
        super().__init__(env)
        self.steps = 0
        self.expected_steps = expected_steps

    def step(self, action):
        self.steps += 1
        obs, _reward, done, info = self.env.step(action)
        if done and _reward > 0.001:
            reward = self.expected_steps / self.steps
        elif done:
            reward = -1
        else:
            reward = 0

        return obs, reward, done, info

    def reset(self):
        self.steps = 0
        return self.env.reset()