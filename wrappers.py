import gym
import numpy as np
from collections import deque
 
# class Extractor3D(gym.ObservationWrapper):
#     def __init__(self, env: gym.Env) -> None:
#         super().__init__(env)
#         self.observation_space = gym.spaces.Box(
#             low=0, high=1., shape=(2, 120, 160), dtype=np.float32
#         )

#     def observation(self, observation):
#         obs = observation['STEREO_CAMERAS']
#         obs = np.moveaxis(obs, 2, 0)
#         obs = obs / 255.
#         assert(type(obs) == np.ndarray)
#         return obs

class Extractor2D(gym.ObservationWrapper):

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            #low=0, high=1., shape=(120, 160), dtype=np.float32
            low=0, high=255, shape=(120, 160, 1), dtype=np.uint8
        )
        self.last_obs = None
        self.metadata = {'render.modes': ['rgb_array'], 'video.frames_per_second': 15 }

    def observation(self, observation):
        obs = observation['STEREO_CAMERAS']
        #obs = np.moveaxis(obs, 1, 0)
        obs = obs[..., 0:1]
        self.last_obs = obs
        #obs = obs / 255.
        #obs = np.moveaxis(obs, 2, 0)
        return obs

    def render(self, mode='rgb_array'):
        if mode == 'rgb_array':
            #img = np.stack([self.last_obs]*3, axis=-1)
            img = np.concatenate([self.last_obs]*3, axis=-1)
            #assert img.shape == (120, 160, 3), f"{img.shape}"
            #img = np.moveaxis(img, 2, 0)
            #assert img.shape == (3, 120, 160)

            #img = np.expand_dims(img, axis=0)
            #assert img.shape == (1, 120, 160, 3)
            #img = np.moveaxis(img, -1, 1)
            return img

# class Stacker(gym.wrappers.FrameStack):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.last_obs = None
    
#     def step(self, action):
#         obs, reward, done, info = super().step(action)
#         self.last_obs = obs
#         return obs, reward, done, info
    
#     def reset(self, **kwargs):
#         obs = super().reset(**kwargs)
#         self.last_obs = obs
#         return obs
    
#     def render(self, mode='rgb_array'):
#         if mode == 'rgb_array':
#             img = np.stack([self.last_obs[-1]]*4, axis=-1)
#             #print(img.shape)
#             return img

class BasicRewardRework(gym.Wrapper):
    def step(self, action):
        self.steps += 1
        obs, _reward, done, info = self.env.step(action)
        if done and _reward > 0.001:
            reward = 1.
        elif done:
            reward = -1.
        else:
            reward = 0.

        return obs, reward, done, info
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