import numpy as np
import gym
import math
from deepracer_gym.zmq_client import DeepracerEnvHelper
from icecream import ic
import pickle

class DeepracerGymEnv(gym.Env):
    def __init__(self):
        self.action_space = gym.spaces.Discrete(5)
        self.deepracer_helper = DeepracerEnvHelper()
        self.observation_space = gym.spaces.Box(float("-inf"), float("inf"), shape = (2, 120, 160,))
        self.max_action_count = 0
        self.direction = 0
        self.points = []
        self.actions = []
        self.current_point = (0,0)
        self.points.append(self.current_point)

    def reset(self):
        observation = self.deepracer_helper.env_reset()
        observation, reward, done, info = self.deepracer_helper.unpack_rl_coach_obs(observation)
        obs_array = observation['STEREO_CAMERAS']
        obs_array = np.swapaxes(obs_array, 0, 2)
        if len(self.actions) > self.max_action_count:
            ic(self.points)
            self.max_action_count = len(self.actions)
            with open('points.pkl', 'wb') as f:
                pickle.dump(self.points, f)
        self.actions.clear()
        self.points.clear()
        self.current_point = (0,0)
        self.points.append(self.current_point)
        self.direction = 0

        return obs_array
    
    def step(self, action):
        rl_coach_obs = self.deepracer_helper.send_act_rcv_obs(action)
        observation, reward, done, info = self.deepracer_helper.unpack_rl_coach_obs(rl_coach_obs)
        obs_array = observation['STEREO_CAMERAS']
        obs_array = np.swapaxes(obs_array, 0, 2)
        self.actions.append(action)
        self.current_point = self.new_point(action)
        self.points.append(self.current_point)
        return obs_array, reward, done, info
    
    def new_point(self, action):
        angle = 0
        if action == 0:
           angle = math.pi / -6.0
        elif action == 1:
           angle = math.pi / -12.0
        elif action == 2:
           angle = 0
        elif action == 3:
           angle = math.pi / 12.0
        elif action == 4:
           angle = math.pi / 6.0
        
        self.direction += angle

        return (self.current_point[0] + 0.6*math.cos(self.direction), self.current_point[1] + 0.6*math.sin(self.direction))

if __name__ == '__main__':
    env = DeepracerGymEnv()
    obs = env.reset()
    steps_completed = 0
    episodes_completed = 0
    total_reward = 0
    for _ in range(500):
        observation, reward, done, info = env.step(np.random.randint(5))
        steps_completed += 1 
        total_reward += reward
        if done:
            episodes_completed += 1
            print("Episodes Completed:", episodes_completed, "Steps:", steps_completed, "Reward", total_reward)
            steps_completed = 0
            total_reward = 0
