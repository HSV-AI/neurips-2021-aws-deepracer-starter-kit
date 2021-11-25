import numpy as np
import gym
from deepracer_gym.zmq_client import DeepracerEnvHelper

class DeepracerGymEnv(gym.Env):
    def __init__(self, port=8888):
        self.action_space = gym.spaces.Discrete(5)
        self.deepracer_helper = DeepracerEnvHelper(port=port)
        self.observation_space = gym.spaces.Box(
            low=float(0), 
            high=float(255), 
            #shape=(2, 160, 120,),
            shape=(120, 160, 2,),
            dtype=np.uint8
        )
    
    def reset(self):
        observation = self.deepracer_helper.env_reset()
        observation, reward, done, info = self.deepracer_helper.unpack_rl_coach_obs(observation)
        return observation
    
    def step(self, action):
        rl_coach_obs = self.deepracer_helper.send_act_rcv_obs(action)
        observation, reward, done, info = self.deepracer_helper.unpack_rl_coach_obs(rl_coach_obs)
        return observation, reward, done, info

    
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
