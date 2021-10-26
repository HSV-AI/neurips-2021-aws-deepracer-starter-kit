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
            #shape=(120, 160, 2, 1,),
            #shape=(2, 120, 160,),
            shape=(1, 120, 160,),
            #shape=(1, 2, 120, 160,),
            dtype=np.uint8
        )
    
    def reset(self):
        observation = self.deepracer_helper.env_reset()
        observation, reward, done, info = self.deepracer_helper.unpack_rl_coach_obs(observation)
        obs_array = observation['STEREO_CAMERAS']
        #obs_array = np.swapaxes(obs_array, 0, 2)
        self.direction = 0

        return self.reshape(obs_array)
    
    @staticmethod
    def reshape(obs):
        obs = np.swapaxes(obs, 0, 2)
        obs = np.swapaxes(obs, 2, 1)
        #obs = obs[1:]
        #obs = np.reshape(obs, (1, 120*2, 160,))
        #obs = np.reshape(obs, (1, 120, 160,))
        #obs = np.expand_dims(obs, axis=0)
        obs = obs[1:]
        return obs
        
    def step(self, action):
        rl_coach_obs = self.deepracer_helper.send_act_rcv_obs(action)
        #ic(rl_coach_obs)
        #exit()
        observation, reward, done, info = self.deepracer_helper.unpack_rl_coach_obs(rl_coach_obs)
        obs_array = observation['STEREO_CAMERAS']
        #obs_array = np.swapaxes(obs_array, 0, 2)
        #self.actions.append(action)
        #self.current_point = self.new_point(action)
        #self.points.append(self.current_point)
        #self.observations.append(obs_array)
        #return self.reshape(obs_array), reward, done, info
        if reward > 1e-3 and done:
            reward = 1
        elif done:
            reward = -1
        else: 
            reward = 0
        if info['goal'] is not None:
            ic(info['goal'])
            ic(reward)
            ic("GOOOOOOOOOOOAAAAAAAAAAAAAAAAAALLLLLLLLLLLLLL")
            print("GOOOOOOOOOOOAAAAAAAAAAAAAAAAAALLLLLLLLLLLLLL")
            reward += 101
        return self.reshape(obs_array), reward, done, info

    
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

        return (self.current_point[0] + 0.04*math.cos(self.direction), self.current_point[1] + 0.04*math.sin(self.direction))

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
