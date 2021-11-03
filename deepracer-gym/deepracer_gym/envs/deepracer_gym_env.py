import numpy as np
import gym
from deepracer_gym.zmq_client import DeepracerEnvHelper
import docker
import time
import fasteners


def is_port_in_use(port):
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def start_container(port):
    client = docker.from_env()

    # lock so only one process can start the container at a time to prevent port conflicts
    #fcntl.flock(f, fcntl.LOCK_EX)
    container = client.containers.run(
        'aicrowd/base-images:deepracer_round1_release',
        command='/bin/bash',
        detach=True,
        ports={8888: port},)

    return container

import random

class DeepracerGymEnv(gym.Env):
    lock = fasteners.InterProcessLock('/tmp/deepracer_gym_env.lock')

    def __init__(self, port=8888):
        self.action_space = gym.spaces.Discrete(5)

        # TODO better race condition prevention
        #wait_time = random.randint(0, 20)

        #time.sleep(wait_time)
        # find nearest open port
        self.container = None
        with self.lock:
            while is_port_in_use(port):
                # TODO could be a race condition here
                port += 1
            self.container = start_container(port)
            # Give time for container to get port
            time.sleep(15)

        self.deepracer_helper = DeepracerEnvHelper(port=port)
    
    def reset(self):
        observation = self.deepracer_helper.env_reset()
        observation, reward, done, info = self.deepracer_helper.unpack_rl_coach_obs(observation)
        return observation
    
    def step(self, action):
        rl_coach_obs = self.deepracer_helper.send_act_rcv_obs(action)
        observation, reward, done, info = self.deepracer_helper.unpack_rl_coach_obs(rl_coach_obs)
        return observation, reward, done, info
    
    def close_container(self):
        if self.container:
            self.container.stop()
            self.container.remove()
            self.container = None

    def close(self):
        #self.container.stop()
        #self.container.remove(force=True)
        self.close_container()

    
    def __del__(self):
        self.close()
    
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
