import gym

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack, VecNormalize, DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv

#env = gym.make("deepracer_gym:deepracer-v0")
def main():
    env = make_vec_env(
        "deepracer_gym:deepracer-v0", 
        n_envs=1,
        vec_env_cls=SubprocVecEnv)
    env = VecFrameStack(env, n_stack=4)
    env = VecNormalize(env, norm_obs=True, norm_reward=False)

    model = PPO(
        "CnnPolicy", 
        env, 
        tensorboard_log="tensorboard_logs/baseline",
        verbose=1)
    model.learn(total_timesteps=2000000)

if __name__ == '__main__':
    main()