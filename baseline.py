import gym
import torch as th

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack, VecNormalize, DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            # Conv2d *should* make kernel 3d automatically over stero channels 
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))


#env = gym.make("deepracer_gym:deepracer-v0")
def main():
    env = make_vec_env(
        "deepracer_gym:deepracer-v0", 
        n_envs=1,
        vec_env_cls=SubprocVecEnv)
    env = VecFrameStack(env, n_stack=4)
    env = VecNormalize(env, norm_obs=True, norm_reward=False)

    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=128),
    )

    model = PPO(
        "MlpPolicy", 
        env, 
        policy_kwargs=policy_kwargs,
        tensorboard_log="tensorboard_logs/baseline",
        verbose=1)
    model.learn(total_timesteps=2000000)

if __name__ == '__main__':
    main()