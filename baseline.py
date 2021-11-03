import gym
from stable_baselines3.common import callbacks
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack, VecNormalize, DummyVecEnv, VecVideoRecorder
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.env_checker import check_env
from torchsummary import summary

from wrappers import *
from modules import *

from stable_baselines3.common.vec_env import VecTransposeImage
from stable_baselines3.common.vec_env.stacked_observations import StackedObservations
from stable_baselines3.common.evaluation import evaluate_policy
from icecream import ic

import datetime

from wandb.integration.sb3 import WandbCallback
import wandb
import time
import docker


def create_env_fn(port):
    ic(port)
    #wait_time = random.randint(0, 60) # need here due to seed issues when parallelizing
    def env_fn():
        nonlocal port
        env = gym.make("deepracer_gym:deepracer-v0", port=port)
        env = Extractor2D(env)
        #env = GymEnv(env, is_image=True)
        #env = RewardRework(env)
        #env = gym.wrappers.FrameStack(env, stack_size)
        env = Monitor(env)
        return env
        t.sleep(2) # to prevent race condition
    return env_fn

import numpy as np
import time as t
import click


# TODO have docker start n_envs automatically with gym.make
#PORT = 8888
#STACK_SIZE = 4
#STEPS = 1_000_000
from time import sleep
#env = gym.make("deepracer_gym:deepracer-v0")
@click.command()
@click.option('--n_envs', default=8, help='Number of environments to run in parallel')

# PPO Stuff
@click.option('--learning_rate', default=0.0003, help='Learning rate')
@click.option('--steps', default=1_000_000, help='Number of steps to run the agent')
@click.option('--batch_size', default=256, help='Batch size')
@click.option('--n_epochs', default=4, help='Number of epochs')
@click.option('--gamma', default=0.99, help='Gamma')
@click.option('--gae_lambda', default=0.95, help='GAE lambda')
@click.option('--clip_range', default=0.2, help='Clip range')
#@click.option('--clip_range_vf', default=None, help='Clip range')
@click.option('--ent_coef', default=0.01, help='Entropy coefficient')
@click.option('--vf_coef', default=0.5, help='Value function coefficient')
@click.option('--max_grad_norm', default=0.5, help='Max gradient norm')
#@click.option('--use_sde', default=4, help='Number of frames to stack')

@click.option('--port', default=8888, help='Starting port to use for the server')
@click.option('--stack_size', default=4, help='Number of frames to stack')
@click.option('--eval_episodes', default=1000, help='Number of episodes to evaluate')
@click.option('--rollout', default=8192, help='Number of steps to rollout')
@click.option('--normalize', default=False, help='Normalize')
#@click.option('--redos', default=5, help='Number of times to redo the training')
def main(
    n_envs,
    learning_rate,
    steps,
    batch_size,
    n_epochs,
    gamma,
    gae_lambda,
    clip_range,
    #clip_range_vf,
    ent_coef,
    vf_coef,
    max_grad_norm,

    port,
    stack_size,
    eval_episodes,
    rollout,
    normalize,

):
    config = dict(
        num_envs=n_envs,
        stack_size=stack_size,
        steps=steps
    )
    run = wandb.init(
        project="racer",
        config=config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
    )

    # TODO make_env seems to preserve rollout info
    #env_checker = check_env(gym.make("deepracer_gym:deepracer-v0"))
    #env_fn = create_env_fn()
    # TODO does training envs also have a bias for shorter episodes? 
    env = SubprocVecEnv(
        [create_env_fn(port+idx) for idx in range(n_envs)])
    # TODO probably don't need to pass the port anymore as it's automatically set
    #t.sleep(10)
    #ic(env.observation_space.shape)
    #env.reset()
    #ic(env.render(mode='rgb_array').shape)
    ic(env.observation_space.shape)

    if normalize:
        norm_env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.)
        ic(norm_env.observation_space.shape)
        env = norm_env

    env = VecFrameStack(env, n_stack=stack_size) #, channels_order='last')
    ic(env.observation_space.shape)

    video_count = 10
    video_freq = steps // video_count
    env = VecVideoRecorder(env, f'videos/{run.id}train', record_video_trigger=lambda x: x % (video_freq // n_envs) == 0, video_length=550)
    # env.reset()
    # ic(env.render(mode='rgb_array').shape)
    ic(env.observation_space.shape)
    #env = VecNormalize(env)
    # TODO pass through info for stacker

    # NOTE OFFICIAL GUIANDANCE 1 AGENT = 1 ENV
    # eval_env = SubprocVecEnv(
    #     [create_env_fn(PORT+idx+N_ENVS, stack_size=STACK_SIZE) for idx in range(EVAL_N_ENVS)])
    # #eval_env = VecFrameStack(eval_env, STACK_SIZE, channels_order='first')
    # eval_env = VecFrameStack(eval_env, STACK_SIZE)
    # eval_env = VecVideoRecorder(eval_env, 'videos/eval/', record_video_trigger=lambda _: True, video_length=999999)
    # eval_env = VecTransposeImage(eval_env)
    #single_eval_env = DummyVecEnv(
        #[create_env_fn(PORT+N_ENVS, stack_size=STACK_SIZE)]
    #)
    #single_eval_env = VecVideoRecorder(single_eval_env, 'videos/eval/', record_video_trigger=lambda _: True, video_length=550)

    #ic(eval_env.observation_space.shape)

    # TODO could we drop rollout size since it'll get various xp anyway?
    rollout = rollout // n_envs
    #rollout = (1024*1) // N_ENVS

    # FOR EVAL pass normal env to get normalization wrapper out
    # value = datetime.datetime.fromtimestamp(t.time())
    # folder_name = value.strftime('%Y-%m-%d_%H:%M:%S')

    # TODO run single and multiple at same time. See diff in performance
    # ev_call = EvalCallback(
    #     eval_env,
    #     #single_eval_env,
    #     #n_eval_episodes=max(10, EVAL_N_ENVS*4),
    #     #n_eval_episodes=EVAL_N_ENVS*1,
    #     # NOTE doesn't need to be equal to envs since may fail a lot
    #     n_eval_episodes=EVAL_N_ENVS*2,
    #     #n_eval_episodes=4,
    #     eval_freq=(rollout*8),
    #     #log_path="./logs/",
    #     best_model_save_path=f"models/{folder_name}",
    #     deterministic=True,
    # )
    policy_kwargs = dict(
        #features_extractor_class=SimpleCNN3D,
        #features_extractor_kwargs=dict(features_dim=512),
        # TODO no shared feature extractor?
        #net_arch=[dict(pi=[], vf=[64])]
        #features_extractor_kwargs=dict(
            #net_arch=[dict(
                #pi=[]
            #)]
        #),
    )  

    ic('Starting training')
    model = PPO(
        #"MlpPolicy", 
        "CnnPolicy", 
        env, 
        learning_rate = lambda progression: learning_rate*progression,
        n_steps=rollout,
        gamma=gamma,
        ent_coef=ent_coef,
        clip_range=clip_range,
        #clip_range_vf=clip_range_vf,
        max_grad_norm=max_grad_norm,
        gae_lambda=gae_lambda,
        vf_coef=vf_coef,
        batch_size=batch_size, # TODO need big batches for batch norm?
        # NOTE bigger batch size speeds up training or is it bigger rollout?
        #learning_rate=1e-4,
        n_epochs=n_epochs,
        policy_kwargs=policy_kwargs,
        tensorboard_log="tensorboard_logs/baseline_eval",
        verbose=1)
    # NOTES
    # 1. decreasing learning rate causes a stale out with large batch size
    # 2. larger batch size alllows a larger learning rate
    # 3. Big steps seem to make first eval WAY better
    ic(model.policy)
    #wandb.log_artifact()
    #summary(model.policy, VecTransposeImage(env).observation_space.shape)
    ic('training...')
    model.learn(
        total_timesteps=steps,
        callback=[
            #ev_call,
            WandbCallback(
                gradient_save_freq=10000,
                model_save_path=f'wandb_models/{run.id}',
                verbose=2)
        ],
        #callback=[ev_call, ev2_call],
    )
    ic('done training.')
    if normalize:
        ic('saving normalizer')
        norm_path = f"wandb_models/{run.id}/vec_normalizer.pkl"
        ic(norm_path)
        norm_env.save(norm_path)
        #artifact = wandb.Artifact(norm_path)
        #artifact.add_file(norm_path)
        wandb.save(norm_path)

    env.close_video_recorder()
    env.close()
    t.sleep(20)
    #del env

    eval_env = SubprocVecEnv(
        [create_env_fn(port+idx) for idx in range(n_envs)])
    if normalize:
        eval_env = VecNormalize.load(norm_path, eval_env)
        eval_env.training = False
        eval_env.norm_reward = False
    eval_env = VecFrameStack(eval_env, n_stack=stack_size)
    eval_env = VecVideoRecorder(eval_env, f'videos/{run.id}/eval', record_video_trigger=lambda x: True, video_length=999999999)

    ic('evaluating...')
    episode_rewards, episode_lengths = evaluate_policy(model, eval_env, n_eval_episodes=eval_episodes, deterministic=True, return_episode_rewards=True)
    ic('done evaluating.')

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_length = np.mean(episode_lengths)
    std_length = np.std(episode_lengths)
    wandb.log({
        "eval/mean_reward": mean_reward,
        "eval/std_reward": std_reward,
        "eval/mean_length": mean_length,
        "eval/std_length": std_length,
        "eval/rewards": wandb.Histogram(episode_rewards),
        "eval/lengths": wandb.Histogram(episode_lengths),
    })
    eval_env.close_video_recorder()
    eval_env.close()
    t.sleep(15)
    #eval_env.close()
    #run.finish()

if __name__ == '__main__':
    main()