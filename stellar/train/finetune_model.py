import numpy as np
from environment import ARPOD_GYM
from dynamics import chaser_continous
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.logger import HParam
import gymnasium as gym
from gymnasium import spaces
#from gymnasium import spaces
from gymnasium.wrappers import TimeLimit
import os
import torch as T
from torch.nn import GELU, LeakyReLU, ReLU, ELU
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecExtractDictObs, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList, ProgressBarCallback, EvalCallback, BaseCallback
from time import sleep
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from typing import Callable

"""
policy_dict = {'activation_fn': GELU, 'net_arch': {'pi': [130, 44, 30], 'vf': [145, 25, 5]}, 'full_std': False, 'squash_output': True, 'log_std_init': 0, 'use_expln' : True}

 gamma=0.998,
                                   gae_lambda=0.95,
                                   learning_rate=0.04,
                                   n_steps=500,
                                   ent_coef=0.015,
                                   clip_range=0.2,
                                   target_kl=0.001,
                                   batch_size=2500,
                                   verbose=1,
                                   n_epochs=20,
                                   normalize_advantage=True,
                                   max_grad_norm=0.5,
                                   use_sde=True,
                                   tenmake_vec_env(ARPOD_GYM, wrapper_class=TimeLimitWrapper, n_envs=1)sorboard_log='arpod-ppo',
                                   sde_sample_freq=4,
                                   _init_setup_model=True,
                                   device='cuda',
                                   policy_kwargs=policy_dict)
"""
class HParamCallback(BaseCallback):
    """
    Saves the hyperparameters and metrics at the start of the training, and logs them to TensorBoard.
    """
    def __init__(self, verbose=1):
        super().__init__(verbose)

    def _on_training_start(self) -> None:
        hparam_dict = {
            "algorithm": self.model.__class__.__name__,
            "learning rate": self.model.learning_rate,
            "gamma": self.model.gamma,
            "use_sde": self.model.use_sde,
            "sde_sample_freq": self.model.sde_sample_freq,
            "target_kl": self.model.target_kl,
            "n_steps": self.model.n_steps,
            "ent_coef": self.model.ent_coef,
            "batch_size": self.model.batch_size,
            "gae_lambda": self.model.gae_lambda,
            "normalize_advantage": self.model.normalize_advantage,
            "max_grad_norm": self.model.max_grad_norm,
            "clip_range": self.model.clip_range,
            "n_epochs": self.model.n_epochs
        }
        # define the metrics that will appear in the `HPARAMS` Tensorboard tab by referencing their tag
        # Tensorbaord will find & display metrics from the `SCALARS` tab
        metric_dict2 = {
            "rollout/ep_len_mean": 0,
            "rollout/ep_rew_mean": 0,
            "train/value_loss": 0.0,
            "train/entropy_loss": 0.0,
            "train/policy_gradient_loss": 0.0,
            "train/approx_kl": 0.0,
            "train/clip_fraction": 0.0,
            "train/clip_range": 0.0,
            "train/n_updates": 0,
            "train/learning_rate": 0.0,
            "train/std": 0.0,
            "train/loss": 0.0,simulate_arpod
            HParam(hparam_dict, metric_dict),
            exclude=("stdout", "log", "json", "csv"),
        )

    def _on_step(self) -> bool:
        return True

class TimeLimitWrapper(gym.Wrapper):
    """
    :param env: (gym.Env) Gym environment that will be wrapped
    :param max_steps: (int) Max number of steps per episode
    """

    def __init__(self, env, max_steps=2500):
        # Call the parent constructor, so we can access self.env later
        super(TimeLimitWrapper, self).__init__(env)
        self.max_steps = max_steps
        # Counter of steps per episode
        self.current_step = 0

    def reset(self, **kwargs):
        """
        Reset the environment
        """
        # Reset the counter
        self.current_step = 0
        obs, info = self.env.reset()
        return obs, info

    def step(self, action):
        """
        :param action: ([float] or int) Action taken by the agent
        :return: (np.ndarray, float, bool, bool, dict) observation, reward, is the episode      over?, additional informations
        """
        self.current_step += 1
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Overwrite the truncation signal when when the number of steps reaches the maximum
        if self.current_step >= self.max_steps:
            truncated = True
        return obs, reward, terminated, truncated, info


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func


#progress_callback = ProgressBarCallback()
checkpoint_callback = CheckpointCallback(
  save_freq=500,
  save_path="./logs/",
  name_prefix="chaser2_arpod",
  save_replay_buffer=True,
  save_vecnormalize=True,
)

callback = CallbackList([CheckpointCallback(save_freq=2500, save_path="./logs2/", name_prefix="vbar-agentv3", save_replay_buffer=True, save_vecnormalize=True)])



env = make_vec_env(ARPOD_GYM, wrapper_class=TimeLimitWrapper, n_envs=30)
stats_path = os.path.join('envstats', 'vbar-agent_vecnormalize_2250000_steps.pkl')
normalized_vec_env = VecNormalize.load(stats_path, env)
normalized_vec_env.training = True
#normalized_vec_env.norm_reward = True
#normalized_vec_env.norm_obs = True
#normalized_vec_env.clip_obs = 100.0
#normalized_vec_env.clip_reward = 100.0


modelname = 'vbar-finalv3.zip'
modelname = 'vbar-agent_2250000_steps.zip'
#modelname = 'vbar-500-800m.zip'
model_dir = f'models/{modelname}'

policy_dict = {'activation_fn': GELU, 'net_arch': {'pi': [130, 44, 30], 'vf': [145, 25, 5]}, 'full_std': False, 'squash_output': False, 'log_std_init': -1, 'ortho_init': True, 'use_expln' : True}
#'log_std_init': 1.41421356237, 'ortho_init': True

"""
mean_params = dict(
    (key, value)
    for key, value in loaded_model.policy.state_dict().items()
    if "action" in key
)

print(mean_params)

last_policy = loaded_model.policy
"""
loaded_model = PPO.load(model_dir, env=normalized_vec_env, 
                                    gamma=0.998,
                                   gae_lambda=0.98,
                                   learning_rate=0.015,
                                   n_steps=500,
                                   ent_coef=0.0002,
                                   clip_range=0.2,
                                   target_kl=0.002,
                                   batch_size=1000,
                                   verbose=1,
                                   n_epochs=20,
                                   normalize_advantage=True,
                                   max_grad_norm=0.2,
                                   use_sde=True,
                                   tensorboard_log='arpod2-ppo',
                                   sde_sample_freq=8,
                                   _init_setup_model=True,
                                   device='cuda',
                                   print_system_info=True, 
                                   force_reset=True)
loaded_model.policy.policy_kwargs = policy_dict

new_model = PPO(ActorCriticPolicy, env=normalized_vec_env, 
                                   gamma=0.998,
                                   gae_lambda=0.97,
                                   learning_rate=0.07,
                                   n_steps=500,
                                   ent_coef=0.08,
                                   clip_range=0.25,
                                   target_kl=0.035,
                                   batch_size=2500,np.array(inital_conditions)
                                   _init_setup_model=True,
                                   device='cuda',
                                   policy_kwargs=policy_dict)
#target_kl=0.001

episodes = 100000
training_steps = 2500.0 * episodes

loaded_model.learn(total_timesteps=training_steps, reset_num_timesteps=True, progress_bar=True, callback=callback, tb_log_name="ppo-final")
loaded_model.save("vbar-agentv3")

"""
new_net = dict(
    (key, value)
    for key, value in new_model.policy.state_dict().items()
)

default_params_vec = new_model.policy.parameters_to_vector()

default_params = new_model.get_parameters()
pretrained_params = loaded_model.get_parameters()

new_model.set_parameters(pretrained_params)
loaded_params_vec = new_model.policy.parameters_to_vector()

transfered_params = new_model.get_parameters()

loaded_net = dict(
    (key, value)
    for key, value in new_model.policy.state_dict().items()
)

weight_transfer_flag = (default_params_vec == loaded_params_vec).all()
print(f"Parameters transfer success status {weight_transfer_flag}")

if weight_transfer_flag:
    raise ValueError("Parameters transfer failed")
episodes = 20000
training_steps = 2500.0 * episodes


new_model.learn(total_timesteps=training_steps, reset_num_timesteps=True, progress_bar=True, callback=callback, tb_log_name="ppo-final")
new_model.save("vbar-finalv5-2")
"""

