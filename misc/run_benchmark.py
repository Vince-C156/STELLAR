import numpy as np
import sys
sys.path.append("gym-env")
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
from typing import Callable
import gymnasium as gym
import os
import torch as T
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecExtractDictObs, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList, ProgressBarCallback, BaseCallback
from time import sleep
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from visualizer_all import write2text
from gymnasium.wrappers import TimeLimit
from torch.nn import GELU, LeakyReLU, ReLU, ELU
from stable_baselines3.common.monitor importnp.array(inital_conditions)
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
            "train/loss": 0.0,
            "train/explained_variance": 0.0
        }
        metric_dict = {"rollout/ep_len_mean": 0,
                        "rollout/ep_rew_mean": 0}
        self.logger.record(
            "hparams",
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
        :return: (np.ndarray, float, bool, bool, dict) observation, reward, is the episode      		over?, additional informations
        """
        self.current_step += 1
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Overwrite the truncation signal when when the number of steps reaches the maximum
        if self.current_step >= self.max_steps:
            truncated = True
        return obs, reward, terminated, truncated, info


class state_dataCB(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseRLModel
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env = None  # type: Union[gym.Env, VecEnv, None]
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = None  # type: Dict[str, Any]
        # self.globals = None  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger = None  # type: logger.Logger
        # # Sometimes, for event callback, it is useful
        # # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """

        print(self.locals)
        return TruePPO.load(policy_dir, env=normalized_vec_env, print_system_info=True, policy_kwargs=policy_args, _init_setup_model=False, force_reset=True, device='cpu')

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass

class EvalCallback(BaseCallback):
    """
    Callback for evaluating an agent.

    :param eval_env: (gym.Env) The environment used for initialization
    :param n_eval_episodes: (int) The number of episodes to test the agent
    :param eval_freq: (int) Evaluate the agent every eval_freq call of the callback.
    """

    def __init__(self, eval_env, n_eval_episodes=5, eval_freq=20):
        super().__init__()
        self.eval_env = eval_env
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.best_mean_(class) TimeLimitWrapper
reward = -np.inf

    def _on_step(self):
        """
        This method will be called by the model.

        :return: (bool)
        """
        print("Best mean reward: {:.2f}".format(self.best_mean_reward))
        # self.n_calls is automatically updated because
        # we derive from BaseCallback
        if self.n_calls % self.eval_freq == 0:
            # === YOUR CODE HERE ===#
            # Evaluate the agent:
            # you need to do self.n_eval_episodes loop using self.eval_env
            # hint: you can use self.model.predict(obs, deterministic=True)

            # Save the agent if needed
            # and update self.best_mean_reward

            print("Best mean reward: {:.2f}".format(self.best_mean_reward))

            # ====================== #
        return True


def evaluate(model, num_episodes=100):
    """
    Evaluate a RL agent
    :param model: (BaseRLModel object) the RL Agent
    :param num_episodes: (int) number of episodes to evaluate it
    :return: (float) Mean reward for the last num_episodes
    """
    # This function will only work for a single Environment
    env = model.get_vec_normalize_env()
    all_episode_rewards = []
    all_terminalstates = []
    episode_lengths = []
    epsiode_fuel = []
    runtime = []

    for i in tgrange(num_episodes,  ncols= 1000, mininterval=0.3, gui=True, desc="Evaluating"):
        sleep(0.1)
        episode_rewards = []
        done = False
        obs = env.reset()
        ep_steps = 0
        fuel_consumed = -1
        while not done:
            # _states are only useful when using LSTM policies

            st = time.time()
            action, _states = model.predict(obs, deterministic=True)
            et = time.time()

            model_runtime = et - st
            runtime.append(model_runtime)

            # here, action, rewards and dones are arrays
            # because we are using vectorized env
            obs, reward, done, info = env.step(action)

            if info[0]['docked_step'] != -1 and fuel_consumed == -1:
                fuel_consumed = info[0]['fuel consumed']
            elif info[0]['collision_step'] != -1 and fuel_consumed == -1:
                fuel_consumed = info[0]['fuel consumed']

            episode_rewards.append(reward)
            ep_steps += 1

        if info[0]['docked_step'] != -1:
            episode_lengths.append(info[0]['docked_step'])
            unnormalized_state = info[0]['docked_state']
        elif info[0]['collision_step'] != -1:
            episode_lengths.append(info[0]['collision_step'])
            unnormalized_state = info[0]['collision_state']
        else:
            episode_lengths.append(ep_steps)
            unnormalized_state = env.unnormalize_obs(info[0]['terminal_observation'])
            fuel_consumed = info[0]['fuel consumed']


        epsiode_fuel.append(fuel_consumed)

        print("----------------------------------------------")
        print(f"Appended {unnormalized_state} to all_terminalstates")
        print("----------------------------------------------")
        all_terminalstates.append(unnormalized_state)
        all_episode_rewards.append(sum(episode_rewards))

    runtime = np.asarray(runtime)
    runtime_sum = 0

    for i in range(len(runtime)):
        runtime_sum += runtime[i]
    
    runtime_mean = runtime_sum / num_episodes

    episode_length_sum = 0

    for i in range(len(episode_lengths)):
        episode_length_sum += episode_lengths[i]
    length_mean = episode_length_sum / num_episodes


    fuel_sum = 0
    for i in range(len(epsiode_fuel)):
        fuel_sum += epsiode_fuel[i]
    fuel_mean = fuel_sum / num_episodes


    mean_episode_reward = np.mean(all_episode_rewards)
    all_terminalstates = np.asarray(all_terminalstates)
    #mean_terminalstate = np.mean(all_terminalstates[:, :3], axis=1)
    #mean_termninalvel = np.mean(all_terminalstates[:, 3:], axis=1)
    #mean_full_state = np.concatenate((mean_terminalstate, mean_termninalvel))

    all_x = all_terminalstates[:, 0]
    all_y = all_terminalstates[:, 1]
    all_z = all_terminalstates[:, 2]
    all_vx = all_terminalstates[:, 3]
    all_vy = all_terminalstates[:, 4]
    all_vz = all_terminalstates[:, 5]

    print(all_y)

    mean_x = np.sum(all_x) / num_episodes
    mean_y = np.sum(all_y) / num_episodes
    mean_z = np.sum(all_z) / num_episodes
    mean_vx = np.sum(all_vx) / num_episodes
    mean_vy = np.sum(all_vy) / num_episodes
    mean_vz = np.sum(all_vz) / num_episodes

    std_x = np.std(all_x)
    std_y = np.std(all_y)
    std_z = np.std(all_z)
    std_vx = np.std(all_vx)
    std_vy = np.std(all_vy)
    std_vz = np.std(all_vz)


    mean_terminal_obs = np.asarray([mean_x, mean_y, mean_z, mean_vx, mean_vy, mean_vz])
    std_terminal_obs = np.asarray([std_x, std_y, std_z, std_vx, std_vy, std_vz])
    print(mean_terminal_obs)
    print("Mean reward:", mean_episode_reward, "Num episodes:", num_episodes)
    print(epsiode_fuel)

    #return mean_episode_reward, np.ndarray([mean_terminal_obs]), np.asarray([length_mean]), np.asarray([fuel_mean]), np.asarray([runtime_mean])

    terminal_data = {'Average-State' : mean_terminal_obs, 'State-Std' : std_terminal_obs, 'State-Median' : np.median(all_terminalstates, axis=0)}
    fuel_data = {'Average-Fuel' : fuel_mean, 'Fuel-Std' : np.std(epsiode_fuel), 'Fuel-Median' : np.median(epsiode_fuel)}
    eplen_data = {'Average-Episode-Length' : length_mean, 'Episode-Length-Std' : np.std(episode_lengths), 'Episode-Length-Median' : np.median(episode_lengths)}
    ep_data = {'Average-Episode-Length' : length_mean, 'Episode-Length-Std' : np.std(episode_lengths), 'Episode-Length-Median' : np.median(episode_lengths)}

    return terminal_data, fuel_data, eplen_data, runtime_mean
    #return mean_episode_reward, mean_terminal_obs, np.asarray([[length_mean]]), np.asarray([[fuel_mean]]), np.asarray([[runtime_mean]])
#policy_dict = {'activation_fn': GELU, 'net_arch': {'pi': [130, 44, 30], 'vf': [145, 25, 5]}, 'full_std': False, 'use_expln' : True, 
#                'squash_output': True, 'log_std_init': -2.3, 'use_expln' : True, 'ortho_init': False}


env = make_vec_env(ARPOD_GYM, wrapper_class=TimeLimitWrapper, n_envs=1)
env = VecMonitor(env, filename='envmonitor')
#god file chaser_vecnormalize_400000_steps.pkl
#not god file chaser_arpod_vecnormalize_2500000_steps.pkl
#vbar-final.pkl
stats_path = os.path.join('envstats', 'vbar-agentv3-3.pkl')
normalized_vec_env = VecNormalize.load(stats_path, env)
normalized_vec_env.training = False
#normalized_vec_env.norm_reward = True
#normalized_vec_env.norm_obs = True
#normalized_vec_env.clip_obs = 10.0
#normalized_vec_env.clip_reward = 10.0

modelname = 'vbar-agent_1250000_steps.zip'
model_dir = f'models/{modelname}'
#loaded_model = PPO.load(model_dir, env=normalized_vec_env, print_system_info=True, force_reset=True)

#policy_kwargs=policy_dict)


def mean_stats_table(mean_fuel_per_episode, mean_time_per_episode, mean_terminal_state, mean_runtime_per_step):



    columns = ['Label', 'Value']
    rows = ['Average-Time/Episode', 'Average-Fuel/Episode', 'Average-Runtime/Step']
    data = [mean_time_per_episode, mean_fuel_per_episode, mean_runtime_per_step]
    data = np.array(data).T  # Transpose to match with rows and columns

    # Create a 0x0 figure (not to be displayed)
    fig, ax = plt.subplots()
    ax.axis('tight')
    ax.axis('off')

    # Add a table
    table = ax.table(cellText=data,
            rowLabels=rows,
            colLabels=columns,
            cellLoc = 'center', 
            loc='center')

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    fig.tight_layout()

    plt.show()
usr_policy = input("Do you want to change the policy? (y/n)")

if usr_policy == 'y':
    policy_args = {'activation_fn': GELU, 'net_arch': {'pi': [130, 44, 30], 'vf': [145, 25, 5]}, 'full_std': False, 'squash_output': False, 'log_std_init': 1.41421356237, 'ortho_init': True, 'use_expln' : True}
    policy_dir = 'models/vbar-agentv3-3.zip'
    fresh_model = PPO.load(policy_dir, env=normalized_vec_env, print_system_info=True, policy_kwargs=policy_args, _init_setup_model=False, force_reset=True, device='cpu')


    terminal_data, fuel_data, eplen_data, runtime_mean = evaluate(model=fresh_model, num_episodes=100)

    print("Terminal State Statistics")
    print("=====================================")
    print(f"Mean Terminal State: {terminal_data['Average-State']}")
    print(f"Standard Deviation Terminal States: {terminal_data['State-Std']}")
    print(f"Median Terminal State: {terminal_data['State-Median']}")
    print("=====================================")
    print("Fuel Statistics")
    print("=====================================")
    print(f"New policy average fuel: {fuel_data['Average-Fuel']} newtons")
    print(f"New policy fuel standard deviation: {fuel_data['Fuel-Std']} newtons")
    print(f"New policy fuel median: {fuel_data['Fuel-Median']} newtons")
    print("=====================================")
    print("Episode Length Statistics")
    print("=====================================")
    print(f"New policy average time: {eplen_data['Average-Episode-Length']} steps/seconds per mission")
    print(f"New policy episode length standard deviation: {eplen_data['Episode-Length-Std']} steps/seconds per mission")
    print(f"New policy episode length median: {eplen_data['Episode-Length-Median']} steps/seconds per mission")
    print("=====================================")
    print("Runtime Statistics")
    print("=====================================")
    print(f"New policy average runtime per step: {runtime_mean} seconds")

else:
    evaluate_policy(model=loaded_model, env=normalized_vec_env, n_eval_episodes=100, deterministic=False)
#new_model.set_parameters(pretrained_params)
#evaluate_policy(model=loaded_model, env=normalized_vec_env, n_eval_episodes=100, deterministic=False)

"""
env = make_vec_env(ARPOD_GYM, wrapper_class=TimeLimitWrapper, n_envs=1)

model_replay = 'chaser_vecnormalize_400000_steps.pkl'
stats_path = os.path.join('envstats', model_replay)
env = VecNormalize.load(stats_path, env)

#  do not update them at test time
env.training = False
# reward normalization is not needed at test time
env.norm_reward = False
"""
