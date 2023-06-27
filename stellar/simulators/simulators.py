import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import gymnasium as gym
from torch.nn import GELU, LeakyReLU, ReLU, ELU
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stellar.arpodenvs.environment import ARPOD_GYM
from stellar.arpodenvs.dynamics import chaser_continous
#from stellar.arpodenvs.reward_shaping import reward_formulation
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


class simulate_arpod:

    def __init__(self, model_file: str, inital_conditions: list, deterministic: bool):

        """
        Initialize the simulate_arpod class with the specified parameters.

        Parameters:
            - model_file (str): The file name or path of the model to be used for simulation.
            - inital_conditions (list): A list of initial conditions [x, y, z, xdot, ydot, zdot] for the simulation.
            - deterministic (bool): A flag indicating whether the simulation should be deterministic or not.

        Raises:
            - AssertionError: If the initial conditions are not of type list or if their length is not equal to 6.
            - AssertionError: If the specified model file does not exist.
            - AssertionError: If the specified model file does not exist in the directory.

        Args:
            model_file (str): The file name or path of the model to be used for simulation.
            inital_conditions (list): A list of initial conditions [x, y, z, xdot, ydot, zdot] for the chaser during simulation.
            deterministic (bool): A flag indicating whether the simulation should be deterministic or not.

        """
        x0_len, x0_type = len(inital_conditions), type(inital_conditions)
        model_dir = os.path.join(__file__.strip(os.path.basename(__file__)), f'../models/{model_file}')
        working_dir = os.path.dirname(os.path.realpath(__file__))
        #validing user inputs
        assert type(
            inital_conditions) == list, f'Parameter inital_conditions is type {x0_type} but need to be a list of length 6'
        
        assert type(deterministic) == bool, f'Parameter deterministic is type {type(deterministic)} but is expecting type bool'

        assert len(
            inital_conditions) == 6, f'Parameter inital_conditions is the correct type {x0_type}, however is length {x0_len} but needs to be 6 to represent [x y z xdot ydot zdot]'

        assert os.path.exists(model_dir) == True, f'{model_file} does not exist in directory {os.path.basename(model_dir)}. Existing options are {os.listdir(model_dir)}'

        #loading environment
        print("Initalizing environment")
        env = make_vec_env(ARPOD_GYM, wrapper_class=TimeLimitWrapper, n_envs=1)
        envpkl = 'vbar-agentv3-3.pkl'
        envpkl_path = os.path.split(working_dir)[0]
        print(envpkl_path)
        envpkl_path = os.path.join(envpkl_path, 'arpodenvs', 'envstats', envpkl)

        print(f"Loading env file {envpkl_path}")
        self.env = VecNormalize.load(envpkl_path, env)
        print("Environment Initalized : Success")

        #loading model
        print("Initalizing PPO Model")
        self.policy_args = {'activation_fn': GELU, 'net_arch': {'pi': [130, 44, 30], 'vf': [145, 25, 5]}, 'full_std': False, 'squash_output': False, 'log_std_init': 1.41421356237, 'ortho_init': True, 'use_expln' : True}
        self.model_dir = model_dir
        self.deterministic = deterministic
        print(f"Loading PPO from {self.model_dir}")
        self.model = PPO.load(self.model_dir, env=self.env, print_system_info=True, policy_kwargs=self.policy_args, _init_setup_model=False, force_reset=True, device='cpu')
        self.x0 = inital_conditions
        print("PPO Model Initalization : Success")




