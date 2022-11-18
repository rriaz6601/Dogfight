from functools import wraps
from time import sleep
from typing import Optional, Union

from gym import Wrapper, spaces

from src.async_arch.envs.create_env import create_env
from src.async_arch.utils.utils import AttrDict, is_module_available, log

# TODO: need to implement other functions in this to make my environment work as required


def is_multiagent_env(env):
    """

    Parameters
    ----------
    env :


    Returns
    -------

    """
    is_multiagent = hasattr(env, "num_agents") and env.num_agents > 1
    if hasattr(env, "is_multiagent"):
        is_multiagent = env.is_multiagent

    return is_multiagent


class MultiAgentWrapper(Wrapper):
    """This wrapper allows us to treat a single-agent environment as multi-agent with 1 agent.
    That is, the data (obs, rewards, etc.) is converted into lists of length 1

    Parameters
    ----------

    Returns
    -------

    """

    def __init__(self, env):
        super().__init__(env)

        self.num_agents = 1
        self.is_multiagent = True

    def reset(self, **kwargs):
        """

        Parameters
        ----------
        **kwargs :


        Returns
        -------

        """
        obs = self.env.reset(**kwargs)
        return [obs]

    def step(self, action):
        """

        Parameters
        ----------
        action :


        Returns
        -------

        """
        action = action[0]
        obs, rew, done, info = self.env.step(action)
        if done:
            obs = self.env.reset()
        return [obs], [rew], [done], [info]


class DictObservationsWrapper(Wrapper):
    """ """

    def __init__(self, env):
        super().__init__(env)
        self.num_agents = env.num_agents
        self.observation_space = spaces.Dict(dict(obs=self.observation_space))

    def reset(self, **kwargs):
        """

        Parameters
        ----------
        **kwargs :


        Returns
        -------

        """
        obs = self.env.reset(**kwargs)
        return [dict(obs=o) for o in obs]

    def step(self, action):
        """

        Parameters
        ----------
        action :


        Returns
        -------

        """
        obs, rew, done, info = self.env.step(action)
        return [dict(obs=o) for o in obs], rew, done, info


def make_env_func(
    cfg, env_config: Optional[AttrDict]
) -> Union[MultiAgentWrapper, DictObservationsWrapper]:
    """Make the environment.

    Parameters
    ----------
    cfg :
        Any predefined arguments required to create the environment.
    env_config :
        Attribute dictionary with additional environment information.
    env_config: Optional[AttrDict] :


    Returns
    -------


    """
    env = create_env(cfg.env, cfg=cfg, env_config=env_config)
    if not is_multiagent_env(env):
        env = MultiAgentWrapper(env)
    if not isinstance(env.observation_space, spaces.Dict):
        env = DictObservationsWrapper(env)
    return env


class EnvCriticalError(Exception):
    """ """


def vizdoom_available():
    """ """
    return is_module_available("vizdoom")


def minigrid_available():
    """ """
    return is_module_available("gym_minigrid")


def dmlab_available():
    """ """
    return is_module_available("deepmind_lab")


def retry(exception_class=Exception, num_attempts=3, sleep_time=1):
    """

    Parameters
    ----------
    exception_class :
         (Default value = Exception)
    num_attempts :
         (Default value = 3)
    sleep_time :
         (Default value = 1)

    Returns
    -------

    """

    def decorator(func):
        """

        Parameters
        ----------
        func :


        Returns
        -------

        """

        @wraps(func)
        def wrapper(*args, **kwargs):
            """

            Parameters
            ----------
            *args :

            **kwargs :


            Returns
            -------

            """
            for i in range(num_attempts):
                try:
                    return func(*args, **kwargs)
                except exception_class as e:
                    if i == num_attempts - 1:
                        raise
                    else:
                        log.error("Failed with error %r, trying again", e)
                        sleep(sleep_time)

        return wrapper

    return decorator


def find_wrapper_interface(env, interface_type):
    """Unwrap the env until we find the wrapper that implements interface_type.

    Parameters
    ----------
    env :

    interface_type :


    Returns
    -------

    """
    unwrapped = env.unwrapped
    while True:
        if isinstance(env, interface_type):
            return env
        elif env == unwrapped:
            return None  # unwrapped all the way and didn't find the interface
        else:
            env = env.env  # unwrap by one layer


class RewardShapingInterface:
    """ """

    def __init__(self):
        pass

    def get_default_reward_shaping(self):
        """Should return a dictionary of string:float key-value pairs defining
        the current reward shaping scheme."""
        raise NotImplementedError

    def get_current_reward_shaping(self, agent_idx: int):
        """

        Parameters
        ----------
        agent_idx: int :


        Returns
        -------

        """
        raise NotImplementedError

    def set_reward_shaping(self, reward_shaping: dict, agent_idx: int):
        """Sets the new reward shaping scheme.

        Parameters
        ----------
        reward_shaping :
            dictionary of string-float key-value pairs
        agent_idx :
            integer agent index (for multi-agent envs)
        reward_shaping: dict :

        agent_idx: int :


        Returns
        -------

        """
        raise NotImplementedError


def get_default_reward_shaping(env):
    """The current convention is that when the environment supports reward shaping, the env.unwrapped should contain
    a reference to the object implementing RewardShapingInterface.
    We use this object to get/set reward shaping schemes generated by PBT.

    Parameters
    ----------
    env :


    Returns
    -------

    """

    reward_shaping_interface = find_wrapper_interface(env, RewardShapingInterface)
    if reward_shaping_interface:
        return reward_shaping_interface.get_default_reward_shaping()

    return None


def set_reward_shaping(env, reward_shaping: dict, agent_idx: int):
    """

    Parameters
    ----------
    env :

    reward_shaping: dict :

    agent_idx: int :


    Returns
    -------

    """
    reward_shaping_interface = find_wrapper_interface(env, RewardShapingInterface)
    if reward_shaping_interface:
        reward_shaping_interface.set_reward_shaping(reward_shaping, agent_idx)


class TrainingInfoInterface:
    """ """

    def __init__(self):
        self.training_info = dict()

    def set_training_info(self, training_info):
        """Send the training information to the environment, i.e. number of
        training steps so far.

        Some environments rely on that i.e. to implement curricula.

        Parameters
        ----------
        training_info :
            dictionary containing information about the current training
            session. Guaranteed to contain 'approx_total_training_steps'
            (approx because it lags a bit behind due to multiprocess
            synchronization)

        Returns
        -------

        """
        self.training_info = training_info


def find_training_info_interface(env):
    """Unwrap the env until we find the wrapper that implements
    TrainingInfoInterface.

    Parameters
    ----------
    env :


    Returns
    -------

    """
    return find_wrapper_interface(env, TrainingInfoInterface)


def set_training_info(training_info_interface, approx_total_training_steps: int):
    """

    Parameters
    ----------
    training_info_interface :

    approx_total_training_steps: int :


    Returns
    -------

    """
    if training_info_interface:
        training_info_dict = dict(
            approx_total_training_steps=approx_total_training_steps
        )
        training_info_interface.set_training_info(training_info_dict)
