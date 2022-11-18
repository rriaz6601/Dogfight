"""Parameters for all aircraft environments.."""

import argparse

import gym

from src.async_arch.envs.aircraft.agent_defs import AcclerationAgent, PositionAgent
from src.async_arch.envs.aircraft.aircraft_gym import AircraftEnv
from src.async_arch.envs.aircraft.wrappers import (
    FullStateAggregation,
    PolicyWrapper,
    StateAggregation,
)


def add_aircraft_env_args(env_name, parser: argparse.ArgumentParser):
    """ """
    parser.add_argument(
        "--episode_max_time_sec",
        default=300,
        type=int,
        help="Real world time of one episode",
    )

    parser.add_argument(
        "--agent_type",
        default="Position",
        type=str,
        help="What type of agent should use for both aircraft.",
    )

    parser.add_argument(
        "--jsbsim_freq",
        default=60,
        type=int,
        help="Physics calculations in the JSBSim simulation.",
    )

    parser.add_argument(
        "--agent_interaction_steps",
        default=6,
        type=int,
        help="Agent interacts with jsbsim simulation every n steps.",
    )

    parser.add_argument(
        "--aircraft_name",
        default="f16",
        type=str,
        help="""Aircraft for simulation, should use one of the available ones
        in the jsbsim aircraft directory.""",
    )

    parser.add_argument(
        "--hard_deck", default=1000, type=int, help="The minimum allowed altitude."
    )


def aircraft_override_defaults(env: gym.Env, parser: argparse.ArgumentParser):
    """RL parameters specific to Aircraft envs."""
    parser.set_defaults(
        encoder_type="mlp",
        encoder_subtype="mlp_mujoco",
        hidden_size=512,
        kl_loss_coeff=0.1,
    )


def make_aircraft_env(full_env_name, cfg=None, env_config=None):
    """Create the full aircraft environment.

    This includes wrapping the environment in appropriate wrappers.
    So I will first create the environment.

    """
    if cfg.agent_type == "Position":
        agent = PositionAgent
        state_wrapper = StateAggregation
    elif cfg.agent_type == "Acceleration":
        agent = AcclerationAgent
        state_wrapper = FullStateAggregation

    env = AircraftEnv(agent, agent, cfg)
    env = PolicyWrapper(env)
    env = state_wrapper(env)

    return env
