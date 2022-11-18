"""The default gym based aircraft environment."""

import time
from collections import deque
from typing import Deque, Dict, Union

import gym
import numpy as np
from navpy import ned2lla

from src.async_arch.envs.aircraft.agent import Agent
from src.async_arch.envs.aircraft.agent_defs import AcclerationAgent, PositionAgent
from src.async_arch.envs.aircraft.aircraft_utils import (
    calculate_relative_position,
    is_locked,
)
from src.async_arch.envs.aircraft.catalogs.catalog import Catalog as c
from src.async_arch.envs.aircraft.xpc import XPlaneComms


class AircraftEnv(gym.Env):
    """This is an environment for training aircraft agent against others.

    This environment does not need to know the type of the agents. The
    aggressor is learning using RL. The target is following some policy to move
    in front of this agent. It is a gym.Env so will implement only those
    required attributes and methods.

    The spaces are implemented as dictionaries here. But what the algorithm
    expects as is let's say box. Where does it ask for? Through the wrappers,
    so it should be fine as long as I am using wrappers correctly.

    Parameters
    ----------
    cfg:

    Attributes
    ----------
    action_space: gym.spaces.Dict{'aggressor': gym.spaces.Box,
                                  'target': gym.spaces.Box}

    observation_space: gym.spaces.Dict{'aggressor': gym.spaces.Box,
                                  'target': gym.spaces.Box}

    reward_range: Tuple (float, float)

    Methods
    -------
    reset, step, render, close, seed (See gym.Env for details.)

    """

    def __init__(
        self,
        aggressor: Union[PositionAgent, AcclerationAgent],
        target: Union[PositionAgent, AcclerationAgent],
        cfg,
    ) -> None:

        self._aggressor = Agent(aggressor)
        self._target = Agent(target)

        self.action_space = self._aggressor.get_action_space()
        self.observation_space = self._aggressor.get_observation_space()
        self.reward_range = (-17, 17)

        self._steps_per_second = cfg.jsbsim_freq / cfg.agent_interaction_steps
        self._floor_alt = cfg.hard_deck
        self._max_timesteps = cfg.episode_max_time_sec * self._steps_per_second
        self._max_rel_dist = 10000

        self.aircraft_name = cfg.aircraft_name
        self.jsbsim_freq = cfg.jsbsim_freq
        self.agent_interaction_steps = cfg.agent_interaction_steps

        self._timestep = 0
        self._state = {}
        self._episode_rew = 0.0
        self._start_time = time.time()

        # For setting a curriculum based environment
        self._past_eps_rewards: Deque[float] = deque([], maxlen=5)

        # Simulations are unable to set health themselves so we calculate here
        # and persist these variables in every episode
        self._target_health = 1.0

        try:
            if not cfg.no_render:
                self._xplane = XPlaneComms()
                self._xplane.pauseSim(True)
        except AttributeError:
            pass

    def step(self, action: Dict):
        """The step function.

        The reward is handled by a wrapper.

        Parameters
        ----------
        action :

        Returns
        -------

        """

        self._aggressor.apply_action(action["aggressor"])
        self._target.apply_action(None)

        self._state = {
            "aggressor": self._aggressor.get_state(),
            "target": self._target.get_state(),
        }

        # self._state["target"][-1] = self._target_health

        self._update_health()

        done = self._is_terminal()
        self._timestep += 1

        reward = self._get_reward()
        self._episode_rew += reward

        return self._state, reward, done, {}

    def reset(self):
        """

        Parameters
        ----------

        Returns
        -------

        """
        self._timestep = 0
        self._target_health = 1.0
        self._start_time = time.time()

        dist = (1500 * np.random.rand()) + 1500
        speed = (100 * np.random.rand()) + 300
        alt = (9000 * np.random.rand()) + 6000

        self._aggressor._init_conditions[c.ic_u_fps] = speed * 1.68781
        self._aggressor._init_conditions[c.ic_h_sl_ft] = alt

        self._aggressor.initialise_simulation(
            self.aircraft_name, self.jsbsim_freq, self.agent_interaction_steps
        )

        self._target._init_conditions[c.ic_u_fps] = speed * 1.68781
        self._target._init_conditions[c.ic_h_sl_ft] = alt
        lla_target = ned2lla([0, dist, 0], 43.697, 1.5501, alt, alt_unit="ft")
        self._target._init_conditions[c.ic_lat_geod_deg] = lla_target[0]
        self._target._init_conditions[c.ic_long_gc_deg] = lla_target[1]

        self._target.initialise_simulation(
            self.aircraft_name, self.jsbsim_freq, self.agent_interaction_steps
        )

        self._target._simulation.jsbsim_exec.do_trim(0)

        self._state = {
            "aggressor": self._aggressor.get_state(),
            "target": self._target.get_state(),
        }

        self._episode_rew = 0.0

        # print(f"Reset finished at: {time.time() - self._start_time}")
        print(f"start: {self._start_time}")

        return self._state

    def close(self):
        """ """

    def render(self, mode="human"):
        """Render the environment using X-Plane 11.

        Need an initiated XPlane client.

        Parameters
        ----------
        mode :
             (Default value = "human")

        Returns
        -------

        """

        aircraft_agg = list(self._state["aggressor"][0:6])
        aircraft_agg.append(0.0)  # X-Plane also expects landing gear

        aircraft_target = list(self._state["target"][0:6])
        aircraft_target.append(0.0)

        # conversion to metres
        aircraft_agg[2] /= 3.28
        aircraft_target[2] /= 3.28

        # # transform points to decrease distance to view
        # rel_pos = calculate_relative_position(aircraft_agg, aircraft_target)
        # rel_pos /= 10
        # new_lla_target = ned2lla(
        #     rel_pos,
        #     aircraft_agg[0],
        #     aircraft_agg[1],
        #     aircraft_agg[2],
        # )
        # aircraft_agg[:3] = new_lla_target[:]

        self._xplane.sendPOSI(aircraft_agg, 0)
        self._xplane.sendPOSI(aircraft_target, 1)

    def _get_reward(self):
        """Get the default reward.

        damage per second is calculated by how much time an agent stays in
        weapons engagement zone and then reward is area under the damage curve.

        Returns
        -------

        """

        reward = -0.001

        rel_pos = calculate_relative_position(
            self._state["aggressor"], self._state["target"]
        )
        if is_locked(rel_pos):
            reward += 0.5

        # Checking for own crash
        if self._state["aggressor"][2] <= self._floor_alt:
            reward += 0.003 * self._timestep - 14

        # Checking for opponent health
        if self._target_health <= 0.0:
            reward += 10

        rel_dist = np.linalg.norm(rel_pos)

        # this is array of angles about all three axis
        track_angle = np.arcsin(rel_pos / rel_dist)

        reward -= 0.001 * max(float(rel_dist - 3000), 0) / self._max_rel_dist
        reward -= 0.001 * np.linalg.norm(track_angle)

        return reward

    def _is_terminal(self) -> bool:
        """Check if in terminal state.

        The terminal states are:
            An agent wins if:
                1. Opponent agent reaches zero health
                2. opponent dips below hard deck of 1000ft
            It's a draw if simulation reaches 300s (checked in main loop)

        Returns
        -------

        """
        terminal = (
            self._state["aggressor"][-1] <= 0
            or self._state["target"][-1] <= 0
            or self._state["aggressor"][2] <= self._floor_alt
            or self._state["target"][2] <= self._floor_alt
            or time.time() - self._start_time >= 300
        )

        if terminal:
            print(time.time() - self._start_time)

        return terminal

    def _update_health(self):
        """Calculate the health of both agents and update the state.

        The agent actually knows about only opponent's health currently,
        because first I am trying to only learn aggressive posture.

        TODO: implement the aggressor_health when needed
        """

        # should calculate from nose not cg
        rel_pos = calculate_relative_position(
            self._state["aggressor"], self._state["target"]
        )
        print(rel_pos)

        if is_locked(rel_pos):
            dps = (np.linalg.norm(rel_pos) - 500.0) / 3000.0
            damage = dps / self._steps_per_second
            self._target_health -= damage

        self._state["aggressor"][-1] = self._target_health
