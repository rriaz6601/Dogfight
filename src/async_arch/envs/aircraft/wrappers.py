"""This module contains main wrappers for making my environment more modular.

Wrapping sequence is unwrapped -> policy -> reward -> state
The policy edits actions before stepping into the environment.
The reward and observation edit after the environment has finished stepping

"""

import numpy as np
from gym import ActionWrapper, ObservationWrapper, RewardWrapper
from gym.spaces import Box
from navpy import angle2dcm

from src.async_arch.envs.aircraft.aircraft_utils import calculate_relative_position


class PolicyWrapper(ActionWrapper):
    """Provide a framework for choosing any policy for target agent."""

    def action(self, action):

        act = {}
        act["aggressor"] = action
        act["target"] = None
        return act


class RewardShapingWrapper(RewardWrapper):
    """Provide a framework for calculating any sort of rewards."""

    def reward(self, reward):
        """More reward shaping.

        Potential function for relative distance, and track angle.
        Need to think more about the track angle
        """
        rel_position = calculate_relative_position(
            self._state["aggressor"], self._state["target"]
        )

        rel_dist = np.linalg.norm(rel_position)

        # this is array of angles about all three axis
        track_angle = np.arcsin(rel_position / rel_dist)

        reward -= max(rel_dist - 300, 0) / self._max_rel_dist
        reward -= np.linalg.norm(track_angle)

        return reward


class StateAggregation(ObservationWrapper):
    """Aggregate the state of two agents into one for learner."""

    def observation(self, observation):

        # TODO: I may need to normalise the observations
        obs_deltas = observation["aggressor"].copy()
        obs_deltas[:3] -= observation["target"][:3]

        assert len(obs_deltas) == 7

        return obs_deltas


class FullStateAggregation(ObservationWrapper):
    """Doing a rich state information and also normalising."""

    def __init__(self, env):
        super().__init__(env)

        self.observation_space = Box(low=-1.0, high=1.0, shape=(22,))
        self._aggressor = env._aggressor
        self._target = env._target

    def observation(self, observation):

        # Relative position and velocity
        ned_rel_pos = calculate_relative_position(
            observation["target"], observation["aggressor"]
        )
        target_rot_mat = angle2dcm(
            observation["target"][5],
            observation["target"][3],
            observation["target"][4],
            input_unit="deg",
        )
        target_vel_ned = np.dot(target_rot_mat, observation["target"][6:9])

        aggressor_rot_mat = angle2dcm(
            observation["aggressor"][5],
            observation["aggressor"][3],
            observation["aggressor"][4],
            input_unit="deg",
        )
        aggressor_vel_ned = np.dot(aggressor_rot_mat, observation["aggressor"][6:9])
        rel_velo = aggressor_vel_ned - target_vel_ned - ned_rel_pos

        obs_deltas = []

        obs_deltas[:3] = ned_rel_pos[:]  # Relative position
        obs_deltas[3:6] = rel_velo[:]  # relative velocity
        obs_deltas[6:9] = observation["aggressor"][3:6]  # own euler angles
        obs_deltas[9:18] = observation["aggressor"][
            9:18
        ]  # own rot vel, and 6 accelerations
        obs_deltas[18:21] = observation["aggressor"][
            23:26
        ]  # own alpha and beta and opp health
        obs_deltas.append(1.0)  #  my health

        assert len(obs_deltas) == 22

        # Normalising the observations
        # Subtract the middle then divide by (high - low) /2
        # TODO: There should be a better way to do this normalisation
        obs_deltas = np.array(obs_deltas)
        middle = np.array(
            [
                1500,
                1500,
                1500,
                700,
                700,
                700,
                0,
                0,
                180,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0.5,
                0.5,
            ]
        )
        difference = np.array(
            [
                1500,
                1500,
                1500,
                700,
                700,
                700,
                90,
                90,
                180,
                2 * np.pi,
                2 * np.pi,
                2 * np.pi,
                4,
                4,
                4,
                (8 / 180) * np.pi,
                (8 / 180) * np.pi,
                (8 / 180) * np.pi,
                90,
                180,
                0.5,
                0.5,
            ]
        )

        obs_deltas -= middle
        normalised_obs = np.divide(obs_deltas, difference)

        return normalised_obs
