"""
This environment has both rules based agents to validate common methods.

All the validation needs to be visualised in X-Plane to be sure.
The methods to validate:
  1. is_target_locked
  2. get_health
  3. reward functions values
"""

import gym
from navpy import ned2lla

from .tasks.dogfight_task import DogfightTask
from .utilities.xpc import XPlaneComms


class EnvBothRule(gym.Env):
    """The environment for validating the common methods."""

    def __init__(self):
        """Initialise the necessary components method."""
        self.xplane = XPlaneComms()
        self.xplane.pauseSim(True)
        self.task = DogfightTask()

        self.ref_vals = [43.697, 1.5501, 10000]
        self.steps_per_second = 5

    def reset(self):
        """Reset method."""
        # The timestep
        self.timestep = 0
        self.opp_health = 1.0

        ego_state = self.get_observation()

        return ego_state

    def step(self):
        """Step method."""
        state = self.get_observation()
        state_ego = state[0:6]
        state_opp = state[-13:-7]

        # X-Plane also expects landing gear position in its state vector
        state_ego.append(0.0)
        state_opp.append(0.0)

        self.xplane.sendPOSI(state_ego, 0)
        self.xplane.sendPOSI(state_opp, 1)

        reward = self.task.get_reward(state)
        done = (
            self.task.is_terminal(state) or self.timestep / self.steps_per_second >= 300
        )

        # Increment the simulation step
        self.timestep += 1

        return state, reward, done, {}

    def close(self):
        """Close method, must be implemented in a gym environment."""

    def render(self, mode="human"):
        """Render method, must be implemented in a gym environment.

        Parameters
        ----------
        mode :
             (Default value = 'human')

        Returns
        -------

        """

    def get_observation(self):
        """ """
        ego_ned = [self.timestep * 100, 0, -self.timestep * 17.5]
        opp_ned = [(self.timestep + 2) * 100, 0, -(self.timestep + 2) * 17.5]

        ego_lla = ned2lla(
            ego_ned, self.ref_vals[0], self.ref_vals[1], self.ref_vals[2], alt_unit="ft"
        )

        opp_lla = ned2lla(
            opp_ned, self.ref_vals[0], self.ref_vals[1], self.ref_vals[2], alt_unit="ft"
        )

        ego_obs = [
            ego_lla[0],
            ego_lla[1],
            ego_lla[2],
            10.0,
            0.0,
            0.0,
            100.0,
            0.0,
            -17.5,
            0.0,
            0.0,
            0.0,
            1.0,
            opp_lla[0],
            opp_lla[1],
            opp_lla[2],
            10.0,
            0.0,
            0.0,
            100.0,
            0.0,
            -17.5,
            0.0,
            0.0,
            0.0,
            self.opp_health,
        ]

        if self.timestep == 0:
            self.opp_health = 1.0
        else:
            self.opp_health = self.task.get_opp_health(ego_obs, self.steps_per_second)
        ego_obs[-1] = self.opp_health

        assert len(ego_obs) == 26

        return ego_obs
