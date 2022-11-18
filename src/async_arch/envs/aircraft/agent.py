"""The link between the environment and simulation.

I want to abstract if the agents are running a physical simulation or just one
formula with no physical constraints.
"""

from typing import Dict, List

import numpy as np
from gym.spaces import Box
from navpy import angle2dcm, ned2lla

from src.async_arch.envs.aircraft.simulation import Simulation


class Agent:
    """Agents live in environment and abstract the back-end simulation.

    This is meant to replace the functionality of task and thus the associated
    tasks folder as well.

    Parameters
    ----------
    physical:
        Tells if the back-end needs to be a JSBSim or a simpler equation.
    initial_conditions:

    Methods
    -------
    apply_action:

    get_observation:

    apply_initial_conditions:


    Attributes
    ----------


    """

    def __init__(self, agent) -> None:
        self._action_var = agent.action_var
        self._state_var = agent.state_var
        self._physical = agent.physical
        self._init_conditions = agent.initial_conditions

        self.max_deviation: float = 10.0

    def initialise_simulation(self, aircraft, freq, interact_n):
        """Apply these provided initial conditions to the agent."""
        if self._physical:
            self._simulation = Simulation(
                aircraft_name=aircraft,
                init_conditions=self._init_conditions,
                jsbsim_freq=freq,
                agent_interaction_steps=interact_n,
            )
        else:
            self._simulation = NonPhysicalSimulation(
                self._action_var,
                self._state_var,
                init_conditions=self._init_conditions,
                steps_per_second=freq / interact_n,
                max_dev=self.max_deviation,
            )

    def apply_action(self, action: np.ndarray):
        """Apply this action to the simulation"""
        if action is not None:
            self._simulation.set_property_values(self._action_var, action)
        self._simulation.run()

    def get_state(self) -> np.ndarray:
        """Return the observation vector from the simulation."""

        return np.array(self._simulation.get_property_values(self._state_var))

    def get_observation_space(self) -> Box:
        low = []
        high = []
        for prop in self._state_var:
            low.append(prop.min)
            high.append(prop.max)
        return Box(low=np.array(low), high=np.array(high), dtype=np.float32)

    def get_action_space(self) -> Box:
        low = []
        high = []
        for prop in self._action_var:
            low.append(prop.min)
            high.append(prop.max)
        return Box(low=np.array(low), high=np.array(high), dtype=np.float32)


class NonPhysicalSimulation:
    """Simulate a non-physical agent.

    This implements the same interface as the the Jsbsim Simulation. Looking at
    the Simulation class, we can see it sets action using
    set_property_values(action_var, action), then calls the environment to step
    using run() and finally gets the state using get_property_values(state_var).

    Let's start with setting the same.
    So it expects action_var and state_var of the type List[Property].

    If I am to implement these four methods, what else do I need to do.

    Parameters
    ----------

    """

    def __init__(
        self,
        action_var,
        state_var,
        init_conditions: Dict,
        steps_per_second: float,
        max_dev: float,
    ) -> None:
        self.max_deviation = max_dev

        self._steps_per_second = steps_per_second
        self._action_var = action_var
        self.state_var = state_var
        self._timestep = 0
        self._initial_lla: List = None
        self._state = None
        self._speed = 250 / steps_per_second  # Constant speed in m/s
        self._heading: float = 0.0
        self._theta: float = 0.0

        self._initialise(init_conditions)
        self.run()

    def set_property_values(self, props, values):
        """Implement any deviation from forward motion."""

    def get_property_values(self, props):
        """The method to read values for a simulation."""
        return self._state

    def run(self):
        """Calculate the next state."""

        ned_vals = self._timestep * self._velocity_ned
        self._state = list(
            ned2lla(
                ned_vals,
                self._initial_lla[0],
                self._initial_lla[1],
                self._initial_lla[2],
                alt_unit="ft",
            )
        )
        self._state.extend([self._theta * 57.3, 0.0, self._heading * 57.3])  # angles
        self._state.extend([self._speed * 3.28, 0.0, 0.0])  # linear velocities in ft/s
        self._state.append(1.0)  # appending health

        self._timestep += 1

    def close(self):
        pass

    def _initialise(self, init_conditions):
        """Handle any initialisation."""
        self._heading = self.max_deviation * (2 * np.random.random() - 1) / 57.3
        self._theta = self.max_deviation * (2 * np.random.random() - 1) / 57.3
        self._initial_lla = [43.710, 1.5501, 10000]

        # Velocity components remain constant in each episode
        rot_mat = angle2dcm(self._heading, self._theta, 0.0, input_unit="rad")
        self._velocity_ned = np.dot(rot_mat, [self._speed, 0, 0])
