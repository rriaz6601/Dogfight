import re
from os import environ

import jsbsim

from .catalogs.catalog import Catalog
from .catalogs.property import CustomProperty, Property


class Simulation:
    """This class wraps an instance of JSBSim and manages communication with it.

    Parameters
    ----------
    aircraft_name:

    init_conditions:

    jsbsim_freq:

    agent_interaction_steps:

    Attributes
    ----------

    Methods
    -------
    run:
        Tell the simulation to take a step further

    set_property_values:
        Set the value of some sim parameters usually to pass actions

    get_property_values:
        Get the value of observation space variables

    close:
        Close the simulation

    """

    def __init__(
        self,
        aircraft_name="F16",
        init_conditions=None,
        jsbsim_freq=60,
        agent_interaction_steps=5,
    ):
        """
        Creates an instance of JSBSim, loads aircraft and sets initial conditions
        """

        self.jsbsim_exec = jsbsim.FGFDMExec(environ["JSBSIM_ROOT_DIR"])
        self.jsbsim_exec.set_debug_level(
            0
        )  # requests JSBSim not to output any messages whatsoever

        self.jsbsim_exec.load_model(aircraft_name)

        # Collect all jsbsim properties in Catalog
        Catalog.add_jsbsim_props(self.jsbsim_exec.get_property_catalog())

        # set jsbsim integration time step
        dt = 1 / jsbsim_freq
        self.jsbsim_exec.set_dt(dt)

        self.agent_interaction_steps = agent_interaction_steps

        self._initialise(init_conditions)

    def _initialise(self, init_conditions):
        """

        Parameters
        ----------
        init_conditions :


        Returns
        -------

        """
        self._set_initial_conditions(init_conditions)
        success = self.jsbsim_exec.run_ic()
        self._propulsion_init_running(-1)
        if not success:
            raise RuntimeError("JSBSim failed to initialise simulation conditions")

    def _propulsion_init_running(self, i):
        """

        Parameters
        ----------
        i :


        Returns
        -------

        """
        propulsion = self.jsbsim_exec.get_propulsion()
        n = propulsion.get_num_engines()
        if i >= 0:
            if i >= n:
                raise IndexError("Tried to initialise a non-existent engine")
            propulsion.get_engine(i).init_running()
            propulsion.get_steady_state()
        else:
            for j in range(n):
                propulsion.get_engine(j).init_running()
            propulsion.get_steady_state()

    def _set_initial_conditions(self, init_conditions):
        """Loads initial conditions into JSBSim

        Parameters
        ----------
        init_conditions :


        Returns
        -------

        """

        if init_conditions is not None:
            for prop, value in init_conditions.items():
                self.set_property_value(prop, value)

    def run(self):
        """Runs JSBSim simulation until the agent interacts and update custom properties.

        Parameters
        ----------

        Returns
        -------

        """
        for _ in range(self.agent_interaction_steps):
            result = self.jsbsim_exec.run()
            if not result:
                raise RuntimeError("JSBSim failed")
        return result

    def get_sim_time(self):
        """Gets the simulation time from JSBSim, a float"""
        return self.jsbsim_exec.get_sim_time()

    def close(self):
        """Closes the simulation and any plots."""

        if self.jsbsim_exec:
            self.jsbsim_exec = None

    def get_property_values(self, props):
        """Gets the values of specified properties

        Parameters
        ----------
        props :
            list of properties

        Returns
        -------
        type
            NamedTuple with properties' names and their values

        """
        return [self.get_property_value(prop) for prop in props]

    def set_property_values(self, props, values):
        """Set the value of specified properties

        Parameters
        ----------
        props :
            list of properties
        values :
            list of values

        Returns
        -------

        """
        if not len(props) == len(values):
            raise ValueError("mismatch between properties and values size")
        for prop, value in zip(props, values):
            self.set_property_value(prop, value)

    def get_property_value(self, prop):
        """Get the value of specified property from the JSBSim simulation

        Parameters
        ----------
        prop :
            The name of the desired property

        Returns
        -------
        type
            float value

        """
        if isinstance(prop, Property):
            if prop.access == "R":
                if prop.update:
                    prop.update(self)
            return self.jsbsim_exec.get_property_value(prop.name_jsbsim)
        elif isinstance(prop, CustomProperty):
            if "R" in prop.access and prop.read:
                return prop.read(self)
            else:
                raise RuntimeError(f"{prop} is not readable")
        else:
            raise ValueError(f"prop type unhandled: {type(prop)} ({prop})")

    def set_property_value(self, prop, value):
        """Set the value of the specified property

        Parameters
        ----------
        prop :
            Property
        value :
            A float value

        Returns
        -------

        """
        if isinstance(prop, Property):
            if value < prop.min:
                value = prop.min
            elif value > prop.max:
                value = prop.max

            self.jsbsim_exec.set_property_value(prop.name_jsbsim, value)

            if "W" in prop.access:
                if prop.update:
                    prop.update(self)
        elif isinstance(prop, CustomProperty):
            if "W" in prop.access and prop.write:
                return prop.write(self, value)
            else:
                raise RuntimeError(f"{prop} is not readable")

        else:
            raise ValueError(f"prop type unhandled: {type(prop)} ({prop})")

    def get_sim_state(self):
        """Get the full simulation state.

        It has not been called anywhere in the code, but can be useful to
        analyse the aircraft's behaviour.

        """
        return {prop: self.get_property_value(prop) for prop in Catalog.values()}

    def state_to_ic(self, state):
        """Save the current state as initial conditions for a future run.

        Parameters
        ----------
        state :


        Returns
        -------

        """
        init_conditions = {}

        state_to_ic = {
            Catalog.position_lat_gc_deg: Catalog.ic_lat_gc_deg,
            Catalog.position_long_gc_deg: Catalog.ic_long_gc_deg,
            Catalog.position_h_sl_ft: Catalog.ic_h_sl_ft,
            Catalog.position_h_agl_ft: Catalog.ic_h_agl_ft,
            Catalog.position_terrain_elevation_asl_ft: Catalog.ic_terrain_elevation_ft,
            Catalog.attitude_psi_deg: Catalog.ic_psi_true_deg,
            Catalog.attitude_theta_deg: Catalog.ic_theta_deg,
            Catalog.attitude_phi_deg: Catalog.ic_phi_deg,
            Catalog.velocities_u_fps: Catalog.ic_u_fps,
            Catalog.velocities_v_fps: Catalog.ic_v_fps,
            Catalog.velocities_w_fps: Catalog.ic_w_fps,
            Catalog.velocities_p_rad_sec: Catalog.ic_p_rad_sec,
            Catalog.velocities_q_rad_sec: Catalog.ic_q_rad_sec,
            Catalog.velocities_r_rad_sec: Catalog.ic_r_rad_sec,
        }

        for prop, value in state.items():
            if not re.match(r"^ic/", prop.name_jsbsim):
                if prop in state_to_ic:
                    init_conditions[state_to_ic[prop]] = value
                elif "RW" in prop.access:
                    init_conditions[prop] = value
        return init_conditions

    def set_sim_state(self, state):
        """

        Parameters
        ----------
        state :


        Returns
        -------

        """
        init_conditions = self.state_to_ic(state)
        self.jsbsim_exec.reset_to_initial_conditions(0)
        self._initialise(init_conditions)
