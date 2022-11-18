"""Include the commonly used agents."""

from dataclasses import dataclass
from typing import Any, Dict, List

from src.async_arch.envs.aircraft.catalogs.catalog import Catalog as c
from src.async_arch.envs.aircraft.catalogs.property import Property


@dataclass
class BasicAgent:
    """These classes define common types of agents.

    Example agent the inheritance doesn't really work in dataclasses.

    Parameters
    ----------
    action_var:
        The action space of the agent.
    state_var:
        The state space of the agent.
    initial_conditions:
        Where to start the simulation from.
    physical:
        True represents that this agent will be simulated using physcial simulator.
    role:
        True means it is the aggressor, and false means it is the target.

    """

    physical: bool
    role: bool
    action_var: List[Property]
    state_var: List[Property]
    initial_conditions: Dict[Property, Any]

    # def __post_init__(self):
    #     """Using this method to edit initial conditions if required.

    #     Algorithm:
    #     1. Check if its a rules based agent.
    #     2. If true then edit the initial latitude, long and alt (LLA).
    #     3. Constraint LLA in a range.
    #         i.   Distance of 100m
    #         ii.  Within the target.
    #         iii. What speed should I set
    #         iv.
    #     4. Also set should it be in the target.

    #     """
    #     if self.role == False:
    #         initial_lla = target_at_x(
    #             [
    #                 self.initial_conditions[c.ic_lat_geod_deg],
    #                 self.initial_conditions[c.ic_lat_geod_deg],
    #                 self.initial_conditions[c.ic_lat_geod_deg],
    #                 0.0,
    #                 0.0,
    #                 0.0,
    #             ],
    #             [100, 0, 0],
    #         )
    #         self.initial_conditions[c.ic_lat_geod_deg] = initial_lla[0]
    #         self.initial_conditions[c.ic_long_gc_deg] = initial_lla[0]
    #         self.initial_conditions[c.h_sl_ft] = initial_lla[0]


@dataclass
class PointAgent:
    """This is the simplest agent."""

    physical: bool
    role: bool
    action_var = [
        c.fcs_aileron_cmd_norm,
        c.fcs_elevator_cmd_norm,
        c.fcs_rudder_cmd_norm,
        c.fcs_throttle_cmd_norm,
    ]
    state_var = [
        c.position_lat_geod_deg,
        c.position_long_gc_deg,
        c.position_h_sl_ft,
        c.health,
    ]
    initial_conditions = {
        c.ic_h_sl_ft: 10000,
        c.ic_long_gc_deg: 1.5501,
        c.ic_lat_geod_deg: 43.69879,
        c.ic_u_fps: 800,
        c.ic_v_fps: 0,
        c.ic_w_fps: 0,
        c.ic_p_rad_sec: 0,
        c.ic_q_rad_sec: 0,
        c.ic_r_rad_sec: 0,
        c.health: 1.0,
    }


@dataclass
class PositionAgent:
    physical: bool
    role: bool
    action_var = [
        c.fcs_aileron_cmd_norm,
        c.fcs_elevator_cmd_norm,
        c.fcs_rudder_cmd_norm,
        c.fcs_throttle_cmd_norm,
    ]
    state_var = [
        c.position_lat_geod_deg,
        c.position_long_gc_deg,
        c.position_h_sl_ft,
        c.attitude_theta_deg,
        c.attitude_phi_deg,
        c.attitude_psi_deg,
        c.health,
    ]
    initial_conditions = {
        c.ic_h_sl_ft: 10000,
        c.ic_terrain_elevation_ft: 0,
        c.ic_long_gc_deg: 1.5501,
        c.ic_lat_geod_deg: 43.697,
        c.ic_theta_deg: 0.0,
        c.ic_u_fps: 800,
        c.ic_v_fps: 0,
        c.ic_w_fps: 0,
        c.ic_p_rad_sec: 0,
        c.ic_q_rad_sec: 0,
        c.ic_r_rad_sec: 0,
        c.ic_roc_fpm: 0,
        c.fcs_throttle_cmd_norm: 0.8,
        c.fcs_mixture_cmd_norm: 1,
        c.gear_gear_pos_norm: 0,
        c.gear_gear_cmd_norm: 0,
        c.health: 1.0,
    }


@dataclass
class AcclerationAgent:
    physical: bool = True
    role: bool = True
    action_var = [
        c.fcs_aileron_cmd_norm,
        c.fcs_elevator_cmd_norm,
        c.fcs_rudder_cmd_norm,
        c.fcs_throttle_cmd_norm,
    ]
    state_var = [
        c.position_lat_geod_deg,
        c.position_long_gc_deg,
        c.position_h_sl_ft,
        c.attitude_theta_deg,
        c.attitude_phi_deg,
        c.attitude_psi_deg,
        c.velocities_u_fps,
        c.velocities_v_fps,
        c.velocities_w_fps,
        c.velocities_p_rad_sec,
        c.velocities_q_rad_sec,
        c.velocities_r_rad_sec,
        c.accelerations_vdot_ft_sec2,
        c.accelerations_wdot_ft_sec2,
        c.accelerations_udot_ft_sec2,
        c.accelerations_pdot_rad_sec2,
        c.accelerations_qdot_rad_sec2,
        c.accelerations_rdot_rad_sec2,
        c.propulsion_total_fuel_lbs,
        c.fcs_elevator_pos_norm,
        c.fcs_rudder_pos_norm,
        c.fcs_left_aileron_pos_norm,
        c.fcs_right_aileron_pos_norm,
        c.aero_alpha_deg,
        c.aero_beta_deg,
        c.health,
    ]
    initial_conditions = {
        c.ic_h_sl_ft: 10000,
        c.ic_terrain_elevation_ft: 0,
        c.ic_long_gc_deg: 1.5501,
        c.ic_lat_geod_deg: 43.697,
        c.ic_theta_deg: 0.0,
        c.ic_u_fps: 800,
        c.ic_v_fps: 0,
        c.ic_w_fps: 0,
        c.ic_p_rad_sec: 0,
        c.ic_q_rad_sec: 0,
        c.ic_r_rad_sec: 0,
        c.ic_roc_fpm: 0,
        c.fcs_throttle_cmd_norm: 0.8,
        c.fcs_mixture_cmd_norm: 1,
        c.gear_gear_pos_norm: 0,
        c.gear_gear_cmd_norm: 0,
        c.health: 1.0,
    }


@dataclass
class AcclerationAgent2:
    physical: bool = True
    role: bool = True
    action_var = [
        c.fcs_aileron_cmd_norm,
        c.fcs_elevator_cmd_norm,
        c.fcs_rudder_cmd_norm,
        c.fcs_throttle_cmd_norm,
    ]
    state_var = [
        c.position_lat_geod_deg,
        c.position_long_gc_deg,
        c.position_h_sl_ft,
        c.attitude_theta_deg,
        c.attitude_phi_deg,
        c.attitude_psi_deg,
        c.velocities_u_fps,
        c.velocities_v_fps,
        c.velocities_w_fps,
        c.velocities_p_rad_sec,
        c.velocities_q_rad_sec,
        c.velocities_r_rad_sec,
        c.accelerations_vdot_ft_sec2,
        c.accelerations_wdot_ft_sec2,
        c.accelerations_udot_ft_sec2,
        c.accelerations_pdot_rad_sec2,
        c.accelerations_qdot_rad_sec2,
        c.accelerations_rdot_rad_sec2,
        c.propulsion_total_fuel_lbs,
        c.fcs_elevator_pos_norm,
        c.fcs_rudder_pos_norm,
        c.fcs_left_aileron_pos_norm,
        c.fcs_right_aileron_pos_norm,
        c.aero_alpha_deg,
        c.aero_beta_deg,
        c.health,
    ]
    initial_conditions = {
        c.ic_h_sl_ft: 10000,
        c.ic_terrain_elevation_ft: 0,
        c.ic_long_gc_deg: 1.5501,
        c.ic_lat_geod_deg: 43.710,
        c.ic_theta_deg: 0.0,
        c.ic_psi_true_deg: 180.0,
        c.ic_u_fps: 600,
        c.ic_v_fps: 0,
        c.ic_w_fps: 0,
        c.ic_p_rad_sec: 0,
        c.ic_q_rad_sec: 0,
        c.ic_r_rad_sec: 0,
        c.ic_roc_fpm: 0,
        c.fcs_throttle_cmd_norm: 0.8,
        c.fcs_mixture_cmd_norm: 1,
        c.gear_gear_pos_norm: 0,
        c.gear_gear_cmd_norm: 0,
        c.health: 1.0,
    }
