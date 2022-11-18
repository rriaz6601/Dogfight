"""Similar interface to sample factory doom environment."""

from typing import List, Union

import numpy as np
from navpy import angle2dcm, lla2ned, ned2lla

# def make_aircraft_env(env_name, cfg, **kwargs):
#     aircraft_spec = aircraft_env_by_name(env_name)
#     env = gym.make(aircraft_spec.env_id)
#     return env


# TODO: what to do with this formatting
# class AircraftSpec:
#     """Defining an aircraft environment.

#     Parameters
#     ----------

#     """

#     def __init__(
#         self,
#         name,
#         env_spec_file,
#         action_space,
#         reward_scaling=1.0,
#         default_timeout=-1,
#         num_agents=1,
#         timelimit=4.0,
#         extra_wrappers: List[Tuple] = None,
#     ) -> None:
#         self.name = name
#         self.env_spec_file = env_spec_file
#         self.action_space = action_space
#         self.reward_scaling = reward_scaling
#         self.default_timeout = default_timeout

#         # 1 for single-player, >1 otherwise
#         self.num_agents = num_agents

#         self.timelimit = timelimit

#         # expect list of tuples (wrapper_cls, wrapper_kwargs)
#         self.extra_wrappers = extra_wrappers


def target_at_x(coords: Union[List, np.ndarray], opp_relative: Union[List, np.ndarray]):
    """Function returns coordinates in target at a given distance.

    Parameters
    ----------
    coords:
        The coordinates of the ego aircraft.
        They are an array-like object containing [lat, long, alt, theta, phi, psi]

    """

    my_euler_angles = coords[5], coords[3], coords[4]
    my_lla = coords[0], coords[1], coords[2]

    rot_mat = angle2dcm(
        my_euler_angles[0], my_euler_angles[1], my_euler_angles[2], input_unit="deg"
    )

    opp_coords_in_my_ned = np.dot(rot_mat.T, opp_relative)

    opp_coord = ned2lla(
        opp_coords_in_my_ned, my_lla[0], my_lla[1], my_lla[2], alt_unit="ft"
    )

    return opp_coord


def is_locked(opp_relative):
    """Checks if the opponent is locked i-e in a 2 degree angle cone.

    Radius of the 2 degree circle at the distance x is given by.
    rad = x * (tan 2)

    """

    rad_at_x = opp_relative[0] * 0.03492
    dist = np.sqrt(opp_relative[0] ** 2 + opp_relative[1] ** 2 + opp_relative[2] ** 2)
    if 500 < dist < 3000:
        return (opp_relative[1] ** 2 + opp_relative[2] ** 2) < rad_at_x**2
    else:
        return False


def calculate_relative_position(coordsA, coordsB):
    """Calculates the relative position of B in local frame of A."""

    rot_mat = angle2dcm(coordsA[5], coordsA[3], coordsA[4], input_unit="deg")
    B_in_A_ned = lla2ned(
        coordsB[0],
        coordsB[1],
        coordsB[2],
        coordsA[0],
        coordsA[1],
        coordsA[2],
        alt_unit="ft",
    )

    B_in_A_body = np.dot(rot_mat, B_in_A_ned)

    return B_in_A_body
