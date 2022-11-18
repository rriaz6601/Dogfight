"""
Draw a circle around point lla:
1. Use this reference point as the centre of the circle
2. Choose a radius and parametric equation of the circle
3. Calculate the next point in 3D space
4. Convert them to lla and visualise them in X-Plane
5. What about pitch, roll and heading?

For double circle fight manoeuvre:
if in range 2pi to 4pi translate the centre 2*radius.

But the problem right now is that what about the speeds?
1. I know going around a circle is constant speed and all speeds are body frame so
2. Roll-rate and yaw-rate are zero, pitch rate is how fast its going around the circle
3. u, v, w might be zero as well because no changes in body frame?
4. The motion is in x-z plane meaning v=0; and u = d/dt(cost), w = d/dt(sint) (Just make sure v is not vertical velocity)

For straight line motion
"""

import numpy as np


def next_point_circle(t, radius: int):
    """

    Parameters
    ----------
    t :

    radius: int :


    Returns
    -------

    """
    x = radius * np.sin(t)
    y = 0
    z = radius * np.cos(t)

    return [x, y, z]


def next_point_cw_circle(t, radius: int):
    """

    Parameters
    ----------
    t :

    radius: int :


    Returns
    -------

    """
    x = -radius * np.sin(t)
    y = 0
    z = radius * np.cos(t)

    return [x, y, z]
