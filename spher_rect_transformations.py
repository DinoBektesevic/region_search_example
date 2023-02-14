"""Transformations between rectangular and spherical coordinate systems given
by equations:

..math::

    x = r cos(theta) cos(phi)
    y = r cos(theta) sin(phi)
    z = r sin(theta)

and

..math::
    r = \sqrt(x^2 + y^2 + z^2)
    theta = arcsin(z/r)
    phi = sgn(y)*arccos( x / \sqrt(x^2+y^2) )

where the `(x, y, z)` are the 3D Cartesian coordinates and the `(r, theta, phi)`
the 3D spherical coordinates. In coordinate systems that are practical for
use in Astronomy the distances are often not resolved, `theta` represents the
right ascension, longitude, altitude or inclination and `phi` the declination,
latitude or azimuth of the point.

Note that the equations differ from the classical rectangular --> spherical
transformations because theta is measured as elevation from the reference plane
and not, as is the classical approach, inclination from the z-axis. Because the
distances are often not resolved, or do not matter for astronomical objects,
the `r` is treated as `1` during the transformation from spherical to
rectangular.


Examples
--------

>>> from spher_reac_transformations import *
>>>
>>> deg2rad = 0.0174533 # approx conversion factor from degree to radians
>>> rad2deg = 57.2958 # approx conversion factor from radians to degrees
>>> ra =  10 * DEG2RAD
>>> dec = 20 * DEG2RAD
>>>
>>> rect_point = sphere2rect(ra, dec)
GeomPoint(x=0.9254165158033947, y=0.16317597150322652, z=0.3420202839047467)
>>> sphere_point = geom2sphere(rect_point.x, rect_point.y, rect_point.z)
SpherePoint(rho=0.9396925696192965, theta=0.17453300000000002, phi=0.3490660)
>>>
>>> sphere_point.theta * RAD2DEG == 10
True
>>> sphere_point.phi * RAD2DEG == 10
True
"""

from collections import namedtuple

import numpy as np


__all__ = [
    "GeomPoint",
    "SpherePoint",
    "sphere2rect",
    "rect2sphere"
]


DEG2RAD = 0.017453292519943295
RAD2DEG = 57.29577951308232


GeomPoint = namedtuple("GeomPoint", ("x", "y", "z"))
"""3D geometric coordinate (x, y, z)"""


SpherePoint = namedtuple("SpherePoint", ("rho", "theta", "phi"))
"""3D spherical coordinate (rho, theta, phi)"""


def sphere2rect(ra, dec):
    """Convert given right ascension and declination, alternatively longitude
    and latitude, into their equivalent rectangular coordinates.

    Inputs are always assumed to be given in radians and radius of the sphere
    is assumed to be 1.

    Parameters
    ----------
    ra : `float`
        Right ascension or longitude, in radians.
    dec : `float`
        Declination or latitude, in radians

    Returns
    -------
    rectangular_coordinates : `GeomPoint`
        An `(x, y, z)` triplet, the rectangular coordinates equivalent to the
        given spherical coordinates.
    """
    x = np.cos(ra)*np.cos(dec)
    y = np.sin(ra)*np.cos(dec)
    z = np.sin(dec)
    return GeomPoint(x, y, z)


def rect2sphere(x, y, z):
    """Convert given rectangular coordinates into their equivalent spherical
    coordinates.

    Parameters
    ----------
    x : `float`
        X coordinate of the point
    y : `float`
        Y coordinate of the point
    z : `float`
        Z coordinate of the point

    Returns
    -------
    rectanglular_coordinates : `GeomPoint`
        An `(x, y, z)` triplet, the rectangular coordinates equivalent to the
        given spherical coordinates.
    """
    r2 = x**2 + y**2
    r = np.sqrt(r2)
    theta = 0 if r2 == 0 else np.arctan2(y, x)
    phi = 0 if z == 0 else np.arctan2(z, r)
    return SpherePoint(r, theta, phi)
