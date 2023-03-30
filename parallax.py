from astropy.coordinates import ICRS, GCRS
import astropy.units as u

import numpy as np

from scipy.optimize import minimize

import matplotlib.pyplot as plt


__all__ = [
    "correct_parallax",
    "correct_parallax1",
    "correct_parallax2",
    "correct_parallax3",
    "corect_parallax4"
]


def correct_parallax1(icrs, obstime, point_on_earth, guess_distance):
    """Given an ICRS coordinate returns an ICRS coordinate corrected for the
    parallax that would occur if the point was at a given guess distance from
    the Sun.

    This is achieved by setting Given_ICRS ~ True_Object_ICRS LOS distance
    down the given ICRS ~ true distance of the object from the barycenter.

    Parameters
    ----------
    icrs : `~astropy.coordinates.ICRS`
        Given ICRS coordinate (the coordinate as reported by header)
    obstime : `~astropy.time.Time`
        Time of observation
    point_on_earth : `~astropy.coordinates.EarthLocation`
        Location from which the observation was made from, observatory location
    guess_distance : `float`
        Guess distance at which we suspect our object is, in AU.

    Returns
    -------
    corrected : `~astropy.coordinates.ICRS`
        ICRS coordinate corrected for the parallax.
    """
    loca = (
        point_on_earth.x.to(u.m).value,
        point_on_earth.y.to(u.m).value,
        point_on_earth.z.to(u.m).value
    )*u.m

    # calculate the unit-vector of the object as if it was observed from earth
    gcrs_earth_obj = icrs.transform_to(
        GCRS(
            obstime=obstime,
            obsgeoloc=loca
    ))

    # add a guess distace
    icrs_dist = ICRS(
        ra=icrs.ra,
        dec=icrs.dec,
        distance=guess_distance*u.AU
    )

    # Compute geocentric distance to the object
    sun_obj = icrs_dist.cartesian
    sun_obs = loc.get_gcrs(obstime).transform_to(ICRS())
    earth_obj = sun_obj - sun_obs.cartesian
    earth_dist = earth_obj.norm()

    # construct our original observation from Earth, using unit-vector of
    # reported ICRS, calculated earth-obj distance, observing time and location
    geo_nolie = GCRS(
        gcrs_earth_obj.ra,
        gcrs_earth_obj.dec,
        distance=earth_dist,
        obstime=obstime,
        obsgeoloc=loca
    )

    # return that as an ICRS coordinate corrected for parallax
    return geo_nolie.transform_to(ICRS())



def correct_parallax2(icrs, point_in_sol, obstime, point_on_earth, septhreshold=0.5):
    """Given an ICRS coordinate, as reported by the header, and a point in
    Solar System, an ICRS coordinate with a specified distance, returns whether
    the given ICRS coordinate contains the given point in solar system within
    a given radius around it, corrected for parallax.

    This is achieved by converting the given ICRS coordinate to Earth's center
    of mass, converting the point in solar system to Earth's center of mass,
    and comparing the angular separation between the two.

    Moving the given point on earth produces a LOS that includes the parallax
    so comparing that LOS with the LOS of the given ICRS coordinate, when
    translated to the Earth as origin, tells us whether our given ICRS
    coordinate sees the object or not - parallax corrected.

    Parameters
    ----------
    icrs : `~astropy.coordinates.ICRS`
        Given ICRS coordinate (the coordinate as reported by header).
    point_in_sol : `astropy.coordinates.ICRS`
        A 3D point in solar system, an ICRS coordinate with a distance.
    obstime : `~astropy.time.Time`
        Time of observation.
    point_on_earth : `~astropy.coordinates.EarthLocation`
        Location from which the observation was made from, observatory location
    septhreshold : `float`
        Angular distance, radius, outside of which the two coordinates are said
        not to be observing the same point in the sky. In arcseconds.

    Returns
    -------
    observe_the_same_point : `bool`
        The coordinates observe the same point when True, and not when False.
    angularsep : `~astropy.coordinates.angles.Angle`
        Angular separation between the given ICRS and ``point_in_sol``, once
        corrected for parallax.
    """
    loca = (
        point_on_earth.x.to(u.m).value,
        point_on_earth.y.to(u.m).value,
        point_on_earth.z.to(u.m).value
    )*u.m

    # unit-vector of the line-of-sigth through the object from earth
    gcrs_earth_obj = icrs.transform_to(
        GCRS(
            obstime=obstime,
            obsgeoloc=loca
        )
    )

    # we want to get all observations that observed this point in the solar
    # system this is how it's seen from Earth at the given time stamp
    gcrs_earth_obj_dist = point_in_sol.transform_to(
        GCRS(
            obstime=obstime,
            obsgeoloc=loca
        )
    )

    angsep = gcrs_earth_obj.separation(gcrs_earth_obj_dist)

    # return that as an ICRS coordinate corrected for parallax
    return angsep<septhreshold*u.arcsecond, angsep


def correct_parallax3(icrs, obstime, point_on_earth, guess_distance, dstep=0.001):
    """Given an ICRS coordinate returns an ICRS coordinate corrected for
    the parallax that would occur if the point was at a given guess distance
    from the Sun.

    This is achieved by moving the given ICRS to Earth as origin, sampling a
    series of distances down that line of sight and then converting them to
    ICRS. The converted ICRS coordinate whose distance most closely matches the
    guess distance is returned as the result.

    Parameters
    ----------
    icrs : `~astropy.coordinates.ICRS`
        Given ICRS coordinate (the coordinate as reported by header)
    obstime : `~astropy.time.Time`
        Time of observation
    point_on_earth : `~astropy.coordinates.EarthLocation`
        Location from which the observation was made from, observatory location
    guess_distance : `float`
        Guess distance at which we suspect our object is, in AU.
    dstep : `float`
        Step between the neighbouring guess distancesm in AU.

    Returns
    -------
    corrected : `~astropy.coordinates.ICRS`
        ICRS coordinate corrected for the parallax.
    """
    loc = (
        point_on_earth.x.to(u.m).value,
        point_on_earth.y.to(u.m).value,
        point_on_earth.z.to(u.m).value
    )*u.m

    # line of sight from earth to the object,
    # the object has an unknown distance from earth
    los_earth_obj = icrs.transform_to(
        GCRS(
            obstime=obstime,
            obsgeoloc=loc
        )
    )

    guess_dists = np.arange(guess_distance-1, guess_distance+1, 0.001)
    guesses = GCRS(
        ra=los_earth_obj.ra,
        dec=los_earth_obj.dec,
        distance=guess_dists*u.AU,
        obstime=obstime,
        obsgeoloc=loc
    ).transform_to(ICRS())

    deltad = np.abs(guess_distance-guesses.distance.value)
    minidx= min(deltad) == deltad
    answer = guesses[minidx]

    # we'll make a new object so that it returns numbers not a list
    res = ICRS(
        ra = answer.ra[0],
        dec = answer.dec[0],
        distance=answer.distance[0]
    )

    return res


def correct_parallax4(icrs, obstime, point_on_earth, guess_distance):
    """Given an ICRS coordinate returns an ICRS coordinate corrected for the
    parallax that would occur if the point was at a given guess distance from
    the Sun.

    This is achieved by moving the given ICRS to Earth as origin, sampling a
    series of distances down that line of sight and then converting them to
    ICRS. The converted ICRS coordinate whose distance most closely matches the
    guess distance is returned as the result.

    Same as `correct_parallax3` except that the sampling is performed by
    `~scipy.optimize.minimize` making it (hopefully) faster and more precise.

    Parameters
    ----------
    icrs : `~astropy.coordinates.ICRS`
        Given ICRS coordinate (the coordinate as reported by header)
    obstime : `~astropy.time.Time`
        Time of observation
    point_on_earth : `~astropy.coordinates.EarthLocation`
        Location from which the observation was made from, observatory location
    guess_distance : `float`
        Guess distance at which we suspect our object is, in AU.

    Returns
    -------
    corrected : `~astropy.coordinates.ICRS`
        ICRS coordinate corrected for the parallax.
    """
    loc = (
        point_on_earth.x.to(u.m).value,
        point_on_earth.y.to(u.m).value,
        point_on_earth.z.to(u.m).value
    )*u.m

    # line of sight from earth to the object,
    # the object has an unknown distance from earth
    los_earth_obj = icrs.transform_to(
        GCRS(
            obstime=obstime,
            obsgeoloc=loc
        )
    )

    cost = lambda d: np.abs(guess_distance - GCRS(
        ra=los_earth_obj.ra,
        dec=los_earth_obj.dec,
        distance=d*u.AU,
        obstime=obstime,
        obsgeoloc=loc
    ).transform_to(ICRS()).distance.to(u.AU).value)

    bounds = (guess_distance-2, guess_distance+2)
    fit = minimize(
        cost,
        (guess_distance, ),
        #tol=1e11,
        bounds=(bounds,),
    )

    answer = GCRS(
        ra=los_earth_obj.ra,
        dec=los_earth_obj.dec,
        distance=fit.x[0]*u.AU,
        obstime=obstime,
        obsgeoloc=loc
    ).transform_to(ICRS())

    return answer


# we just rename the one we trust the most as "the" correct_parallax
correct_parallax = correct_parallax3
