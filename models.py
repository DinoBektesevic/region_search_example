import numpy as np

import astropy.units as u
from astropy.time import Time
from astropy.coordinates import (
    ICRS,
    GeocentricTrueEcliptic,
    HeliocentricTrueEcliptic,
    angular_separation
)

from sqlalchemy.types import TypeDecorator
from sqlalchemy import (
    create_engine,
    ForeignKey,
    select
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
    Session,
    sessionmaker
)

from spher_rect_transformations import (
    sphere2rect,
    rect2sphere,
    DEG2RAD,
    RAD2DEG
)
from type_decorators import NpFloat, NpInt


__all__ = [
    "engine",
    "session",
    "TanWcs",
    "RectangularCoords",
    "RectangularHeliocentricCoords",
]


engine = create_engine("sqlite+pysqlite:///example.sqlite3")
"""An Engine object providing DB connectivity and behavior functionality."""


session = sessionmaker(engine)
"""An Session factory providing transactional functionality."""


class Base(DeclarativeBase):
    """Declarative Base used to keep track of DB's metadata."""
    pass


class TanWcs(Base):
    """
    A table representing the simplified tangential plane projection (TAN) of an
    Wold Coordinate System (WCS) of an exposure.

    The recorded values are sufficient to perform the linear transformations to
    intermediate pixel coordinates, as described by Greisen and Calabretta in
    their 1st paper. The assumptions that the values are given in decimal
    degree format, in a Earth-centered Ecliptic Coordinate System, with no
    additional non-linear transformations are required to complete the
    transformation to the final world coordinates.
    """
    __tablename__= "tan_wcs"

    id: Mapped[int] = mapped_column(NpInt, primary_key=True)
    """Unique auto-incremented ID of the WCS."""

    # these are the basic values WCS stores
    # * reference pixel in img coordinates
    # * reference pixel in on sky-coordinates
    # * delta-change of sky coordinates, if you move 1 pixel up/down/left/right
    # good enough for approximate calculations of positions
    crpix1: Mapped[float] = mapped_column(NpFloat)
    """Center reference pixel's x-coordinate, in image coordinates."""
    crpix2: Mapped[float] = mapped_column(NpFloat)
    """Center reference pixel's y-coordinate, in image coordinates."""

    crval1: Mapped[float] = mapped_column(NpFloat)
    """Center reference pixel's right ascension, in sky coordinates."""
    crval2: Mapped[float] = mapped_column(NpFloat)
    """Center reference pixel's declination, in sky coordinates."""

    cd11: Mapped[float] = mapped_column(NpFloat)
    """Element (1, 1) of the affine transformation matrix."""
    cd12: Mapped[float] = mapped_column(NpFloat)
    """Element (1, 2) of the affine transformation matrix."""
    cd21: Mapped[float] = mapped_column(NpFloat)
    """Element (2, 1) of the affine transformation matrix."""
    cd22: Mapped[float] = mapped_column(NpFloat)
    """Element (2, 2) of the affine transformation matrix."""

    mjd: Mapped[float] = mapped_column(NpFloat)
    """MJD of observation"""

    rectcoord: Mapped["RectangularCoords"] = relationship(back_populates="tanwcs")
    """Related rectangular coordinates associated with the reference, and
    corner pixel."""
    rectheliocoord: Mapped["RectangularHeliocentricCoords"] = relationship(back_populates="tanwcs")
    """Related rectangular heliocentric coordinates associated with the
    reference, and corner pixel."""

    def __repr__(self):
        return f"TanWcs({self.id}, {self.crval1}, {self.crval2})"

    @classmethod
    def query_square_naive(cls, ra, dec, size=1):
        """Returns all records with coordinates within a square of given size
        around the given center `(ra, dec)` coordinates.

        Note that this selection is rather naive as it's performed as:
        ```
        crval1 BETWEEN ra-size/2 AND ra+size/2
        crval2 BETWEEN dec-size/2 AND dec+size/2
        ```
        with no explicit wrapping or spherical distance calculations.

        Parameters
        ----------
        ra : `float`
            Right ascension of the center of the square, in degrees
        dec : `float`
            Declination of the center of the square, in degrees
        size : `float`, optional
            Size of the square box around the center coordinates, in degrees.
            Defaults to a square 1x1 degree in size.

        Returns
        -------
        records : `numpy.array`
            An array of `(ra, dec)` pairs within the requested square box.
        """
        with session.begin() as transaction:
            stmt = (
                select(cls.crval1, cls.crval2)
                .where(cls.crval1.between(ra-size/2, ra+size/2))
                .where(cls.crval2.between(dec-size/2, dec+size/2))
            )
            res = transaction.execute(stmt).all()
        return np.array(res)

    @classmethod
    def all(cls, withTime=False):
        """Returns an array of `(ra, dec)` pairs of all records in the table."""
        cols = (cls.crval1, cls.crval2)
        if withTime:
            cols = (cls.crval1, cls.crval2, cls.mjd)

        with session.begin() as transaction:
            res = transaction.execute(
                select(*cols)
            )
        return np.array(res.all())


class RectangularCoords(Base):
    """
    A table representing the rectangular coordinate representations associated
    with the `tan_wcs` rows. The rectangular coordinates represented the
    reference pixel, the top-left corner pixel and the radius records the
    Euclidean distance between the two.
    """
    __tablename__ = "rect_coords"

    # these are the extended values, calculated at ingestion
    # these are first order improvement on searching WCS pointing
    # data directly because these columns can be indexed and subselected
    # from more easily than the TanWcs
    id: Mapped[int] = mapped_column(NpInt, primary_key=True)
    """Unique auto-incremented ID of the associate rectangular coordinates."""

    radius: Mapped[float] = mapped_column(NpFloat)
    """Euclidean distance between the reference pixel and top left corner."""

    refpix_x: Mapped[float] = mapped_column(NpFloat)
    """X coordinate of the reference pixel."""
    refpix_y: Mapped[float] = mapped_column(NpFloat)
    """Y coordinate of the reference pixel."""
    refpix_z: Mapped[float] = mapped_column(NpFloat)
    """Z coordinate of the reference pixel."""

    corner_x: Mapped[float] = mapped_column(NpFloat)
    """X coordinate of the top-left corner pixel."""
    corner_y: Mapped[float] = mapped_column(NpFloat)
    """X coordinate of the top-left corner pixel."""
    corner_z: Mapped[float] = mapped_column(NpFloat)
    """X coordinate of the top-left corner pixel."""

    tanwcs_id = mapped_column(NpInt, ForeignKey("tan_wcs.id"))
    """The ID of the associated TanWCS."""
    tanwcs: Mapped[TanWcs] = relationship(back_populates="rectcoord")
    """The associated TanWCS object."""

    def __repr__(self):
        return f"RectCoord({id}, {refpix_x:.2}, {refpix_y:.2}, {refpix_z:.2})"

    def get_spherical_coords(self):
        """Returns the spherical representation of the reference and corner
        pixels.
        """
        return (
            rect2sphere(self.refpix_x, self.refpix_y, self.refpix_z),
            rect2sphere(self.corner_x, self.corner_y, self.corner_z)
        )

    @classmethod
    def query_square(cls, ra, dec, size=1):
        """Returns all records with coordinates within a square of given size
        around the given center `(ra, dec)` coordinates.

        Note that this selection is nor rather naive as its TanWCS counterpart
        as the conversion to rectangular coordinates, of the given coordinates
        occurs before querying, So, while the query resembles the TanWCS one:
        ```
        x BETWEEN x-size/2 AND x+size/2
        y BETWEEN y-size/2 AND y+size/2
        z BETWEEN z-size/2 AND z+size/2
        ```
        the end result does in fact include wrapping and correct distance
        calculation between points.

        Parameters
        ----------
        ra : `float`
            Right ascension of the center of the square, in degrees
        dec : `float`
            Declination of the center of the square, in degrees
        size : `float`, optional
            Size of the square box around the center coordinates, in degrees.
            Defaults to a square 1x1 degree in size.

        Returns
        -------
        records : `numpy.array`
            An array of `(ra, dec)` pairs within the requested square box.
        """
        center = sphere2rect(ra*DEG2RAD, dec*DEG2RAD)
        # we do mins and maxes here to avoid getting trapped in one
        # of the poles by accident, almost positive there's a better way
        edges = np.array([
            sphere2rect((ra+size)*DEG2RAD, dec*DEG2RAD),
            sphere2rect((ra-size)*DEG2RAD, dec*DEG2RAD),
            sphere2rect(ra*DEG2RAD, (dec+size)*DEG2RAD),
            sphere2rect(ra*DEG2RAD, (dec-size)*DEG2RAD)
        ])
        dists = (center.x-edges[:,0])**2 + (center.y-edges[:,1])**2 + (center.z-edges[:,2])**2
        d = dists.max()

        with session.begin() as transaction:
            stmt = (
                select(TanWcs.crval1, TanWcs.crval2)
                .join(cls.tanwcs)
                .where(
                    (cls.refpix_x - center.x)*(cls.refpix_x - center.x) +
                    (cls.refpix_y - center.y)*(cls.refpix_y - center.y) +
                    (cls.refpix_z - center.z)*(cls.refpix_z - center.z) <= d)
            )
            res = transaction.execute(stmt).all()

        return np.array(res)

    @classmethod
    def all(cls):
        """Returns an array of `(x, y, z)` triplets of all of the recorded
        reference pixels.
        """
        with session.begin() as transaction:
            res = transaction.execute(
                select(cls.refpix_x, cls.refpix_y, cls.refpix_z)
            )
        return np.array(res.all())


class RectangularHeliocentricCoords(Base):
    """
    A table representing the rectangular coordinate representation of the
    associated `tan_wcs` rows expressed in heliocentric coordinates at a
    distance of 50AU as they would-be observed through the year 2023. The

    Rectangular coordinates apply only to the reference pixel for purposes of
    an example.
    """
    __tablename__ = "rect_helio"

    # these are the extended values, calculated at ingestion
    # these are first order improvement on searching WCS pointing
    # data directly because these columns can be indexed and subselected
    # from more easily than the TanWcs
    id: Mapped[int] = mapped_column(NpInt, primary_key=True)
    """Unique auto-incremented ID of the associate rectangular coordinates."""

    refpix_x: Mapped[float] = mapped_column(NpFloat)
    """X coordinate of the reference pixel."""
    refpix_y: Mapped[float] = mapped_column(NpFloat)
    """Y coordinate of the reference pixel."""
    refpix_z: Mapped[float] = mapped_column(NpFloat)
    """Z coordinate of the reference pixel."""

    tanwcs_id = mapped_column(NpInt, ForeignKey("tan_wcs.id"))
    """The ID of the associated TanWCS."""
    tanwcs: Mapped[TanWcs] = relationship(back_populates="rectheliocoord")
    """The associated TanWCS object."""

    def __repr__(self):
        return f"RectHelioCoord({id}, {refpix_x:.2}, {refpix_y:.2}, {refpix_z:.2})"

    def get_spherical_coords(self):
        """Returns the spherical representation of the reference pixel."""
        return rect2sphere(self.refpix_x, self.refpix_y, self.refpix_z)

    @classmethod
    def query_helio_square(cls, lon, lat, size=1):
        """Returns all records with coordinates within a square of given size
        around the given center `(lon, lat, d=50AU)` coordinates.

        Note that this selection is nor rather naive as its TanWCS counterpart
        as the conversion of the given coordinates to the rectangular
        heliocentric coordinates occurs before querying,
        So, while the query resembles the TanWCS one:
        ```
        x BETWEEN x-size/2 AND x+size/2
        y BETWEEN y-size/2 AND y+size/2
        z BETWEEN z-size/2 AND z+size/2
        ```
        the end result does in fact include wrapping and parallax of the
        observed frame centers at the pre-calculated distances.

        Parameters
        ----------
        lon : `float`
            Longitude, in decimal degrees.
        lat : `float`
            Latitude, in decimal degrees
        size : `float`, optional
            Size of the square box around the center coordinates, in degrees.
            Defaults to a square 1x1 degree in size.

        Returns
        -------
        records : `numpy.array`
            An array of `(lon, lat)` pairs within the requested square box.
        """
        center = sphere2rect(lon*DEG2RAD, lat*DEG2RAD)
        # we do mins and maxes here to avoid getting tlonpped in one
        # of the poles by accident, almost positive there's a better way
        edges = np.array([
            sphere2rect((lon+size)*DEG2RAD, lat*DEG2RAD),
            sphere2rect((lon-size)*DEG2RAD, lat*DEG2RAD),
            sphere2rect(lon*DEG2RAD, (lat+size)*DEG2RAD),
            sphere2rect(lon*DEG2RAD, (lat-size)*DEG2RAD)
        ])
        dists = (center.x-edges[:,0])**2 + (center.y-edges[:,1])**2 + (center.z-edges[:,2])**2
        d = dists.max()

        with session.begin() as transaction:
            stmt = (
                select(TanWcs.crval1, TanWcs.crval2)
                .join(cls.tanwcs)
                .where(
                    (cls.refpix_x - center.x)*(cls.refpix_x - center.x) +
                    (cls.refpix_y - center.y)*(cls.refpix_y - center.y) +
                    (cls.refpix_z - center.z)*(cls.refpix_z - center.z) <= d)
            )
            res = transaction.execute(stmt).all()
        return np.array(res)

    @classmethod
    def query_square_dist2(cls, lon, lat, distance, size=1):
        t = Time(59945.0, format="mjd", scale="utc")
        icrs = HeliocentricTrueEcliptic(
            lon=lon*u.deg,
            lat=lat*u.deg,
            distance=distance*u.AU
        ).transform_to((GeocentricTrueEcliptic(obstime=t)))

        return RectangularCoords.query_square(icrs.lon.value, icrs.lat.value, size=size)


    @classmethod
    def query_square_dist(cls, ra, dec, distance, size=1):
        """Returns all records with coordinates within a square of given size
        around the given center and distance `(lon, lat, d=distanceAU)` coordinates,
        **Assumes all observations were taken at the same time!**

        Note that this selection is nor rather naive as its TanWCS counterpart
        as the conversion of the given coordinates to the rectangular
        heliocentric coordinates occurs before querying,
        So, while the query resembles the TanWCS one:
        ```
        x BETWEEN x-size/2 AND x+size/2
        y BETWEEN y-size/2 AND y+size/2
        z BETWEEN z-size/2 AND z+size/2
        ```
        the end result does in fact include wrapping and parallax of the
        observed frame centers at the pre-calculated distances.

        Parameters
        ----------
        lon : `float`
            Longitude, in decimal degrees.
        lat : `float`
            Latitude, in decimal degrees
        size : `float`, optional
            Size of the square box around the center coordinates, in degrees.
            Defaults to a square 1x1 degree in size.

        Returns
        -------
        records : `numpy.array`
            An array of `(lon, lat)` pairs within the requested square box.
        """
        #t = Time(59945.0, format="mjd", scale="utc")
        res = TanWcs.all(withTime=True)
        racat, deccat, t = res[:, 0], res[:, 1], res[:, 2]
        t = Time(t, format="mjd", scale="utc")

        # remove the origin shift of ICRS assumption and calculates the Geo
        # coords as seen at the time of observation
        refpix_icrs = ICRS(ra=racat*u.deg, dec=deccat*u.deg)
        refpix_geo = refpix_icrs.transform_to(GeocentricTrueEcliptic(
            obstime=t
        ))

        # Add an assumed distance to the observed position
        refpix_nolie = GeocentricTrueEcliptic(
            refpix_geo.lon,
            refpix_geo.lat,
            distance=distance,
            obstime=t
        )

        # Transform back to ICRS, now including parallax due to distance
        # not just origin shift
        icrs = refpix_nolie.transform_to(ICRS())

        dists = angular_separation(ra*u.deg, dec*u.deg, icrs.ra, icrs.dec)
        resmask = dists <= size*u.deg

        return np.array([racat[resmask], deccat[resmask]])

    @classmethod
    def all(cls):
        """Returns an array of `(x, y, z)` triplets of all of the recorded
        reference pixels.
        """
        with session.begin() as transaction:
            res = transaction.execute(
                select(cls.refpix_x, cls.refpix_y, cls.refpix_z)
           )
        return np.array(res.all())
