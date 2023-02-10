from collections import namedtuple

import numpy as np
import astropy.units as u

from sqlalchemy import create_engine, ForeignKey, select
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
    Session
)


__all__ = [
    "engine",
    "TanWcs",
    "GeometricCoords",
    "Session",
    "GeomPoint",
    "SpherePoint",
    "sphere2geom",
    "geom2sphere"
]


engine = create_engine("sqlite+pysqlite:///example.sqlite3")

# SQLAlch 2.0 seems to come with some bright and shiny new toys!
class Base(DeclarativeBase):
    pass


class TanWcs(Base):
    __tablename__= "tan_wcs"

    id: Mapped[int] = mapped_column(primary_key=True)

    # these are the basic values WCS stores
    # * reference pixel in img coordinates
    # * reference pixel in on sky-coordinates
    # * delta-change of sky coordinates, if you move 1 pixel up/down/left/right
    # good enough for approximate calculations of positions
    crpix1: Mapped[float]
    crpix2: Mapped[float]
    crval1: Mapped[float]
    crval2: Mapped[float]
    cd11: Mapped[float]
    cd12: Mapped[float]
    cd21: Mapped[float]
    cd22: Mapped[float]

    geomcoord: Mapped["GeometricCoords"] = relationship(back_populates="tanwcs")

    def __repr__(self):
        return f"TanWcs({self.id}, {self.crval1}, {self.crval2})"

    @classmethod
    def query_square_naive(cls, ra, dec, size=1):
        with Session(engine) as session:
            stmt = (
                select(cls.crval1, cls.crval2)
                .where(cls.crval1.between(ra-size/2, ra+size/2))
                .where(cls.crval2.between(dec-size/2, dec+size/2))
            )
            res = session.execute(stmt).all()
        return np.array(res)

    @classmethod
    def all(cls):
        with Session(engine) as session:
            res = session.execute(
                select(cls.crval1, cls.crval2)
            )
        return np.array(res.all())


GeomPoint = namedtuple("GeomPoint", ("x", "y", "z"))


SpherePoint = namedtuple("SpherePoint", ("rho", "theta", "phi"))


def sphere2geom(ra, dec):
    x = np.cos(ra)*np.cos(dec)
    y = np.sin(ra)*np.cos(dec)
    z = np.sin(dec)
    return GeomPoint(x, y, z)


def geom2sphere(x, y, z):
    r2 = x**2 + y**2
    theta = 0 if r2 == 0 else np.atan2(y, x)
    phi = 0 if z == 0 else np.atan2(z, np.sqrt(r2))
    return SpherePoint(rho, theta, phi)


class GeometricCoords(Base):
    __tablename__ = "geom_coords"

    # these are the extended values, calculated at ingestion
    # these are first order improvement on searching WCS pointing
    # data directly because these columns can be indexed and subselected
    # from more easily than the TanWcs
    id: Mapped[int] = mapped_column(primary_key=True)

    radius: Mapped[float]

    center_x: Mapped[float]
    center_y: Mapped[float]
    center_z: Mapped[float]

    corner_x: Mapped[float]
    corner_y: Mapped[float]
    corner_z: Mapped[float]

    tanwcs_id = mapped_column(ForeignKey("tan_wcs.id"))
    tanwcs: Mapped[TanWcs] = relationship(back_populates="geomcoord")

    def __repr__(self):
        return f"Geom({id}, {center_x:.2}, {center_y:.2}, {center_z:.2})"

    def get_sphere_representation(self):
        return (
            geom2sphere(self.center_x, self.center_y, self.center_z),
            geom2sphere(self.corner_x, self.corner_y, self.corner_z)
        )

    @classmethod
    def query_square_naive(cls, ra, dec, gsize=0.1):
        center = sphere2geom((ra*u.deg).to(u.rad), (dec*u.deg).to(u.rad))
        with Session(engine) as session:
            stmt = (
                select(TanWcs.crval1, TanWcs.crval2)
                .join(cls.tanwcs)
                .where(cls.center_x.between(center.x-gsize/2, center.x+gsize/2))
                .where(cls.center_y.between(center.y-gsize/2, center.y+gsize/2))
                .where(cls.center_z.between(center.z-gsize/2, center.z+gsize/2))
            )
            res = session.execute(stmt).all()
        return np.array(res)

    @classmethod
    def all(cls):
        with Session(engine) as session:
            res = session.execute(
                select(cls.center_x, cls.center_y, cls.center_z)
            )
        return np.array(res.all())


Base.metadata.create_all(engine)








# from sqlalchemy.types import Double, Integer, TypeDecorator
# and then promptly squashes my optimism by breaking what they
# used to be happy to do...
# https://github.com/sqlalchemy/sqlalchemy/issues/5167
# https://github.com/sqlalchemy/sqlalchemy/issues/3586
# https://github.com/sqlalchemy/sqlalchemy/issues/5552
# https://github.com/sqlalchemy/sqlalchemy/discussions/5948
# https://docs.sqlalchemy.org/en/13/faq/thirdparty.html#i-m-getting-errors-related-to-numpy-int64-numpy-bool-etc
#    class GenFloat(TypeDecorator):
#        # coerces given data to float() before storing
#        # to avoid explosions due to np.int types
#        impl = Double
#        cache_ok = True
#
#        def process_bind_param(self, value, dialect):
#            return float(value)
#
#    class GenInteger(TypeDecorator):
#        # coerces given data to float() before storing
#        # to avoid explosions due to np.int types
#        impl = Integer
#        cache_ok = True
#
#        def process_bind_param(self, value, dialect):
#            return int(value)
# won't use this atm because it's just faster to cast everything
# than to rewrite the tables. TODO: later.
