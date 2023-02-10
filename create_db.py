import os
import astropy.units as u
from astropy.wcs import WCS
from astropy.io import fits

import numpy as np

from models import (
    engine,
    TanWcs,
    GeometricCoords,
    Session,
    sphere2geom,
    geom2sphere,
)


#if os.path.exists("example.sqlite3"):
#  os.remove("example.sqlite3")


NFRAMES = 10000

# here we have to fake a bunch (NFRAMES) of WCSs
# this has to be done with some amount of attention to details
# because validation will be difficult if not
header_dat = {
    "NAXIS": 2,
    "CTYPE1": "RA---TAN",
    "CTYPE2": "DEC--TAN",
    "CRPIX1": 500.0,
    "CRPIX2": 500.0,
    "CD1_1": -0.000721190155326047,
    "CD1_2": 0.000258453338150409,
    "CD2_1": 0.000258874499780883,
    "CD2_2": 0.000720016854665084,
}

idxs = np.arange(1, NFRAMES+1, dtype=int)
ras = np.random.uniform(0, 360.0, size=NFRAMES)
decs = np.random.uniform(-90.0, 90.0, size=NFRAMES)


# We need the corner coordinates, so we fake minimum required
# information to calculate those consistently
hdu = fits.PrimaryHDU()
for k, v in header_dat.items():
    hdu.header[k] = v


tanwcss, geomcoords = [], []
for idx, ra, dec in zip(idxs, ras, decs):
    # create our TanWCS objects for the DB
    # ingestion, assign the idx to speed up
    # ingestion into DB later
    tanwcss.append(TanWcs(
        id=int(idx), # I'm gonna lose my mind
        crpix1 = float(header_dat["CRPIX1"]),
        crpix2 = float(header_dat["CRPIX2"]),
        crval1 = float(ra),
        crval2 = float(dec),
        cd11 =   float(header_dat["CD1_1"]),
        cd12 =   float(header_dat["CD1_2"]),
        cd21 =   float(header_dat["CD2_1"]),
        cd22 =   float(header_dat["CD2_2"]),
    ))

    # time to fake semi-realistic Geometric Coords
    # first we complete the header, get a WCS through
    # which we calculate sky coordinates of image corners
    # and then calculate their geometric equivalents
    hdu.header["CRVAL1"] = ra
    hdu.header["CRVAL2"] = dec
    wcs = WCS(hdu)

    ra_rad, dec_rad = (ra*u.degree).to(u.rad), (dec*u.degree).to(u.rad)
    center =  sphere2geom(ra_rad, dec_rad)

    corner_sky = wcs.wcs_pix2world([ra, ], [dec, ], 1)
    corner_sky_rad = (corner_sky*u.degree).to(u.rad)
    corner = sphere2geom(*corner_sky_rad)

    radius = np.sqrt((corner.x-center.x)**2 + (corner.y-center.y)**2 + (corner.z-center.z)**2)
    geomcoords.append(GeometricCoords(
        id=int(idx),
        center_x = float(center.x),
        center_y = float(center.y),
        center_z = float(center.z),
        corner_x = float(corner.x),
        corner_y = float(corner.y),
        corner_z = float(corner.z),
        radius =   float(radius),
        tanwcs_id = int(idx),
        tanwcs =   tanwcss[-1],
    ))


# insert the data into the DB
with Session(engine) as session:
    session.bulk_save_objects(tanwcss)
    session.bulk_save_objects(geomcoords)
    session.commit()
