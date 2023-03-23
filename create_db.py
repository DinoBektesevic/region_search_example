import os
import time
import logging

import numpy as np
import astropy.units as u
from astropy.wcs import WCS
from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import (
    ICRS,
    HeliocentricTrueEcliptic,
    GeocentricTrueEcliptic
)

from models import (
    session,
    TanWcs,
    RectangularCoords,
    RectangularHeliocentricCoords,
    Base,
    engine
)
from spher_rect_transformations import (
    sphere2rect,
    rect2sphere,
    DEG2RAD,
    RAD2DEG
)


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)


###############################################################################
#                                    CONFIG
###############################################################################

# Sets the number of frames that will be faked and ingested
NFRAMES = 30000
DISTANCE = 50*u.AU


###############################################################################
#                               ADVANCED CONFIG
###############################################################################
# these settings are modifiable but only if you know what they docs. We are
# faking a bunch of WCSs. This has to be done with some amount of attention to
# details because validation will be difficult and results won't make sense.
chunksize = int(NFRAMES/2) if NFRAMES <= 10000 else 10000

nchunks = int(np.rint(NFRAMES/chunksize))
nchunks = 1 if nchunks==0 else nchunks


def uniform_sphere_sample(
        n,
        thetalim=[-np.pi, np.pi],
        philim=[-np.pi/2, np.pi/2]
):
    theta = 2*np.pi*np.random.uniform(0, 1, size=n)
    phi = np.arcsin(2*np.random.uniform(0, 1, size=n) - 1)
    return theta*RAD2DEG, phi*RAD2DEG

idxs = np.arange(1, NFRAMES+1, dtype=int)
mjds = np.random.uniform(59945.0, 60307.0, size=NFRAMES)
# uniform sphere
ras = 2*np.pi * np.random.uniform(0, 1, size=NFRAMES) * RAD2DEG
decs = np.arcsin(2*np.random.uniform(0, 1, size=NFRAMES) - 1) * RAD2DEG
# uniform flat
#ras = np.random.uniform(0, 360.0, size=NFRAMES)
#decs = np.random.uniform(-90.0, 90.0, size=NFRAMES)


hdu = fits.PrimaryHDU()
header = hdu.header
header["NAXIS"] = 2
header["CTYPE1"] = "RA---TAN"
header["CTYPE2"] = "DEC--TAN"
header["CRPIX1"] = 500.0
header["CRPIX2"] = 500.0
header["CD1_1"] = -0.000721190155326047
header["CD1_2"] = 0.000258453338150409
header["CD2_1"] = 0.000258874499780883
header["CD2_2"] = 0.000720016854665084

logging.info(f"Creating example.sqlite3 DB, "
             f"containing {NFRAMES} faked pointings ranging from "
             f"{Time(mjds[0], format='mjd', scale='utc').isot} to "
             f"{Time(mjds[-1], format='mjd', scale='utc').isot}")


###############################################################################
#                                FAKING DATA
###############################################################################

def chunkify(iterable, chunk_size):
    """Yields successive `chunk_size`d chunks from `iterable`."""
    for i in range(0, len(iterable), chunk_size):
        yield iterable[i:i+chunk_size]

# cleanup old DB
if os.path.exists("example.sqlite3"):
    os.remove("example.sqlite3")
    logging.info(f"Found old DB, removed it.")
    Base.metadata.create_all(engine)

logging.info("Faking pointings....")
st = time.time()

# create the lists of related TanWCS, RectangularCoords and
# RectangularHeliocentricCoords to ingest into DB.
tanwcss, rectcoords, rectheliocoords = [], [], []
for idx, ra, dec, mjd in zip(idxs, ras, decs, mjds):
    # create our TanWCS, we assign the idx to speed up ingestion into DB
    tanwcss.append(TanWcs(
        id = idx,
        crpix1 = header["CRPIX1"],
        crpix2 = header["CRPIX2"],
        crval1 = ra,
        crval2 = dec,
        cd11 = header["CD1_1"],
        cd12 = header["CD1_2"],
        cd21 = header["CD2_1"],
        cd22 = header["CD2_2"],
        mjd = mjd
    ))

    # Construct the WCS, calc corner, convert to rect, create and append obj
    # This section is basically what's required for RectangularCoords
    hdu.header["CRVAL1"] = ra
    hdu.header["CRVAL2"] = dec
    wcs = WCS(hdu)

    ra_rad, dec_rad = ra*DEG2RAD, dec*DEG2RAD
    refpix =  sphere2rect(ra_rad, dec_rad)

    corner_sky = wcs.wcs_pix2world([ra, ], [dec, ], 1)
    corner_sky_rad = (corner_sky*u.degree).to(u.rad)
    corner = sphere2rect(*corner_sky_rad)

    radius = np.sqrt(
        (corner.x-refpix.x)**2 +
        (corner.y-refpix.y)**2 +
        (corner.z-refpix.z)**2
    )

    rectcoords.append(RectangularCoords(
        id = idx,
        refpix_x = refpix.x,
        refpix_y = refpix.y,
        refpix_z = refpix.z,
        corner_x = corner.x,
        corner_y = corner.y,
        corner_z = corner.z,
        radius = radius,
        tanwcs_id = idx,
        tanwcs = tanwcss[-1],
    ))

    # Specify the remaining missing WCS values - coord. sys., scale and times
    # Transform to heliocentric ecliptic and then represent as rectangular
    # coordinates. This is what's required for RectangularHeliocentricCoords
    t = Time(mjd, format="mjd", scale="utc")

    # THIS IS NOT A CORRECT CALCULATION
    # SEE NOTEBOOK 02, WILL BE UPDATED PROMPTLY (note left @ March 22)
    refpix_helio = ICRS(
        ra=ra*u.deg,
        dec=dec*u.deg,
        distance=DISTANCE
    ).transform_to(HeliocentricTrueEcliptic(
        obstime=t
    ))

    refpix = sphere2rect(refpix_helio.lon.rad, refpix_helio.lat.rad)
    rectheliocoords.append(RectangularHeliocentricCoords(
        id = idx,
        refpix_x = refpix.x,
        refpix_y = refpix.y,
        refpix_z = refpix.z,
        tanwcs_id = idx,
        tanwcs = tanwcss[-1],
    ))

    if idx % chunksize == 0:
        et = time.time()
        logging.info(
            f"    {int(idx/chunksize)}/{nchunks} chunks faked. "
            f"Time elapsed: {et-st} seconds."
        )

# ingest the data into the DB in 10k slices because SQLite can't deal with more
# https://www.sqlite.org/limits.html#max_sql_length
# https://www.sqlite.org/limits.html#max_variable_number
twcs_chunks = chunkify(tanwcss, chunksize)
rect_chunks = chunkify(rectcoords, chunksize)
recthelio_chunks = chunkify(rectheliocoords, chunksize)

logging.info("Ingesting the faked pointings...")
for i, (tw_chunk, rc_chunk, rhc_chunk) in enumerate(zip(twcs_chunks, rect_chunks, recthelio_chunks)):
    with session.begin() as transaction:
        transaction.bulk_save_objects(tw_chunk)
        transaction.bulk_save_objects(rc_chunk)
        transaction.bulk_save_objects(rhc_chunk)
    logging.info(f"    {i+1}/{nchunks} chunks ingested successfully.")
et = time.time()
logging.info(f"Success! Total time elapsed {et-st} seconds.")
