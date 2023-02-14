import os
import time
import logging

import numpy as np
import astropy.units as u
from astropy.wcs import WCS
from astropy.io import fits

from models import (
    session,
    TanWcs,
    RectangularCoords,
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
logging.info(f"Creating example.sqlite3 DB containing {NFRAMES} faked pointings.")


###############################################################################
#                               ADVANCED CONFIG
###############################################################################
# these settings are modifiable but only if you know what they docs. We are
# faking a bunch of WCSs. This has to be done with some amount of attention to
# details because validation will be difficult and results won't make sense.
chunksize = 10000
idxs = np.arange(1, NFRAMES+1, dtype=int)
ras = np.random.uniform(0, 360.0, size=NFRAMES)
decs = np.random.uniform(-90.0, 90.0, size=NFRAMES)


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


###############################################################################
#                                FAKING DATA
###############################################################################

def chunkify(iterable, chunk_size=10000):
    """Yields successive `chunk_size`d chunks from `iterable`."""
    for i in range(0, len(iterable), chunk_size):
        yield iterable[i:i+chunk_size]


# cleanup old remains
if os.path.exists("example.sqlite3"):
    os.remove("example.sqlite3")
    logging.info(f"Found old DB, removed it.")
    Base.metadata.create_all(engine)


logging.info("Faking pointings....")
st = time.time()

# create the lists of related TanWCS and RectangularCoords to ingest into DB.
# The faked WCSs will all have the same distortion coefficients to the sky
# coordinates and the reference pixel, which is a realistic scenario for an
# space observatory over reasonably-short time periods.
# Each, however, will have a different value for the CRVAL12, setting different
# values of sky coordinates for the reference pixel.
# For each of these, we need to construct a WCS, in order to properly calculate
# the value of the corner pixel and radii. This is not immediately important
# for us, but is important when trying to calculate overlap.
# These, refpix and corner, are then converted to rectangular coordinates.
tanwcss, rectcoords = [], []
for idx, ra, dec in zip(idxs, ras, decs):
    # create our TanWCS, assign the idx to speed up ingestion into DB
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
    ))

    # Construct the WCS, calc corner, convert to rect, create and append obj
    hdu.header["CRVAL1"] = ra
    hdu.header["CRVAL2"] = dec
    wcs = WCS(hdu)

    ra_rad, dec_rad = ra*DEG2RAD, dec*DEG2RAD
    refpix =  sphere2rect(ra_rad, dec_rad)

    corner_sky = wcs.wcs_pix2world([ra, ], [dec, ], 1)
    corner_sky_rad = (corner_sky*u.degree).to(u.rad)
    corner = sphere2rect(*corner_sky_rad)

    radius = np.sqrt((corner.x-refpix.x)**2 + (corner.y-refpix.y)**2 + (corner.z-refpix.z)**2)

    # note indices must match for foreign keys, but for us it's the same
    rectcoords.append(RectangularCoords(
        id = idx,
        refpix_x = refpix.x,
        refpix_y = refpix.y,
        refpix_z = refpix.z,
        corner_x = corner.x,
        corner_y = corner.y,
        corner_z = corner.z,
        radius =   radius,
        tanwcs_id = idx,
        tanwcs =   tanwcss[-1],
    ))

et = time.time()
logging.info(f"Successfully faked {NFRAMES} pointings in {et-st} seconds.")


# ingest the data into the DB in 10k slices because SQLite can't deal with more
# https://www.sqlite.org/limits.html#max_sql_length
# https://www.sqlite.org/limits.html#max_variable_number
twcs_chunks = chunkify(tanwcss, chunksize)
rect_chunks = chunkify(rectcoords, chunksize)

nchunks = int(np.rint(NFRAMES/chunksize))
logging.info("Ingesting the faked pointings...")
for i, (tw_chunk, rc_chunk) in enumerate(zip(twcs_chunks, rect_chunks)):
    with session.begin() as transaction:
        transaction.bulk_save_objects(tw_chunk)
        transaction.bulk_save_objects(rc_chunk)
    logging.info(f"    {i+1}/{nchunks} chunks ingested successfully.")
logging.info("Success!")
