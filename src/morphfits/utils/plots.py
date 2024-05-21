"""Visualize output data from MorphFITS.
"""

# Imports


import logging
from pathlib import Path

import numpy as np
from astropy.io import fits
from matplotlib import pyplot as plt

# montagepy doesn't have well-typed init
# See montage.ipac.caltech.edu/MontageNotebooks/mSubimage.html for more info
import MontagePy

from ..galwrap.setup import config


## Typing


from numpy import ndarray
from matplotlib.pyplot import Figure
from matplotlib.axes import Axes

from ..galwrap.config import GalWrapConfig


# Constants


logger = logging.getLogger("PLOT")
"""Logger object for processes from this module.
"""


# Functions


def make_cutouts(
    image_path: str | Path,
    stamp_path: str | Path,
    ra: float,
    dec: float,
    size: float = 6.0,
    remove_edge: bool = False,
):
    """Generate cutouts from an image.

    Parameters
    ----------
    image_path : str | Path
        Path to FITS image.
    stamp_path : str | Path
        Path to which to write generated cutout.
    ra : float
        Right ascension of query centre, in decimal degrees.
    dec : float
        Declination of query centre, in decimal degrees.
    size : float, optional
        Length of square cutout edge, in arcsec, by default 6.0.
    remove_edge : bool, optional
        Flag to remove generated cutout if image is largely on edge of mosaic,
        by default False.
    """
    # Cutout subimage from image and record status
    logger.info(f"Creating cutout for {image_path}.")
    subimage_status = MontagePy.main.mSubimage(
        infile=image_path,
        outfile=stamp_path,
        ra=ra,
        dec=dec,
        xsize=size,
        ysize=size,
        mode=0,
    )

    # Remove generated cutouts if source image improper
    if subimage_status["status"] == "0":
        if subimage_status["content"] == b"blank":
            logger.debug("Blank image, cutout not generated.")
            stamp_path.unlink()
        elif remove_edge and (np.median(fits.open(stamp_path)[0].data) == 0):
            logger.debug("Edge image, cutout not generated.")
            stamp_path.unlink()
