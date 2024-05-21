"""Science utility functions for MorphFITS.
"""

# Imports


from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.io.fits import PrimaryHDU


# Functions


def get_pixname(pixscale: float) -> str:
    """Get a resolution name from its corresponding scale.

    Parameters
    ----------
    pixscale : float
        Pixel scale.

    Returns
    -------
    str
        Pixel scale as human-readable text.
    """
    return str(int(pixscale * 10**3)) + "mas"


def get_pixscale(pixname: str) -> float:
    """Get a resolution scale from its corresponding name.

    Parameters
    ----------
    pixname : str
        Pixel scale as text.

    Returns
    -------
    float
        Corresponding pixel scale.
    """
    return float(pixname[:-3]) / 10**3


def get_zeropoint(image_path: str | Path, magnitude_system: str = "AB") -> float:
    """Calculate image zeropoint in passed magnitude system.

    Parameters
    ----------
    image_path : str | Path
        Path to image FITS file.
    magnitude_system : str, optional
        Magnitude system to calculate zeropoint from, by default "AB".

    Returns
    -------
    float
        Zeropoint of image.

    Raises
    ------
    NotImplementedError
        Magnitude system unknown.

    Notes
    -----
    Image header must contain keys `PHOTFLAM` and `PHOTPLAM`.

    References
    ----------
    1. https://www.stsci.edu/hst/instrumentation/acs/data-analysis/zeropoints#:~:text=The%20PHOTFLAM%20and%20PHOTPLAM%20header,TPLAM)%E2%88%92
    """
    # Open FITS image
    image: PrimaryHDU = fits.open(image_path)[0]

    # Find zeropoint by magnitude system
    match magnitude_system:
        case "AB":
            return (
                -2.5 * np.log10(image.header["PHOTFLAM"])
                - 5 * (np.log10(image.header["PHOTPLAM"]))
                - 2.408
            )
        case "ST":
            return -2.5 * np.log10(image.header["PHOTFLAM"]) - 21.1
        case _:
            raise NotImplementedError(
                f"Magnitude system {magnitude_system} not implemented."
            )
