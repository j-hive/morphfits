"""Science utility functions for MorphFITS.
"""

# Imports


import logging
from pathlib import Path

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from astropy.wcs import utils as wcs_utils

from . import path


# Constants


logger = logging.getLogger("SCIENCE")
"""Logging object for this module."""


MINIMUM_IMAGE_SIZE = 32
"""Minimum pixel length of square stamp image.
"""


KRON_SCALE_FACTOR = 3
"""Scale factor by which to multiply Kron radius for image size.
"""


# Functions


def get_fits_data(
    name: str,
    morphfits_config,
    ficl,
    object: int | None = None,
) -> tuple[np.ndarray, fits.Header]:
    """Get the image data and headers from a FITS file.

    Closes the file so the limit of open files is not encountered.

    Parameters
    ----------
    name : str
        MorphFITS path name of FITS file.
    morphfits_config : MorphFITSConfig
        Config object for this program run.
    ficl : FICL
        The field, image version, catalog version, filter, and other details of
        a single observation.
    object : int | None, optional
        Integer ID of object in its catalog, by default None (N/A).

    Returns
    -------
    tuple[np.ndarray, fits.Header]
        The image as a 2D float array, and its corresponding header object.
    """
    # Get path to FITS file
    fits_path = path.get_path(
        name=name,
        morphfits_config=morphfits_config,
        ficl=ficl,
        object=object,
    )

    # Open FITS file
    fits_file = fits.open(fits_path)

    # Get data and headers from file
    image, headers = fits_file["PRIMARY"].data, fits_file["PRIMARY"].header

    # Close file and return
    fits_file.close()
    return image, headers


def get_zeropoint(headers: fits.Header, magnitude_system: str = "AB") -> float:
    """Calculate the zeropoint of an observation.

    If the observation's headers contains the keyword 'ZP', this will be the
    returned value, otherwise, calculate from the 'AB' or 'ST' magnitude system
    formulas.

    Parameters
    ----------
    headers : fits.Header
        Headers of this observation's FITS file.
    magnitude_system : str, optional
        Magnitude system by which to calculate zeropoint, by default "AB".

    Returns
    -------
    float
        Zeropoint of this observation.

    Raises
    ------
    NotImplementedError
        Unrecognized magnitude system. Only 'AB' and 'ST' implemented.
    """
    # If zeropoint stored in headers, return
    if "ZP" in headers:
        return headers["ZP"]

    # Otherwise, calculate zeropoint from magnitude system
    match magnitude_system:
        case "AB":
            return (
                -2.5 * np.log10(headers["PHOTFLAM"])
                - 5 * (np.log10(headers["PHOTPLAM"]))
                - 2.408
            )
        case "ST":
            return -2.5 * np.log10(headers["PHOTFLAM"]) - 21.1
        case _:
            raise NotImplementedError(
                f"Magnitude system {magnitude_system} not implemented."
            )


def get_position(input_catalog: Table, object: int) -> SkyCoord:
    """Retrieve the RA and Dec of an object in its catalog as a SkyCoord object.

    Parameters
    ----------
    input_catalog : Table
        Catalog detailing each identified object in a field.
    object : int
        Integer ID of object in catalog.

    Returns
    -------
    SkyCoord
        Position of object as a SkyCoord astropy object.
    """
    return SkyCoord(
        ra=input_catalog[object]["ra"], dec=input_catalog[object]["dec"], unit="deg"
    )


def get_image_size(
    input_catalog: Table, catalog_version: str, object: int, pixscale: tuple[int, int]
) -> int:
    """Calculate the square pixel length of an image containing an object, from
    its cataloged Kron radius.

    Parameters
    ----------
    input_catalog : Table
        Catalog detailing each identified object in a field.
    catalog_version : str
        Version of cataloging, e.g. 'dja-v7.2'.
    object : int
        Integer ID of object in catalog.
    pixscale : tuple[int, int]
        Pixel scale along x-axis and y-axis of the observation, in
        arcseconds/pixel.

    Returns
    -------
    int
        Number of pixels in each edge of a square image containing this object.
    """
    # Expecting DJA catalog version keys to get kron radius
    if "dja" in catalog_version:
        # Get Kron radius from catalog
        if "kron_radius_circ" in input_catalog.keys():
            kron_radius = input_catalog[object]["kron_radius_circ"]
        else:
            kron_radius = input_catalog[object]["kron_radius"]

        # Calculate image size from scale factor
        image_size = int(kron_radius / np.nanmax(pixscale) * KRON_SCALE_FACTOR)

        # Return maximum between calculated and minimum image size
        return np.nanmax([image_size, MINIMUM_IMAGE_SIZE])
    # Other catalog versions may store their kron radius elsewhere
    else:
        logger.warning(f"Catalog version {catalog_version} unrecognized.")
        return MINIMUM_IMAGE_SIZE


def get_surface_brightness(
    image: np.ndarray, pixscale: tuple[int, int], zeropoint: float
) -> float:
    """Calculate the flux per pixel of an object at its center (peak).

    Parameters
    ----------
    image : ndarray
        Observation cutout of object, as a 2D float array.
    pixscale : tuple[int, int]
        Pixel scale along x-axis and y-axis of the image, in arcseconds/pixel.
    zeropoint : float
        Zeropoint of the observation, in AB magnitude.

    Returns
    -------
    float
        Surface brightness of the object at its center.
    """
    # Get location of center of image
    center = int(image.shape[0] / 2)

    # If image size is odd, get 9 center pixels, otherwise 4
    odd_flag = image.shape[0] % 2

    # Get total flux and area across center pixels
    total_flux = np.sum(
        image[
            center - 1 : center + 1 + odd_flag,
            center - 1 : center + 1 + odd_flag,
        ]
    )
    total_area = ((2 + odd_flag) ** 2) * pixscale[0] * pixscale[1]

    # Get flux per pixel
    flux_per_pixel = total_flux / total_area

    # Return zeroed log of flux per pixel
    return np.nan_to_num(-2.5 * np.log10(flux_per_pixel) + zeropoint)


def get_pixscale(fits_path: Path):
    """Get an observation's pixscale from its FITS image frame.

    Used because not every frame has the same pixel scale. For the most
    part, long wavelength filtered observations have scales of 0.04 "/pix,
    and short wavelength filters have scales of 0.02 "/pix.

    Parameters
    ----------
    science_path : Path
        Path to science frame.

    Raises
    ------
    KeyError
        Coordinate transformation matrix element headers missing from frame.
    """
    # Get headers from FITS file
    headers = fits.getheader(fits_path)

    # Raise error if keys not found in header
    if any([header not in headers for header in ["CD1_1", "CD2_2", "CD1_2", "CD2_1"]]):
        raise KeyError(
            f"Science frame for science frame {fits_path.name} "
            + "missing coordinate transformation matrix element header."
        )

    # Calculate and set pixel scales
    pixscale_x = np.sqrt(headers["CD1_1"] ** 2 + headers["CD1_2"] ** 2) * 3600
    pixscale_y = np.sqrt(headers["CD2_1"] ** 2 + headers["CD2_2"] ** 2) * 3600
    return (pixscale_x, pixscale_y)
