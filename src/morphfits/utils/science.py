"""Science utility functions for MorphFITS.
"""

# Imports


import logging
from pathlib import Path

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table


# Constants


logger = logging.getLogger("SCIENCE")
"""Logging object for this module."""


MINIMUM_IMAGE_SIZE = 32
"""Minimum pixel length of square stamp image.
"""


KRON_SCALE_FACTOR = 20
"""Scale factor by which to multiply Kron radius for image size.
"""


PHOTOMETRY_ZEROPOINT = 23.9
"""AB zeropoint for fluxes from the DJA photometric catalogs.
"""


# Functions


def get_fits_data(
    path: Path, hdu: str | int = "PRIMARY"
) -> tuple[np.ndarray, fits.Header]:
    """Get the image data and headers from a FITS file.

    Closes the file so the limit of open files is not encountered.

    Parameters
    ----------
    path : Path
        Path to FITS file.
    hdu : int | str, optional
        Index in HDU list at which to retrieve HDU, by default "PRIMARY".

    Returns
    -------
    tuple[np.ndarray, fits.Header]
        The image as a 2D float array, and its corresponding header object.
    """
    # Open FITS file
    fits_file = fits.open(path)

    # Get data and headers from file
    image, headers = fits_file[hdu].data, fits_file[hdu].header

    # Close file and return
    fits_file.close()
    return image, headers


def get_all_objects(input_catalog_path: Path) -> list[int]:
    """Get a list of all object integer IDs in a catalog.

    Parameters
    ----------
    input_catalog_path : Path
        Path to input catalog FITS file.

    Returns
    -------
    list[int]
        List of all integer IDs corresponding to each object's ID in the input
        catalog.
    """
    # Read input catalog
    input_catalog = Table.read(input_catalog_path)

    # Return list of IDs as integers
    return [int(id_object) - 1 for id_object in input_catalog["id"]]


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


def get_kron_radius(input_catalog: Table, catalog_version: str, object: int) -> float:
    """Get the Kron radius for an object from its corresponding photometric
    catalog.

    Parameters
    ----------
    input_catalog : Table
        Catalog detailing each identified object in a field.
    catalog_version : str
        Version of cataloging, e.g. 'dja-v7.2'.
    object : int
        Integer ID of object in catalog.

    Returns
    -------
    float
        Characteristic radius of object.

    Raises
    ------
    NotImplementedError
        Unknown catalog version.
    """
    # Expecting DJA catalog version keys to get kron radius
    if "dja" in catalog_version:
        # Get Kron radius from catalog
        if "kron_radius_circ" in input_catalog.keys():
            kron_radius = input_catalog[object]["kron_radius_circ"]
        else:
            kron_radius = input_catalog[object]["kron_radius"]

        # Return radius
        return kron_radius

    # Other catalog versions may store their kron radius elsewhere
    else:
        raise NotImplementedError("unrecognized")


def get_flux(
    input_catalog: Table, catalog_version: str, filter: str, object: int
) -> float:
    """Get the integrated flux for an object from its corresponding photometric
    catalog.

    Parameters
    ----------
    input_catalog : Table
        Catalog detailing each identified object in a field.
    catalog_version : str
        Version of cataloging, e.g. 'dja-v7.2'.
    filter : str
        Name of filter, e.g. 'f200w-clear'.
    object : int
        Integer ID of object in catalog.

    Returns
    -------
    float
        Integrated flux within an effective radius for object in a given filter.

    Raises
    ------
    NotImplementedError
        Unknown catalog version.
    """
    # Expecting DJA catalog version keys to get kron radius
    if "dja" in catalog_version:
        # Get cleaned filter name
        if "-" in filter:
            filters = filter.split("-")
            filter = filters[1] if "clear" in filters[0] else filters[0]

        # Get flux from catalog
        flux_key = f"{filter}_corr_1"
        if flux_key in input_catalog.keys():
            flux = input_catalog[object][flux_key]
        else:
            raise KeyError(f"flux key {flux_key} not found")

        # Return flux
        return flux

    # Other catalog versions may store their kron radius elsewhere
    else:
        raise NotImplementedError("unrecognized")


def get_image_size(radius: float) -> int:
    """Calculate the square pixel length of an image containing an object, from
    its cataloged Kron radius.

    Parameters
    ----------
    radius : float
        Characteristic radius of object, in pixels. By default, known as Kron
        radius.

    Returns
    -------
    int
        Number of pixels in each edge of a square image containing this object.
    """
    # Calculate image size from scale factor
    image_size = int(radius * KRON_SCALE_FACTOR)

    # Return maximum between calculated and minimum image size
    return np.nanmax([image_size, MINIMUM_IMAGE_SIZE])


def get_half_light_radius(input_catalog: Table, object: int) -> float:
    """Get the half light radius for an object from its entry in its
    corresponding input catalog.

    Parameters
    ----------
    input_catalog : Table
        Table cataloging each object in its field.
    object : int
        Integer ID of object in the catalog.

    Returns
    -------
    float
        Half light radius of object.
    """
    return input_catalog[object]["a_image"]


def get_axis_ratio(input_catalog: Table, object: int) -> float:
    """Get the axis ratio for an object from its entry in its corresponding
    input catalog.

    Parameters
    ----------
    input_catalog : Table
        Table cataloging each object in its field.
    object : int
        Integer ID of object in the catalog.

    Returns
    -------
    float
        Axis ratio of object.
    """
    return input_catalog[object]["b_image"] / input_catalog[object]["a_image"]


def get_surface_brightness(
    radius: float,
    pixscale: tuple[int, int],
    flux: float,
    zeropoint: float = PHOTOMETRY_ZEROPOINT,
) -> float:
    """Calculate an estimate of the surface brightness of an object, as an AB
    magnitude.

    Parameters
    ----------
    radius : float
        Characteristic radius of object, in pixels. By default, known as Kron
        radius.
    pixscale : tuple[int, int]
        Pixel scale along the x and y axes, respectively, in arcseconds per
        pixel.
    flux : float
        Integrated flux across an effective radius (distinct from the radius
        parameter for this function). By default, from the photometric catalog.
    zeropoint : float
        Zeropoint magnitude for this field, as an AB magnitude.

    Returns
    -------
    float
        Surface brightness of the object at its center.
    """
    # Calculate magnitude from integrated flux and offset by zeropoint
    magnitude = -2.5 * np.log10(flux) + zeropoint

    # Calculate area within radius as squared arcseconds
    area = np.pi * np.power(radius * np.average(pixscale), 2)

    # Calculate and return surface brightness as magnitude offset by area
    return np.nan_to_num(magnitude + 2.5 * np.log10(area))


def get_surface_brightness_from_headers(
    runtime_settings, headers: fits.Header
) -> float:
    """Get the surface brightness for a FICLO's product from its headers.

    Parameters
    ----------
    runtime_settings : RuntimeSettings
        Settings for this runtime.
    headers : fits.Header
        Headers for this FICLO product.

    Returns
    -------
    float
        Surface brightness for this FICLO product.
    """
    return headers["SB"]


def get_pixscale(path: Path):
    """Get an observation's pixscale from its FITS image frame.

    Used because not every frame has the same pixel scale. For the most
    part, long wavelength filtered observations have scales of 0.04 "/pix,
    and short wavelength filters have scales of 0.02 "/pix.

    Parameters
    ----------
    path : Path
        Path to FITS frame.

    Raises
    ------
    KeyError
        Coordinate transformation matrix element headers missing from frame.
    """
    # Get headers from FITS file
    headers = fits.getheader(path)

    # Get pixel scale if directly set as header
    if "PIXELSCL" in headers:
        return headers["PIXELSCL"]

    # Raise error if keys not found in header
    if any([header not in headers for header in ["CD1_1", "CD2_2", "CD1_2", "CD2_1"]]):
        raise KeyError(
            f"Science frame for science frame {path.name} "
            + "missing coordinate transformation matrix element header."
        )

    # Calculate and set pixel scales
    pixscale_x = np.sqrt(headers["CD1_1"] ** 2 + headers["CD1_2"] ** 2) * 3600
    pixscale_y = np.sqrt(headers["CD2_1"] ** 2 + headers["CD2_2"] ** 2) * 3600
    return (pixscale_x, pixscale_y)


def get_str_from_pixscale(pixscale: tuple[int, int]) -> str:
    """Get a pixel scale as a string.

    Parameters
    ----------
    pixscale : tuple[int,int]
        Pixel scale along x and y axes, respectively, in arcseconds per pixel.

    Returns
    -------
    str
        Pixel scale as a string, in milli-arcseconds per pixel (only expressing
        'mas').
    """
    # Get maximum pixel scale from pair
    max_pixscale = np.nanmax(pixscale)

    # Return pixel scale in milli-arcseconds, as a str
    return str(round(max_pixscale * 1000)) + "mas"
