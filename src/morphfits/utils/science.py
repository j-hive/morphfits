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


PHOTOMETRY_ZEROPOINT = 23.9
"""AB zeropoint for fluxes from the DJA photometric catalogs.
"""


APERTURE_1_RADIUS = 0.25
"""Radius of aperture 1, in arcseconds, corresponding to retrieving
'filter_corr_1' for flux.

See Also
--------
https://dawn-cph.github.io/dja/blog/2023/07/14/photometric-catalog-demo/
    Other aperture radii, under 'Photometric apertures'.
"""


# Functions


## FITS


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
    fits_hdu: fits.PrimaryHDU = fits_file[hdu]

    # Get data and headers from file
    image, headers = fits_hdu.data, fits_hdu.header

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
                f"magnitude system {magnitude_system} not implemented"
            )


def get_integrated_magnitude_from_headers(headers: fits.Header) -> float:
    """Get the integrated magnitude for a FICLO's product from its headers.

    Parameters
    ----------
    headers : fits.Header
        Headers for this FICLO product.

    Returns
    -------
    float
        Integrated magnitude for this FICLO product.
    """
    return headers["IM"]


def get_surface_brightness_from_headers(headers: fits.Header) -> float:
    """Get the surface brightness for a FICLO's product from its headers.

    Parameters
    ----------
    headers : fits.Header
        Headers for this FICLO product.

    Returns
    -------
    float
        Surface brightness for this FICLO product.
    """
    return headers["SB"]


## Photometric Catalog


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
    return [int(id_object) for id_object in input_catalog["id"]]


def get_unflagged_objects(
    objects: list[int], ingest_catalog_path: Path, filter: str
) -> list[int]:
    """Get a list of object IDs not flagged in a passed ingest catalog.

    Parameters
    ----------
    objects : list[int]
        Initial list of object IDs.
    ingest_catalog_path : Path
        Catalog with a row for each object, with a general and per-filter
        quality flag.
    filter : str
        Filter name.

    Returns
    -------
    list[int]
        List of IDs validated against ingest catalog.
    """
    # Read ingest catalog
    ingest_catalog = Table.read(ingest_catalog_path)

    # Get cleaned filter name
    if "-" in filter:
        filter_split = filter.split("-")
        band = filter_split[1 if "clear" in filter_split[0] else 0]

    # Iterate over each object in initial list and add to new list if not
    # flagged
    unflagged_objects = []
    flag_header = f"ingest_{band}"
    for object in objects:
        if not ingest_catalog[ingest_catalog["id"] == object][flag_header]:
            unflagged_objects.append(object)

    # Return not flagged objects
    return unflagged_objects


def get_catalog_row(input_catalog: Table, object: int) -> Table:
    """Get an object's data row in its corresponding photometric catalog.

    Parameters
    ----------
    input_catalog : Table
        Catalog detailing each identified object in a field.
    object : int
        Integer ID of object in catalog.

    Returns
    -------
    Table
        Data row of object, if it is found.
    """
    return input_catalog[input_catalog["id"] == object]


def get_catalog_datum(
    key: str,
    type: type,
    row: Table | None = None,
    input_catalog: Table | None = None,
    object: int | None = None,
) -> bool | int | float:
    """Get a casted datum from a catalog row.

    Parameters
    ----------
    key : str
        Key in catalog, i.e. column name.
    type : type
        Type to which to cast.
    row : Table | None, optional
        Data row of object in catalog, by default None (catalog provided).
    input_catalog : Table | None, optional
        Catalog detailing each identified object in a field, by default None
        (row provided).
    object : int | None, optional
        Integer ID of object in catalog, by default None (row provided).


    Returns
    -------
    bool | int | float
        Casted datum from catalog row.
    """
    # Get row if not passed
    if row is None:
        row = get_catalog_row(input_catalog, object)

    # Return casted value
    return type(row[key])


def get_position(
    row: Table | None = None,
    input_catalog: Table | None = None,
    object: int | None = None,
) -> SkyCoord:
    """Retrieve the RA and Dec of an object, as a SkyCoord object.

    Parameters
    ----------
    row : Table | None, optional
        Data row of object in catalog, by default None (catalog provided).
    input_catalog : Table | None, optional
        Catalog detailing each identified object in a field, by default None
        (row provided).
    object : int | None, optional
        Integer ID of object in catalog, by default None (row provided).

    Returns
    -------
    SkyCoord
        Position of object as a SkyCoord astropy object.
    """
    return SkyCoord(
        ra=get_catalog_datum("ra", float, row, input_catalog, object),
        dec=get_catalog_datum("dec", float, row, input_catalog, object),
        unit="deg",
    )


def get_flux(
    filter: str,
    row: Table | None = None,
    input_catalog: Table | None = None,
    object: int | None = None,
) -> float:
    """Get the integrated flux for an object, in Jy.

    Parameters
    ----------
    filter : str
        Filter from which to get flux for object.
    row : Table | None, optional
        Data row of object in catalog, by default None (catalog provided).
    input_catalog : Table | None, optional
        Catalog detailing each identified object in a field, by default None
        (row provided).
    object : int | None, optional
        Integer ID of object in catalog, by default None (row provided).

    Returns
    -------
    float
        Integrated flux within an effective radius for object in a given filter,
        in Jy.
    """
    # Get cleaned filter name
    if "-" in filter:
        filters = filter.split("-")
        filter = filters[1] if "clear" in filters[0] else filters[0]

    # Get flux from catalog and convert from uJy
    flux_key = f"{filter}_corr_1"
    return get_catalog_datum(flux_key, float, row, input_catalog, object) * 1e3


def get_flux_radius(
    row: Table | None = None,
    input_catalog: Table | None = None,
    object: int | None = None,
) -> float:
    """Get the flux radius for an object, in arcseconds.

    Parameters
    ----------
    row : Table | None, optional
        Data row of object in catalog, by default None (catalog provided).
    input_catalog : Table | None, optional
        Catalog detailing each identified object in a field, by default None
        (row provided).
    object : int | None, optional
        Integer ID of object in catalog, by default None (row provided).

    Returns
    -------
    float
        Flux radius of object.
    """
    return get_catalog_datum("flux_radius", float, row, input_catalog, object)


def get_kron_radius(
    row: Table | None = None,
    input_catalog: Table | None = None,
    object: int | None = None,
) -> float:
    """Get the Kron radius for an object, in pixels.

    Parameters
    ----------
    row : Table | None, optional
        Data row of object in catalog, by default None (catalog provided).
    input_catalog : Table | None, optional
        Catalog detailing each identified object in a field, by default None
        (row provided).
    object : int | None, optional
        Integer ID of object in catalog, by default None (row provided).

    Returns
    -------
    float
        Characteristic radius of object.
    """
    return get_catalog_datum("kron_radius", float, row, input_catalog, object)


def get_a(
    row: Table | None = None,
    input_catalog: Table | None = None,
    object: int | None = None,
) -> float:
    """Get the semi-major axis for an object, in arcseconds.

    Parameters
    ----------
    row : Table | None, optional
        Data row of object in catalog, by default None (catalog provided).
    input_catalog : Table | None, optional
        Catalog detailing each identified object in a field, by default None
        (row provided).
    object : int | None, optional
        Integer ID of object in catalog, by default None (row provided).

    Returns
    -------
    float
        Semi-major axis of object, in arcseconds.
    """
    return get_catalog_datum("a_image", float, row, input_catalog, object)


def get_axis_ratio(
    row: Table | None = None,
    input_catalog: Table | None = None,
    object: int | None = None,
) -> float:
    """Get the axis ratio for an object.

    Parameters
    ----------
    row : Table | None, optional
        Data row of object in catalog, by default None (catalog provided).
    input_catalog : Table | None, optional
        Catalog detailing each identified object in a field, by default None
        (row provided).
    object : int | None, optional
        Integer ID of object in catalog, by default None (row provided).

    Returns
    -------
    float
        Axis ratio of object.
    """
    a = get_catalog_datum("a_image", float, row, input_catalog, object)
    b = get_catalog_datum("b_image", float, row, input_catalog, object)
    return b / a


## Calculation


def get_image_size(radius: float, scale: float, minimum: int) -> int:
    """Calculate the square pixel length of an image containing an object, from
    its cataloged Kron radius.

    Parameters
    ----------
    radius : float
        Characteristic radius of object, in pixels.
    scale : float
        Scale factor by which to multiply initial radius.
    minimum : int
        Minimum image size, in pixels.

    Returns
    -------
    int
        Number of pixels in each edge of a square image containing this object.
    """
    # Calculate image size from scale factor
    image_size = int(radius * scale)

    # Return maximum between calculated and minimum image size
    return np.nanmax([image_size, minimum])


def get_integrated_magnitude(
    flux: float, zeropoint: float = PHOTOMETRY_ZEROPOINT
) -> float:
    """Calculate an estimate of the integrated magnitude of an object, as an AB
    magnitude.

    Parameters
    ----------
    flux : float
        Integrated flux across an effective radius (distinct from the radius
        parameter for this function). By default, from the photometric catalog.
    zeropoint : float, optional
        Zeropoint magnitude for this field, as an AB magnitude, by default 23.9.

    Returns
    -------
    float
        Integrated magnitude of the object.

    Raises
    ------
    ValueError
        Negative flux.
    """
    # Raise error if flux negative
    if flux <= 0:
        raise ValueError(f"flux {flux} negative")

    # Calculate and return magnitude from integrated flux and offset by zeropoint
    magnitude = -2.5 * np.log10(flux) + zeropoint
    return magnitude


def get_surface_brightness(
    flux: float,
    radius: float = APERTURE_1_RADIUS,
    pixscale: tuple[int, int] | None = None,
    zeropoint: float = PHOTOMETRY_ZEROPOINT,
) -> float:
    """Calculate an estimate of the surface brightness of an object, as an AB
    magnitude.

    Parameters
    ----------
    flux : float
        Integrated flux across an effective radius (distinct from the radius
        parameter for this function). By default, from the photometric catalog.
    radius : float
        Characteristic radius of object, in arcseconds or pixels, by default
        0.25".
    pixscale : tuple[int, int] | None, optional
        Pixel scale along the x and y axes, respectively, in arcseconds per
        pixel, by default None (radius in arcseconds).
    zeropoint : float, optional
        Zeropoint magnitude for this field, as an AB magnitude, by default 23.9.

    Returns
    -------
    float
        Surface brightness of the object at its center.

    Raises
    ------
    ValueError
        Negative flux.
    """
    # Raise error if flux negative
    if flux <= 0:
        raise ValueError(f"flux {flux} negative")

    # Calculate magnitude from integrated flux and offset by zeropoint
    magnitude = -2.5 * np.log10(flux) + zeropoint

    # Calculate area within radius as squared arcseconds
    radius_in_as = radius * (1 if pixscale is None else max(pixscale))
    area = np.pi * np.power(radius_in_as, 2)
    offset = 2.5 * np.log10(area)

    # Calculate and return surface brightness as magnitude offset by area
    return magnitude + offset


def get_length_in_px(length: float, pixscale: tuple[float, float]) -> float:
    """Get a length in pixels, from arcseconds.

    Parameters
    ----------
    length : float
        Length to convert to pixels, in arcseconds.
    pixscale : tuple[float,float]
        Pixel scale along each axis, in arcseconds per pixel.

    Returns
    -------
    float
        Length in pixels.
    """
    return length / max(pixscale)


def get_pixscale(path: Path) -> tuple[float, float]:
    """Get an observation's pixscale from its FITS image frame.

    Used because not every frame has the same pixel scale. For the most
    part, long wavelength filtered observations have scales of 0.04 "/pix,
    and short wavelength filters have scales of 0.02 "/pix.

    Parameters
    ----------
    path : Path
        Path to FITS frame.

    Returns
    -------
    tuple[float, float]
        Pixel scale along x and y axes, respectively, in arcseconds per pixel.

    Raises
    ------
    KeyError
        Coordinate transformation matrix element headers missing from frame.
    """
    # Get headers from FITS file
    headers = fits.getheader(path)

    # Get pixel scale if directly set as header
    if "PIXELSCL" in headers:
        pixscale_str = headers["PIXELSCL"]

        # Try to get pixel scale from header as float
        try:
            pixscale = float(pixscale_str)
            return (pixscale, pixscale)

        # Try to get pixel scale from header as string
        except:
            try:
                pixscale = float(pixscale_str.split("mas")) * 1e-3
                return (pixscale, pixscale)

            # Other formats unknown
            except:
                raise ValueError(f"pixel scale {pixscale_str} unrecognized")

    # Get pixel scale from coordinate matrix headers if not set as header
    else:
        # Raise error if keys not found in header
        if any(
            [header not in headers for header in ["CD1_1", "CD2_2", "CD1_2", "CD2_1"]]
        ):
            raise KeyError(f"frame {path.name} missing coordinate matrix headers")

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
