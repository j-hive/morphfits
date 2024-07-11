"""Science utility functions for MorphFITS.
"""

# Imports


from pathlib import Path

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.io.fits import PrimaryHDU
from astropy.wcs import WCS
from astropy.wcs import utils as wcs_utils


# Functions


def get_pixname_from_scale(pixscale: float) -> str:
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


def get_pixscale_from_name(pixname: str) -> float:
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


def get_pixels_from_skycoord(
    skycoord: SkyCoord, wcs: WCS, image_size: int
) -> tuple[tuple[int, int], tuple[int, int]]:
    """TODO May be deleted. Check if unused.

    Calculate corresponding pixels in an image based on sky coordinates and a
    WCS.

    Parameters
    ----------
    skycoord : SkyCoord
        Position of object in sky.
    wcs : WCS
        Coordinate system from pixel to sky.
    image_size : int
        Number of pixels along one square image side.

    Returns
    -------
    tuple[tuple[int, int], tuple[int, int]]
        Integer pixel coordinates in order of less x, greater x, less y, greater
        y.
    """
    x_range, y_range = wcs_utils.skycoord_to_pixel(coords=skycoord, wcs=wcs)
    left, right = int(x_range - image_size), int(x_range + image_size)
    down, up = int(y_range - image_size), int(y_range + image_size)

    return (left, right), (down, up)


def get_pixscale(science_path: Path):
    """Get a FICL's pixscale from its science frame.

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
    # Get headers from science frame
    science_headers = fits.getheader(science_path)

    # Raise error if keys not found in header
    if any(
        [
            header not in science_headers
            for header in ["CD1_1", "CD2_2", "CD1_2", "CD2_1"]
        ]
    ):
        raise KeyError(
            f"Science frame for FICL {self} missing "
            + "coordinate transformation matrix element header."
        )

    # Calculate and set pixel scales
    pixscale_x = np.sqrt(science_headers["CD1_1"] ** 2 + science_headers["CD1_2"])
    pixscale_y = np.sqrt(science_headers["CD2_1"] ** 2 + science_headers["CD2_2"])
    return (pixscale_x, pixscale_y)


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
    image: PrimaryHDU = fits.open(image_path)["PRIMARY"]

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
