"""Rescale and re-bin a simulated PSF FITS file to a new pixel scale
corresponding to a science observation FITS file.
"""

# Imports


import sys
import re
from pathlib import Path

import numpy as np
from scipy import ndimage as spn
from astropy.io import fits


# Functions


## Secondary


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


def get_pixscale(path: Path) -> float:
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
    float
        Pixel scale of FITS image.

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
            f"Frame '{path.name}' missing coordinate transformation matrix headers."
        )

    # Calculate and set pixel scales
    pixscale_x = np.sqrt(headers["CD1_1"] ** 2 + headers["CD1_2"] ** 2) * 3600
    pixscale_y = np.sqrt(headers["CD2_1"] ** 2 + headers["CD2_2"] ** 2) * 3600
    return np.average(np.array([pixscale_x, pixscale_y]))


# Primary


def rescale(
    psf_path: Path,
    science_path: Path | None = None,
    pixscale: float | None = None,
):
    """Re-bin a PSF FITS file to a new pixel scale corresponding to a science
    frame from the same wavelength band.

    Parameters
    ----------
    psf_path : Path
        Path to PSF file to re-bin.
    science_path : Path | None, optional
        Path to science frame with same filter, by default None (pixel scale
        provided instead).
    pixscale : float | None, optional
        Pixel scale by which to rescale PSF, by default None (science frame
        provided instead).
    """
    # Get new pixel scale as float
    if pixscale is None:
        pixscale = get_pixscale(science_path)
    elif (pixscale is not None) and (isinstance(pixscale, tuple)):
        pixscale = max(pixscale)

    # Open PSF
    psf_data, psf_headers = get_fits_data(psf_path)

    # Get original pixel scale from headers
    psf_pixscale = get_pixscale(psf_path)

    # Rebin PSF to new pixel scale
    psf_rebin = spn.zoom(input=psf_data, zoom=psf_pixscale / pixscale, order=0)

    # Set new pixel scale in headers
    psf_headers["PIXELSCL"] = pixscale

    # Get new filename
    filter = re.findall(r"F\d{3}[A-Z]", psf_path.name)[0].lower() + "-clear"
    pixscale_str = str(round(pixscale * 1000)) + "mas"
    filename = f"{filter}_{pixscale_str}_psf.fits"

    # Save to FITS file in this directory
    fits.writeto(filename, psf_rebin, psf_headers)


if __name__ == "__main__":
    # Get path to PSF from first argument
    psf_path = Path(sys.argv[1])
    assert psf_path.exists(), "PSF not found."

    # Get path to science frame or pixel scale from second argument
    try:
        science_path = Path(sys.argv[2])
        assert science_path.exists(), "Science not found."
    except:
        science_path = None
    try:
        pixscale = float(sys.argv[2])
    except:
        pixscale = None

    # Terminate if neither science frame nor pixel scale given
    if science_path is None and pixscale is None:
        raise KeyError("Missing pixel scale or path to science frame.")

    # Re-bin PSF and write to scripts directory
    rescale(psf_path=psf_path, science_path=science_path, pixscale=pixscale)
