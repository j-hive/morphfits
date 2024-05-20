"""Science utility functions for MorphFITS.
"""

# Imports


from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from astropy.io.fits import PrimaryHDU
from jinja2 import Template

from .. import DATA_ROOT


# Functions


## Python


### Path


## General


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


## GALFIT


def generate_feedfile(
    feedfile_path: str | Path,
    galfit_output_path: str | Path,
    constraints_path: str | Path,
    stamp_path: str | Path,
    rms_path: str | Path,
    psf_path: str | Path,
    mask_path: str | Path,
    image_size: int,
    half_light_radius: float,
    axis_ratio: float,
    magnitude: float = 30.0,
    zeropoint: float = 31.50,
):
    """Generate and write a feedfile whose parameters configure a GALFIT run.

    Parameters
    ----------
    feedfile_path : str | Path
        Path to feedfile to be written.
    galfit_output_path : str | Path
        Path to directory to which to write GALFIT output.
    constraints_path : str | Path
        Path to GALFIT parameter constraints file.
    stamp_path : str | Path
        Path to input image stamp.
    rms_path : str | Path
        Path to input RMS map.
    psf_path : str | Path
        Path to input PSF.
    mask_path : str | Path
        Path to input mask.
    image_size : int
        Number of pixels in one axis of square image.
    half_light_radius : float
        Radius at which half of a source's light is contained, in pixels.
    axis_ratio : float
        Ratio between an object's ellipsoid axes.
    magnitude : float, optional
        Integrated magnitude, by default 30.0.
    zeropoint : float, optional
        Photometric magnitude zeropoint, by default 31.50.
    """

    def get_aligned_variable(variable: Any, align_size: int = 60) -> str:
        """Return a left-justified string of a variable, for comment alignment.

        Parameters
        ----------
        variable : Any
            Jinja variable to be left-justified.
        align_size : int, optional
            Number of space characters to left-justify variable with, i.e.
            length of str including empty characters, by default 60, i.e.
            comments begin on column 64.

        Returns
        -------
        str
            Left-justified string representation of variable.
        """
        aligned_variable = str(variable).ljust(align_size)
        return (
            aligned_variable[:align_size]
            if len(aligned_variable) > align_size
            else aligned_variable
        )

    # Set configuration parameters from input
    feedfile_dict = {
        "galfit_output_path": get_aligned_variable(galfit_output_path),
        "constraints_path": get_aligned_variable(constraints_path),
        "stamp_path": get_aligned_variable(stamp_path),
        "rms_path": get_aligned_variable(rms_path),
        "psf_path": get_aligned_variable(psf_path),
        "mask_path": get_aligned_variable(mask_path),
        "image_size": get_aligned_variable(image_size, align_size=6),
        "position": get_aligned_variable(image_size / 2.0, align_size=6),
        "half_light_radius": get_aligned_variable(half_light_radius, align_size=8),
        "axis_ratio": get_aligned_variable(round(axis_ratio, 2), align_size=8),
        "magnitude": get_aligned_variable(round(magnitude, 2), align_size=8),
        "zeropoint": get_aligned_variable(zeropoint, align_size=8),
    }

    # Write new feedfile from template and save to output directory
    with open(DATA_ROOT / "galfit_feedfile_template.jinja", "r") as feedfile_template:
        template = Template(feedfile_template.read())
    lines = template.render(feedfile_dict)
    with open(feedfile_path, "w") as feedfile:
        feedfile.write(lines)


if __name__ == "__main__":
    FEEDDIR = (
        f"/arc/projects/canucs/MORPHDIR/FITDIR/{OBJ}{FIELD}/{IMVER}.{CATVER}"
        + "FEEDDIR"
    )
    ID = "{ID}"
    FILT = "F140W"
    PIXNAME = "40mas"

    SCINAME = "../STAMPDIR/{}/{}_{}{}-{}_sci.fits".format(FILT, ID, OBJ, FIELD, FILT)
    STAMPOUT = "../GALFITOUTDIR/{}/{}_{}{}-{}_model.fits".format(
        FILT, ID, OBJ, FIELD, FILT
    )
    FEEDFILE = "{}{}/{}_{}{}-{}.feedfile".format(FEEDDIR, FILT, ID, OBJ, FIELD, FILT)
    PSFNAME = "../PSF/{}{}_{}_{}_psf_{}.{}_crop.fits".format(
        OBJ, FIELD, FILT, PIXNAME, IMVER, CATVER
    )
    MASKNAME = "../MASKDIR/{}_{}{}_mask.fits".format(ID, OBJ, FIELD)
    RMSNAME = "../RMSDIR/{}/{}_{}{}-{}_rms.fits".format(FILT, ID, OBJ, FIELD, FILT)

    IMSIZE = 2000
    MAG = 20.0
    A, B = 50.2, 120.0
    BA = B / A

    generate_feedfile(
        feedfile_path="./test.feedfile",
        galfit_output_path=STAMPOUT,
        constraints_path="BLUH",
        stamp_path=SCINAME,
        rms_path=RMSNAME,
        psf_path=PSFNAME,
        mask_path=MASKNAME,
        image_size=IMSIZE,
        half_light_radius=1230.3,
        axis_ratio=BA,
    )
