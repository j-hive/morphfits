"""Create J-HIVE MorphFITS GalWrap products for GALFIT usage.
"""

# Imports


import logging
from pathlib import Path
from typing import Any

from astropy.io import fits
from jinja2 import Template

from . import paths, GALWRAP_DATA_ROOT
from .setup import FICLO, GalWrapConfig
from ..utils import science


from pathlib import Path

import typer

from astropy.coordinates import SkyCoord
from astropy.nddata.utils import Cutout2D
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from tqdm import tqdm
import scipy.ndimage as nd

import numpy as np
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib import colors as c


# Constants


logger = logging.getLogger("PRODUCTS")
"""Logger object for this module.
"""


# Functions


def generate_feedfile(
    feedfile_path: str | Path,
    galfit_output_path: str | Path,
    stamp_path: str | Path,
    rms_path: str | Path,
    psf_path: str | Path,
    mask_path: str | Path,
    image_size: int,
    half_light_radius: float,
    axis_ratio: float,
    magnitude: float = 30.0,
    zeropoint: float = 31.50,
    feedfile_template_path: str | Path = GALWRAP_DATA_ROOT / "feedfile.jinja",
    constraints_path: str | Path = GALWRAP_DATA_ROOT / "default.constraints",
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
    with open(feedfile_template_path, "r") as feedfile_template:
        template = Template(feedfile_template.read())
    lines = template.render(feedfile_dict)
    with open(feedfile_path, "w") as feedfile:
        feedfile.write(lines)


def generate_cutout(
    input_path: Path,
    product_path: Path,
    catalog: Table | None = None,
    catalog_path: Path | None = None,
    object: int | None = None,
    objects: list[int] | None = None,
    minimum_image_size: int = 32,
):
    # Load data
    logger.info(f"Loading data from {input_path}")
    file = fits.open(input_path)["PRIMARY"]
    image, wcs = file.data, WCS(file.header)

    # Load catalog
    if (catalog is None) and (catalog_path is None):
        logger.error("No catalog provided for cutout generation.")
    catalog = Table.read(catalog_path) if catalog is None else catalog

    # Create cutouts and save to file
    if (object is None) and (objects is None):
        logger.error("No object provided for cutout generation.")
    skipped = []
    objects = [object] if objects is None else objects

    ## Create cutout for each passed object
    for object in objects:
        logger.info(f"Generating cutout for object {object}")
        try:
            # Create object position from catalog
            position = SkyCoord(
                ra=catalog[object]["ra"], dec=catalog[object]["dec"], unit="deg"
            )

            # Determine image size from maximum between Kron radius multiple or
            # minimum image size
            kron_radius = catalog[object][
                (
                    "kron_radius_circ"
                    if "kron_radius_circ" in catalog[object]
                    else "kron_radius"
                )
            ]
            image_size = (
                np.nanmax(
                    [
                        int(kron_radius / 0.04 * 5),
                        minimum_image_size,
                    ]
                )
                if isinstance(kron_radius, float)
                else minimum_image_size
            )

            # Make cutout
            cutout = Cutout2D(data=image, position=position, size=image_size, wcs=wcs)

            # Save cutout if image of correct size and data > 0
            if (np.amax(cutout.data) > 0) and (
                cutout.data.shape == (image_size, image_size)
            ):
                fits.PrimaryHDU(
                    data=cutout.data, header=cutout.wcs.to_header()
                ).writeto(product_path)
        # Catch errors in creating cutouts
        except Exception as e:
            logger.error(
                f"Error generating cutout for object {object}"
                + f" in image {input_path}"
            )
            skipped.append(object)
            continue
    if len(skipped) > 0:
        logger.debug(f"Skipped making cutouts for {len(skipped)} objects: {skipped}")


def generate_psf(input_path: Path, product_path: Path, pixscale: float = 0.04):
    # Open input PSF
    file = fits.open(input_path)["PRIMARY"]
    input_psf, headers = file.data, file.header

    # Calculate image size from ratio of PSF pixscale to science frame pixscale
    half_image_size = (headers["NAXIS1"] * headers["PIXELSCL"] / pixscale) / 2
    center = headers["NAXIS1"] / 2

    # Cutout square of length image_size centered at PSF center
    psf = input_psf[
        int(center - half_image_size) : int(center + half_image_size),
        int(center - half_image_size) : int(center + half_image_size),
    ]

    # Write to file
    fits.PrimaryHDU(data=psf).writeto(product_path)


def generate_sigma(
    exposure_path: Path, science_path: Path, weights_path: Path, product_path: Path
):
    exposure = fits.open(exposure_path)["PRIMARY"]
    science = fits.open(science_path)["PRIMARY"]
    weights = fits.open(weights_path)["PRIMARY"]

    # Grow the exposure map to the original frame
    full_exposure = np.zeros(science.data.shape, dtype=int)
    full_exposure[2::4, 2::4] += exposure.data * 1
    full_exposure = nd.maximum_filter(full_exposure, 4)

    # Calculate multiplicative factors
    headers = exposure.header
    phot_scale = 1.0 / headers["PHOTMJSR"] / headers["PHOTSCAL"]
    if "OPHOTFNU" in headers:
        phot_scale *= headers["PHOTFNU"] / headers["OPHOTFNU"]

    # Calculate effective gain, max variance, and Poisson variance
    effective_gain = phot_scale * full_exposure
    max_variance = np.maximum(science.data, 0)
    poisson_variance = max_variance / effective_gain
    weight_variance = 1 / weights.data
    total_variance = weight_variance + poisson_variance

    # Calculate sigma and write to file
    sigma = np.sqrt(total_variance)
    fits.PrimaryHDU(data=sigma).writeto(product_path)


def generate_products(
    ficlo: FICLO, galwrap_config: GalWrapConfig, product_root: Path | None = None
):
    # Load in catalog
    catalog_path = paths.get_path(
        "catalog", galwrap_config=galwrap_config, product_root=product_root, ficlo=ficlo
    )
    logger.info(f"Loading in catalog from {catalog_path}")
    catalog = Table.read(catalog_path)

    # Generate stamp if it does not exist
    stamp_path = paths.get_path(
        "stamp", galwrap_config=galwrap_config, product_root=product_root, ficlo=ficlo
    )
    if not stamp_path.exists():
        science_path = paths.get_path(
            "science",
            galwrap_config=galwrap_config,
            product_root=product_root,
            ficlo=ficlo,
        )
        logger.info(f"Generating stamp cutout from {science_path}")
        generate_cutout(
            input_path=science_path,
            product_path=stamp_path,
            catalog=catalog,
            object=ficlo.object,
        )
    image_size = fits.open(stamp_path)["PRIMARY"].data.shape[0]

    # Generate mask if it does not exist
    mask_path = paths.get_path(
        "mask", galwrap_config=galwrap_config, product_root=product_root, ficlo=ficlo
    )
    if not mask_path.exists():
        segmap_path = paths.get_path(
            "segmap",
            galwrap_config=galwrap_config,
            product_root=product_root,
            ficlo=ficlo,
        )
        logger.info(f"Generating mask cutout from {segmap_path}")
        generate_cutout(
            input_path=segmap_path,
            product_path=mask_path,
            catalog_path=catalog_path,
            object=ficlo.object,
        )

    # Generate PSF if it does not exist
    psf_path = paths.get_path(
        "psf", galwrap_config=galwrap_config, product_root=product_root, ficlo=ficlo
    )
    if not psf_path.exists():
        input_psf_path = paths.get_path(
            "input_psf",
            galwrap_config=galwrap_config,
            product_root=product_root,
            ficlo=ficlo,
        )
        logger.info(f"Generating PSF cutout from {input_psf_path}")
        generate_psf(
            input_path=input_psf_path,
            product_path=psf_path,
            image_size=image_size,
            # TODO find the pixscale
            # pixscale=fits.open(stamp_path)["PRIMARY"].header["PIXSCALE"]
        )

    full_sigma_path = paths.get_path(
        "fullsigma",
        galwrap_config=galwrap_config,
        product_root=product_root,
        ficlo=ficlo,
    )
    if not full_sigma_path.exists():
        exposure_path = paths.get_path(
            "exposure",
            galwrap_config=galwrap_config,
            product_root=product_root,
            ficlo=ficlo,
        )
        science_path = paths.get_path(
            "science",
            galwrap_config=galwrap_config,
            product_root=product_root,
            ficlo=ficlo,
        )
        weights_path = paths.get_path(
            "weights",
            galwrap_config=galwrap_config,
            product_root=product_root,
            ficlo=ficlo,
        )
        logger.info(f"Generating sigma map from {exposure_path} and {weights_path}")
        generate_sigma(
            exposure_path=exposure_path,
            science_path=science_path,
            weights_path=weights_path,
            product_path=full_sigma_path,
        )

    sigma_path = paths.get_path(
        "sigma", galwrap_config=galwrap_config, product_root=product_root, ficlo=ficlo
    )
    if not sigma_path.exists():
        logger.info(f"Generating sigma map cutout from {sigma_path}")
        generate_cutout(
            input_path=sigma_path,
            product_path=sigma_path,
            catalog_path=catalog_path,
            object=ficlo.object,
        )

    # Generate feedfile if it does not exist
    feedfile_path = paths.get_path(
        "feedfile",
        galwrap_config=galwrap_config,
        product_root=product_root,
        ficlo=ficlo,
    )
    if not feedfile_path.exists():
        output_ficlo_path = paths.get_path(
            "output_ficlo",
            galwrap_config=galwrap_config,
            product_root=product_root,
            ficlo=ficlo,
        )

        # Define parameters from stamp and catalog
        half_light_radius = catalog[ficlo.object]["a_image"]
        axis_ratio = catalog[ficlo.object]["b_image"] / catalog[ficlo.object]["a_image"]
        magnitude = catalog[ficlo.object]["mag_auto"]
        zeropoint = science.get_zeropoint(image_path=stamp_path)

        logger.info(f"Generating feedfile to {feedfile_path}")
        generate_feedfile(
            feedfile_path=feedfile_path,
            galfit_output_path=output_ficlo_path,
            stamp_path=stamp_path,
            rms_path=sigma_path,
            psf_path=psf_path,
            mask_path=mask_path,
            image_size=image_size,
            half_light_radius=half_light_radius,
            axis_ratio=axis_ratio,
            magnitude=magnitude,
            zeropoint=zeropoint,
        )
