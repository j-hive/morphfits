"""Create J-HIVE MorphFITS GalWrap products for GALFIT usage.
"""

# Imports


import gc
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


logger = logging.getLogger("PRODUCT")
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
    sublogger = logging.getLogger("CUTOUT")

    # Load data
    sublogger.info(f"Loading data from {input_path.name}")
    file = fits.open(input_path)["PRIMARY"]
    image, wcs = file.data, WCS(file.header)

    # Load catalog
    if (catalog is None) and (catalog_path is None):
        sublogger.error("No catalog provided for cutout generation.")
    catalog = Table.read(catalog_path) if catalog is None else catalog

    # Create cutouts and save to file
    if (object is None) and (objects is None):
        sublogger.error("No object provided for cutout generation.")
    skipped = []
    objects = [object] if objects is None else objects

    ## Create cutout for each passed object
    for object in objects:
        sublogger.info(f"Generating cutout for object {object}.")
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
            sublogger.error(
                f"Error generating cutout for object {object}"
                + f" in image {input_path.name}"
            )
            skipped.append(object)
            continue
    if len(skipped) > 0:
        sublogger.debug(f"Skipped making cutouts for {len(skipped)} objects.")


def generate_psf(input_path: Path, product_path: Path, pixscale: float = 0.04):
    sublogger = logging.getLogger("PSF")

    # Open input PSF
    sublogger.info("Loading original PSF.")
    file = fits.open(input_path)["PRIMARY"]
    input_psf, headers = file.data, file.header

    sublogger.info("Calculating new PSF.")

    # Calculate image size from ratio of PSF pixscale to science frame pixscale
    half_image_size = (headers["NAXIS1"] * headers["PIXELSCL"] / pixscale) / 2
    center = headers["NAXIS1"] / 2

    # Cutout square of length image_size centered at PSF center
    psf = input_psf[
        int(center - half_image_size) : int(center + half_image_size),
        int(center - half_image_size) : int(center + half_image_size),
    ]

    # Write to file
    sublogger.info("Writing new PSF to file.")
    fits.PrimaryHDU(data=psf).writeto(product_path)


def generate_sigma(
    exposure_path: Path, science_path: Path, weights_path: Path, product_path: Path
):
    sublogger = logging.getLogger("SIGMA")
    sigma_saves = exposure_path.parent / "sigma_saves"

    # Load weights map, calculate and save weights variance, and free memory
    weight_variance_path = sigma_saves / "sigma_temp_weight_variance.npy"
    if weight_variance_path.exists():
        pass
    else:
        sublogger.info(f"Loading weights map from {weights_path.name}.")
        weights = fits.open(weights_path)["PRIMARY"].data
        sublogger.info("Calculating weights variance.")
        weight_variance = 1 / weights
        np.save(weight_variance_path, weight_variance)
        del weights
        del weight_variance
        gc.collect()

    # Load science frame, calculate and save max variance, and free memory
    max_variance_path = sigma_saves / "sigma_temp_max_variance.npy"
    if max_variance_path.exists():
        pass
    else:
        sublogger.info(f"Loading science frame from {science_path.name}.")
        science = fits.open(science_path)["PRIMARY"].data
        sublogger.info("Calculating max variance.")
        max_variance = np.maximum(science.data, 0)
        np.save(max_variance_path, max_variance)
        del max_variance
        del science
        gc.collect()

    # Calculate multiplicative factors and purge headers memory
    variables_path = sigma_saves / "sigma_temp_variables.npy"
    if variables_path.exists():
        sublogger.info(f"Loading variables from {variables_path.name}.")
        variables = np.load(variables_path)
        phot_scale = variables[0]
        science_shape = [int(variables[1]), int(variables[2])]
    else:
        sublogger.info("Calculating variables.")
        headers = fits.open(exposure_path)["PRIMARY"].header
        phot_scale = 1.0 / headers["PHOTMJSR"] / headers["PHOTSCAL"]
        if "OPHOTFNU" in headers:
            phot_scale *= headers["PHOTFNU"] / headers["OPHOTFNU"]
        del headers
        gc.collect()
        headers = fits.open(science_path)["PRIMARY"].header
        science_shape = [headers["NAXIS2"], headers["NAXIS1"]]
        del headers
        gc.collect()
        np.save(
            variables_path, np.array([phot_scale, science_shape[0], science_shape[1]])
        )

    # Load exposure frame and grow exposure map to original frame
    full_exposure_path = sigma_saves / "sigma_temp_full_exposure.npy"
    if full_exposure_path.exists():
        sublogger.info(f"Loading full exposure map from {full_exposure_path.name}.")
        full_exposure_cleared = np.load(full_exposure_path)
    else:
        sublogger.info(f"Loading exposure map from {exposure_path.name}.")
        exposure = fits.open(exposure_path)["PRIMARY"].data
        sublogger.info("Growing exposure map to original frame size.")
        full_exposure = np.zeros(science_shape, dtype=int)
        full_exposure[2::4, 2::4] += exposure * 1
        del science_shape
        del exposure
        gc.collect()
        sublogger.info("Filtering exposure map with ndimage.")
        full_exposure_cleared = nd.maximum_filter(full_exposure, 4)
        del full_exposure
        gc.collect()
        np.save(full_exposure_path, full_exposure_cleared)

    # Calculate effective gain and free memory
    effective_gain_path = sigma_saves / "sigma_temp_effective_gain.npy"
    if effective_gain_path.exists():
        del full_exposure_cleared
        gc.collect()
        sublogger.info(f"Loading effective gain from {effective_gain_path.name}.")
        effective_gain = np.load(effective_gain_path)
    else:
        sublogger.info("Calculating effective gain. This may take 20+ minutes.")
        # Separate into quarters
        iterations = 200
        exposure_size = int(full_exposure_cleared.shape[0] / iterations)
        for i in tqdm(range(iterations)):
            division_path = sigma_saves / f"sigma_temp_effective_gain_{i}.npy"
            if division_path.exists():
                continue
            effective_gain_division = full_exposure_cleared[
                i * exposure_size : (i + 1) * exposure_size
            ].astype(np.float64)
            effective_gain_division *= phot_scale
            np.save(division_path, effective_gain_division)
            del effective_gain_division
            gc.collect()
        del phot_scale
        del full_exposure_cleared
        gc.collect()
        effective_gain = np.load(sigma_saves / "sigma_temp_effective_gain_0.npy")
        for i in range(iterations):
            effective_gain = np.append(
                effective_gain,
                np.load(sigma_saves / f"sigma_temp_effective_gain_{i}.npy"),
                0,
            )
        np.save(effective_gain_path, effective_gain)

    # Calculate poisson variance
    poisson_variance_path = sigma_saves / "sigma_temp_poisson_variance.npy"
    if poisson_variance_path.exists():
        del effective_gain
        gc.collect()
        sublogger.info(f"Loading Poisson variance from {poisson_variance_path.name}.")
        max_variance = np.load(poisson_variance_path)
    else:
        sublogger.info("Calculating Poisson variance.")
        max_variance = np.load(max_variance_path)
        max_variance /= effective_gain
        del effective_gain
        gc.collect()
        np.save(poisson_variance_path, max_variance)

    # Calculate total variance
    total_variance_path = sigma_saves / "sigma_temp_total_variance.npy"
    if total_variance_path.exists():
        del max_variance
        gc.collect()
        sublogger.info(f"Loading total variance from {total_variance_path.name}.")
        weight_variance = np.load(total_variance_path)
    else:
        sublogger.info("Calculating total variance.")
        weight_variance = np.load(weight_variance_path)
        weight_variance += max_variance
        del max_variance
        gc.collect()
        np.save(total_variance_path, weight_variance)

    # Calculate sigma and write to file
    sublogger.info("Calculating sigma map and writing to file.")
    sigma = np.sqrt(total_variance)
    del total_variance
    gc.collect()
    fits.PrimaryHDU(data=sigma).writeto(product_path)
    del sigma
    gc.collect()

    # Delete saves
    sublogger.info("Deleting temporary save files.")
    for save_file in paths.get_files(sigma_saves):
        save_file.unlink()
    sigma_saves.rmdir()


def generate_products(
    ficlo: FICLO, galwrap_config: GalWrapConfig, product_root: Path | None = None
):
    # Load in catalog
    catalog_path = paths.get_path(
        "catalog", galwrap_config=galwrap_config, product_root=product_root, ficlo=ficlo
    )
    logger.info(f"Loading in catalog from {catalog_path.name}")
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
        logger.info("Generating stamp cutout. This may take a while.")
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
        logger.info(f"Generating mask cutout from {segmap_path.name}")
        generate_cutout(
            input_path=segmap_path,
            product_path=mask_path,
            catalog=catalog,
            object=ficlo.object,
        )

    # Generate PSF if it does not exist
    psf_path = paths.get_path(
        "psf", galwrap_config=galwrap_config, product_root=product_root, ficlo=ficlo
    )
    if not psf_path.exists():
        input_psf_path = paths.get_path(
            "rawpsf",
            galwrap_config=galwrap_config,
            product_root=product_root,
            ficlo=ficlo,
        )
        logger.info(f"Generating PSF cutout from {input_psf_path.name}")
        generate_psf(
            input_path=input_psf_path,
            product_path=psf_path,
            # TODO find the pixscale
            # pixscale=fits.open(stamp_path)["PRIMARY"].header["PIXSCALE"]
        )

    # Generate full sigma map if it does not exist
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
        logger.info("Generating sigma map. This may take a while.")
        generate_sigma(
            exposure_path=exposure_path,
            science_path=science_path,
            weights_path=weights_path,
            product_path=full_sigma_path,
        )

    # Generate sigma cutout if it does not exist
    sigma_path = paths.get_path(
        "sigma", galwrap_config=galwrap_config, product_root=product_root, ficlo=ficlo
    )
    if not sigma_path.exists():
        logger.info(f"Generating sigma map cutout from {sigma_path.name}")
        generate_cutout(
            input_path=sigma_path,
            product_path=sigma_path,
            catalog=catalog,
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

        logger.info(f"Generating feedfile to {feedfile_path.name}")
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
