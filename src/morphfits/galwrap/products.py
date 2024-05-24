"""Create J-HIVE MorphFITS GalWrap products for GALFIT usage.
"""

# Imports


import gc
import logging
from pathlib import Path
from typing import Any

import numpy as np
import scipy.ndimage as nd
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata.utils import Cutout2D
from astropy.table import Table
from astropy import wcs
import fitsio
from jinja2 import Template
from tqdm import tqdm

from . import paths, GALWRAP_DATA_ROOT
from .setup import FICLO, GalWrapConfig
from ..utils import science


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
    feedfile_template_path : str | Path, optional
        Path to GALFIT feedfile template, by default the template in the data
        directory.
    constraints_path : str | Path, optional
        Path to GALFIT parameter constraints file, by default the default
        constraints in the data directory.
    """

    def get_aligned_variable(variable: Any, align_size: int = 240) -> str:
        """Return a left-justified string of a variable, for comment alignment.

        Parameters
        ----------
        variable : Any
            Jinja variable to be left-justified.
        align_size : int, optional
            Number of space characters to left-justify variable with, i.e.
            length of str including empty characters, by default 240, i.e.
            comments begin on column 244. This length is to accommodate long
            paths.

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


def generate_stamp(
    input_root: Path,
    product_root: Path,
    field: str,
    image_version: str,
    catalog_version: str,
    filter: str,
    object: int,
    ra:float,
    dec:float,
    kron_radius:float,
    minimum_size: int = 32,
):
    sublogger = logging.getLogger("CUTOUT")

    # Calculate coordinates
    position = SkyCoord(ra=ra,dec=dec,unit="deg")
    image_size = 

    # Load in image and header data
    sci_path = paths.get_path(
        "science",
        input_root=input_root,
        field=field,
        image_version=image_version,
        filter=filter,
    )
    sublogger.info(f"Loading data from {sci_path.name}.")
    stamp = fitsio.read(sci_path, rows=[])


def generate_cutout(
    input_path: Path,
    product_path: Path,
    catalog: Table | None = None,
    catalog_path: Path | None = None,
    object: int | None = None,
    objects: list[int] | None = None,
    minimum_image_size: int = 32,
    kron_factor: int = 3,
):
    sublogger = logging.getLogger("CUTOUT")

    # Load in image and header data (memory expensive)
    sublogger.info(f"Loading data from {input_path.name}")
    file = fits.open(input_path)
    image, wcs = file["PRIMARY"].data, WCS(file["PRIMARY"].header)
    file.close()
    del file
    gc.collect()

    # Load in catalog
    if (catalog is None) and (catalog_path is None):
        sublogger.error("No catalog provided for cutout generation.")
    catalog = Table.read(catalog_path) if catalog is None else catalog

    # Check at least one object is specified
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
                        int(kron_radius / 0.04 * kron_factor),
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
                sublogger.info(
                    f"Writing object {object} cutout to {product_path.name}."
                )
                fits.PrimaryHDU(
                    data=cutout.data, header=cutout.wcs.to_header()
                ).writeto(product_path, overwrite=True)

            # Clear memory
            del position
            del cutout
            gc.collect()

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

    # Clear memory
    del image
    del wcs
    gc.collect()


def generate_psf(input_path: Path, product_path: Path, pixscale: float = 0.04):
    sublogger = logging.getLogger("PSF")

    # Open input PSF
    sublogger.info("Loading original PSF.")
    file = fits.open(input_path)
    input_psf, headers = file["PRIMARY"].data, file["PRIMARY"].header

    # Close file and clear from memory
    file.close()
    del file
    gc.collect()

    # Calculate image size from ratio of PSF pixscale to science frame pixscale
    half_image_size = (headers["NAXIS1"] * headers["PIXELSCL"] / pixscale / 5) / 2
    center = headers["NAXIS1"] / 2

    # Cutout square of length image_size centered at PSF center
    sublogger.info("Calculating new PSF.")
    psf = input_psf[
        int(center - half_image_size) : int(center + half_image_size),
        int(center - half_image_size) : int(center + half_image_size),
    ]

    # Write to file
    sublogger.info("Writing new PSF to file.")
    fits.PrimaryHDU(data=psf).writeto(product_path, overwrite=True)

    # Clear memory
    del psf
    gc.collect()


def generate_sigma(
    input_root: Path,
    product_root: Path,
    field: str,
    image_version: str,
    catalog_version: str,
    filter: str,
):
    # Setup
    sublogger = logging.getLogger("SIGMA")
    exposure_path = paths.get_path(
        "exposure",
        input_root=input_root,
        field=field,
        image_version=image_version,
        filter=filter,
    )
    science_path = paths.get_path(
        "science",
        input_root=input_root,
        field=field,
        image_version=image_version,
        filter=filter,
    )
    weights_path = paths.get_path(
        "weight",
        input_root=input_root,
        field=field,
        image_version=image_version,
        filter=filter,
    )
    full_sigma_path = paths.get_path(
        "fullsigma",
        product_root=product_root,
        field=field,
        image_version=image_version,
        catalog_version=catalog_version,
    )
    sigma_saves_path = exposure_path.parent / "sigma_saves"

    # Load weights map, calculate and save weights variance, and free memory
    weight_variance_path = (
        paths.get_path(
            "input_images",
            input_root=input_root,
            field=field,
            image_version=image_version,
            filter=filter,
        )
        / "sigma_temp_weight_variance.npy"
    )
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
        # Separate into smaller iterations
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
        for i in tqdm(range(iterations)):
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
        total_variance = np.load(total_variance_path)
    else:
        sublogger.info("Calculating total variance.")
        total_variance = np.load(weight_variance_path)
        total_variance += max_variance
        del max_variance
        gc.collect()
        np.save(total_variance_path, weight_variance)

    # Calculate sigma and write to file
    sublogger.info("Calculating sigma map and writing to file.")
    sigma = np.sqrt(total_variance)
    del total_variance
    gc.collect()
    fits.PrimaryHDU(data=sigma).writeto(full_sigma_path, overwrite=True)
    del sigma
    gc.collect()

    # Delete saves
    sublogger.info("Deleting temporary save files.")
    for save_file in paths.get_files(sigma_saves):
        save_file.unlink()
    sigma_saves.rmdir()



def generate_stamp(ficlo:FICLO,galwrap_config:GalWrapConfig,position:SkyCoord,image_size:int):
    pass


def generate_products(
    ficlo: FICLO,
    galwrap_config: GalWrapConfig,
    regenerate: bool = False,
    regenerate_stamp: bool = False,
    regenerate_psf: bool = False,
    regenerate_mask: bool = False,
    regenerate_sigma: bool = False,
    regenerate_feedfile: bool = True,
    use_mask: bool = True,
    use_psf: bool = True,
    use_sigma: bool = True,
    minimum_image_size:int=32,
):
    # Get paths
    ## Input
    catalog_path = paths.get_path("catalog", galwrap_config=galwrap_config,ficlo=ficlo)
    rawpsf_path = paths.get_path("rawpsf", galwrap_config=galwrap_config,ficlo=ficlo)
    segmap_path = paths.get_path("segmap", galwrap_config=galwrap_config,ficlo=ficlo)
    exposure_path = paths.get_path("exposure", galwrap_config=galwrap_config,ficlo=ficlo)
    science_path = paths.get_path("science", galwrap_config=galwrap_config,ficlo=ficlo)
    weights_path = paths.get_path("weights", galwrap_config=galwrap_config,ficlo=ficlo)

    ## Product
    stamp_path = paths.get_path("stamp", galwrap_config=galwrap_config,ficlo=ficlo)
    psf_path = paths.get_path("psf", galwrap_config=galwrap_config,ficlo=ficlo)
    mask_path = paths.get_path("mask", galwrap_config=galwrap_config,ficlo=ficlo)
    sigma_path = paths.get_path("sigma", galwrap_config=galwrap_config,ficlo=ficlo)
    feedfile_path = paths.get_path("feedfile", galwrap_config=galwrap_config,ficlo=ficlo)

    ## Output
    output_galfit_path = paths.get_path("output_galfit", galwrap_config=galwrap_config,ficlo=ficlo)

    # Determine which products to generate
    make_stamp = (not stamp_path.exists()) or regenerate or regenerate_stamp
    make_psf = (not psf_path.exists()) or regenerate or regenerate_psf
    make_mask = (not mask_path.exists()) or regenerate or regenerate_mask
    make_sigma = (not sigma_path.exists()) or regenerate or regenerate_sigma
    make_feedfile = (not feedfile_path.exists()) or regenerate or regenerate_feedfile

    # Load in catalog
    if make_stamp or make_mask or make_sigma or make_feedfile:
        logger.info(f"Loading in catalog from {catalog_path.name}")
        catalog = Table.read(catalog_path)

        # Calculate cutout location information
        position = SkyCoord(ra=catalog[ficlo.object]["ra"],dec=catalog[ficlo.object]["dec"],unit="deg")
        kron_factor = 3
        pixscale = 0.04
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
                        int(kron_radius / pixscale * kron_factor),
                        minimum_image_size,
                    ]
                )
                if isinstance(kron_radius, float)
                else minimum_image_size
            )
    
    # Generate stamp if missing or requested
    if make_stamp:
        logger.info(f"Generating stamp for {ficlo}.")
        generate_stamp(ficlo=ficlo,galwrap_config=galwrap_config,position=position,image_size=image_size)
    
    # Generate psf if missing or requested
    if make_psf:
        logger.info(f"Generating PSF for {ficlo.filter}.")
        generate_psf(ficlo=ficlo,galwrap_config=galwrap_config,position=position,image_size=image_size)
    
    # Generate mask if missing or requested
    if make_mask:
        logger.info(f"Generating mask for {ficlo}.")
        generate_mask(ficlo=ficlo,galwrap_config=galwrap_config,position=position,image_size=image_size)
    
    # Generate sigma if missing or requested
    if make_sigma:
        logger.info(f"Generating sigma for {ficlo}.")
        generate_sigma(ficlo=ficlo,galwrap_config=galwrap_config,position=position,image_size=image_size)
    
    # Generate feedfile if missing or requested
    if make_feedfile:
        logger.info(f"Generating feedfile for {ficlo}.")
        half_light_radius = catalog[ficlo.object]["a_image"]
        axis_ratio = catalog[ficlo.object]["b_image"] / catalog[ficlo.object]["a_image"]
        magnitude = catalog[ficlo.object]["mag_auto"]
        zeropoint = science.get_zeropoint(image_path=science_path)
        generate_feedfile(
            feedfile_path=feedfile_path,
            galfit_output_path=output_galfit_path,
            stamp_path=stamp_path,
            rms_path=sigma_path if use_sigma else "",
            psf_path=psf_path if use_psf else "",
            mask_path=mask_path if use_mask else "",
            image_size=image_size,
            half_light_radius=half_light_radius,
            axis_ratio=axis_ratio,
            magnitude=magnitude,
            zeropoint=zeropoint,
        )