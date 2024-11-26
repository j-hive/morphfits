"""Make products, run morphology fitting, and other operations for the
morphology pipeline using GALFIT.
"""

# Imports


import logging
import shutil, os
from subprocess import Popen, PIPE, STDOUT
from pathlib import Path

from astropy.table import Table
from jinja2 import Template
from tqdm import tqdm

from .. import catalog, settings, DATA_ROOT
from ..settings import RuntimeSettings, ScienceSettings
from ..utils import science


# Constants


logger = logging.getLogger("GALWRAP")
"""Logging object for this module.
"""


GALFIT_DATA_ROOT = DATA_ROOT / "galfit"
"""Path to root directory of GALFIT data standards.
"""


GALWRAP_OUTPUT_START = "GALWRAP OUTPUT:\n"
GALWRAP_OUTPUT_END = "RETURN CODE: "
"""Strings to append to the start and end of a GALFIT output log file.
"""


FEEDFILE_COMMENT_COLUMN = 64
"""Column at which comments begin in the feedfile.
"""


FEEDFILE_FLOAT_LENGTH = 12
"""Number of characters in str representing float in a feedfile.
"""


FEEDFILE_TEMPLATE_PATH = GALFIT_DATA_ROOT / "feedfile.jinja"
DEFAULT_CONSTRAINTS_PATH = GALFIT_DATA_ROOT / "default.constraints"
"""Paths to feedfile template and default constraints.
"""


NUM_FITS_TO_MONITOR = 100
"""Number of fits after which an append statement will be made to the temporary
catalog file in the run directory.
"""


# Functions


## Object Level


def make_feedfile(
    path: Path,
    stamp_path: Path,
    model_galfit_path: Path,
    sigma_path: Path,
    psf_path: Path,
    mask_path: Path,
    constraints_path: Path,
    image_size: int,
    zeropoint: float,
    pixscale: tuple[float, float],
    magnitude: float,
    half_light_radius: float,
    axis_ratio: float,
):
    """Write a GALFIT feedfile for a single object.

    Parameters
    ----------
    path : Path
        Path to which to write feedfile.
    stamp_path : Path
        Path to stamp for this object.
    model_galfit_path : Path
        Path to which to output model file for this object.
    sigma_path : Path
        Path to sigma map for this object.
    psf_path : Path
        Path to PSF crop for this object.
    mask_path : Path
        Path to mask for this object.
    constraints_path : Path
        Path to constraints file for this run.
    image_size : int
        Number of pixels along each axis of square image.
    zeropoint : float
        Magnitude zeropoint for this object.
    pixscale : tuple[float, float]
        Pixel scale along each axis for this object, in "/pix.
    magnitude : float
        Magnitude for this object.
    half_light_radius : float
        Half light radius for this object.
    axis_ratio : float
        Axis ratio of this object.
    """
    # Define functions for feedfile column alignment
    path_str = lambda x: str(x).ljust(FEEDFILE_COMMENT_COLUMN)
    float_str = lambda x: str(x).ljust(FEEDFILE_FLOAT_LENGTH)[:FEEDFILE_FLOAT_LENGTH]

    # Set feedfile values from parameters
    # Note paths are all relative to ficlo_products
    feedfile_dict = {
        "stamp_path": path_str(stamp_path.name),
        "output_galfit_path": path_str(model_galfit_path.name),
        "sigma_path": path_str(sigma_path.name),
        "psf_path": path_str(psf_path.name),
        "mask_path": path_str(mask_path.name),
        "constraints_path": path_str(constraints_path),
        "image_size": str(image_size),
        "zeropoint": float_str(zeropoint),
        "pixscale_x": float_str(pixscale[0]),
        "pixscale_y": float_str(pixscale[1]),
        "position": str(image_size / 2),
        "magnitude": float_str(magnitude),
        "half_light_radius": float_str(half_light_radius),
        "axis_ratio": float_str(axis_ratio),
    }

    # Write new feedfile from template and save to output directory
    with open(FEEDFILE_TEMPLATE_PATH, "r") as feedfile_template:
        template = Template(feedfile_template.read())
    lines = template.render(feedfile_dict)
    with open(path, "w") as feedfile:
        feedfile.write(lines)


def run(
    galfit_path: Path,
    product_ficlo_path: Path,
    constraints_path: Path,
    feedfile_path: Path,
    galfit_log_path: Path,
    galfit_model_path: Path,
):
    """Execute the GALFIT binary program on a single object.

    Parameters
    ----------
    galfit_path : Path
        Path to GALFIT binary executable file.
    product_ficlo_path : Path
        Path to product directory for this FICLO.
    constraints_path : Path
        Path to constraints file for this run.
    feedfile_path : Path
        Path to feedfile for this object.
    galfit_log_path : Path
        Path to GALFIT log for this object.
    galfit_model_path : Path
        Path to output model for this object.

    Raises
    ------
    RuntimeError
        GALFIT exits on failure.
    """
    # Copy GALFIT binary executable and constraints file to FICLO directory
    # Due to GALFIT needing to be in the same directory as the feedfile and
    # where its paths are based from
    try:
        os.symlink(galfit_path, product_ficlo_path / "galfit")
    except:
        pass
    try:
        os.symlink(constraints_path, product_ficlo_path / ".constraints")
    except:
        pass

    # Run GALFIT via subprocess
    process = Popen(
        f"cd {str(product_ficlo_path)} && ./galfit {feedfile_path.name}",
        stdout=PIPE,
        shell=True,
        stderr=STDOUT,
        close_fds=True,
    )

    # Capture GALFIT output and close subprocess
    galfit_lines = []
    for line in iter(process.stdout.readline, b""):
        galfit_lines.append(line.rstrip().decode("utf-8"))
    process.stdout.close()
    process.wait()
    return_code = process.returncode

    # Write captured output to GALFIT log file
    with open(galfit_log_path, mode="w") as galfit_log_file:
        galfit_log_file.write(GALWRAP_OUTPUT_START)
        for line in galfit_lines:
            galfit_log_file.write(line + "\n")

    # Clean up files output by GALFIT in product FICLO directory
    for item_path in product_ficlo_path.iterdir():
        # Move model to output directory
        if "galfit.fits" in item_path.name:
            item_path.rename(galfit_model_path)

        # Move logs to output directory
        elif "log" in item_path.name:
            summary = []
            with open(item_path, mode="r") as summary_file:
                for line in summary_file.readlines():
                    summary.append(line.strip())
            with open(galfit_log_path, mode="a") as galfit_log_file:
                for line in summary:
                    galfit_log_file.write(line + "\n")
            item_path.unlink()

        # Remove output feedfiles
        elif "galfit." in item_path.name:
            item_path.unlink()

    # Remove binary and constraints files
    (product_ficlo_path / "galfit").unlink()
    (product_ficlo_path / ".constraints").unlink()

    # Write return code to end of GALFIT log file
    with open(galfit_log_path, mode="a") as galfit_log_file:
        galfit_log_file.write(GALWRAP_OUTPUT_END + str(return_code))

    # Raise error if GALFIT did not return successful
    if return_code != 0:
        raise RuntimeError(f"return code {return_code}")


## FICL Level


def make_all_feedfiles(
    runtime_settings: RuntimeSettings, science_settings: ScienceSettings
):
    """Make all GALFIT feedfiles for a program run.

    Parameters
    ----------
    runtime_settings : RuntimeSettings
        Settings for this program run.
    science_settings : ScienceSettings
        Science configurations for this program run.
    """
    # Iterate over each FICL in this run
    for ficl in runtime_settings.ficls:
        # Try to open required files for FICL
        try:
            logger.info(f"FICL {ficl}: Making feedfiles.")

            # Open input catalog
            input_catalog_path = settings.get_path(
                name="input_catalog", path_settings=runtime_settings.roots, ficl=ficl
            )
            input_catalog = Table.read(input_catalog_path)

            # Open science frame
            science_path = settings.get_path(
                name="science", path_settings=runtime_settings.roots, ficl=ficl
            )
            science_image, science_headers = science.get_fits_data(science_path)
            zeropoint = science.get_zeropoint(headers=science_headers)

            # Get iterable object list, displaying progress bar if flagged
            if runtime_settings.progress_bar:
                objects = tqdm(iterable=ficl.objects, unit="obj", leave=False)
            else:
                objects = ficl.objects

        # Catch any error opening FICL or input catalog
        except Exception as e:
            logger.error(f"FICL {ficl}: Skipping feedfiles - failed loading input.")
            logger.error(e)
            continue

        # Iterate over each object
        skipped = 0
        for object in objects:
            # Try making feedfile for object
            try:
                # Get path to feedfile
                feedfile_path = settings.get_path(
                    name="feedfile",
                    path_settings=runtime_settings.roots,
                    ficl=ficl,
                    object=object,
                )

                # Skip existing feedfiles unless requested
                if feedfile_path.exists() and not runtime_settings.remake.others:
                    if not runtime_settings.progress_bar:
                        logger.debug(f"Object {object}: Skipping feedfile - exists.")
                    skipped += 1
                    continue

                # Get paths to products and output
                stamp_path = settings.get_path(
                    name="stamp",
                    path_settings=runtime_settings.roots,
                    ficl=ficl,
                    object=object,
                )
                model_galfit_path = settings.get_path(
                    name="model_galfit",
                    path_settings=runtime_settings.roots,
                    ficl=ficl,
                    object=object,
                )
                sigma_path = settings.get_path(
                    name="sigma",
                    path_settings=runtime_settings.roots,
                    ficl=ficl,
                    object=object,
                )
                psf_path = settings.get_path(
                    name="psf",
                    path_settings=runtime_settings.roots,
                    ficl=ficl,
                    object=object,
                    resolve=False,
                )
                mask_path = settings.get_path(
                    name="mask",
                    path_settings=runtime_settings.roots,
                    ficl=ficl,
                    object=object,
                )
                constraints_path = DEFAULT_CONSTRAINTS_PATH

                # Skip objects with missing products
                if (
                    (not stamp_path.exists())
                    or (not sigma_path.exists())
                    or (not psf_path.exists())
                    or (not mask_path.exists())
                ):
                    if not runtime_settings.progress_bar:
                        logger.debug(
                            f"Object {object}: Skipping feedfile - missing products."
                        )
                    skipped += 1
                    continue

                # Get science details for object
                stamp_path = settings.get_path(
                    name="stamp",
                    path_settings=runtime_settings.roots,
                    ficl=ficl,
                    object=object,
                )
                stamp_image, stamp_headers = science.get_fits_data(stamp_path)
                kron_radius = science.get_kron_radius(
                    input_catalog=input_catalog,
                    catalog_version=ficl.catalog_version,
                    object=object,
                )
                image_size = science.get_image_size(
                    radius=kron_radius, scale=science_settings.scale
                )
                magnitude = science.get_surface_brightness_from_headers(
                    runtime_settings=runtime_settings, headers=stamp_headers
                )
                half_light_radius = science.get_half_light_radius(
                    input_catalog=input_catalog, object=object
                )
                axis_ratio = science.get_axis_ratio(
                    input_catalog=input_catalog, object=object
                )

                # Apply boost to magnitude estimate
                magnitude -= science_settings.morphology.boost

                # Make feedfile for object
                make_feedfile(
                    path=feedfile_path,
                    stamp_path=stamp_path,
                    model_galfit_path=model_galfit_path,
                    sigma_path=sigma_path,
                    psf_path=psf_path,
                    mask_path=mask_path,
                    constraints_path=constraints_path,
                    image_size=image_size,
                    zeropoint=zeropoint,
                    pixscale=ficl.pixscale,
                    magnitude=magnitude,
                    half_light_radius=half_light_radius,
                    axis_ratio=axis_ratio,
                )

            # Catch any errors making feedfile for object and skip to next
            except Exception as e:
                if not runtime_settings.progress_bar:
                    logger.debug(f"Object {object}: Skipping feedfile - {e}.")
                skipped += 1
                continue

        # Log number of skipped or failed objects
        logger.info(
            f"FICL {ficl}: Made feedfiles - skipped {skipped}/{len(objects)} objects."
        )


def run_all(runtime_settings: RuntimeSettings):
    """Run GALFIT on all FICLOs in this program run.

    Updates to the temporary catalog under the run directory every n number of
    fits, where n is set by morphfits.wrappers.galfit.NUM_FITS_TO_MONITOR.

    Parameters
    ----------
    runtime_settings : RuntimeSettings
        Settings for this program run.
    """
    # Iterate over each FICL in this run
    for ficl in runtime_settings.ficls:
        # Try to get objects from FICL
        try:
            # Skip if FICL has no objects
            if len(ficl.objects) == 0:
                logger.warning(f"FICL {ficl}: Skipping GALFIT - no objects to fit.")
                continue

            # Log progress
            logger.info(f"FICL {ficl}: Running GALFIT.")
            logger.info(
                f"Objects: {min(ficl.objects)} to {max(ficl.objects)} "
                + f"({len(ficl.objects)} objects)."
            )

            # Get iterable object list, displaying progress bar if flagged
            if runtime_settings.progress_bar:
                objects = tqdm(iterable=ficl.objects, unit="obj", leave=False)
            else:
                objects = ficl.objects

            # Track objects fitted
            fitted_objects = []

        # Catch any error opening FICL
        except Exception as e:
            logger.error(f"FICL {ficl}: Skipping GALFIT - {e}.")
            continue

        # Iterate over each object
        skipped = 0
        for object in objects:
            # Try running GALFIT for object
            try:
                # Get path to model
                model_path = settings.get_path(
                    name="model_galfit",
                    path_settings=runtime_settings.roots,
                    ficl=ficl,
                    object=object,
                )

                # Skip previously fitted objects unless requested
                if model_path.exists() and not runtime_settings.remake.morphology:
                    if not runtime_settings.progress_bar:
                        logger.debug(f"Object {object}: Skipping GALFIT - exists.")
                    skipped += 1
                    fitted_objects.append(object)
                    continue

                # Get path to feedfile
                feedfile_path = settings.get_path(
                    name="feedfile",
                    path_settings=runtime_settings.roots,
                    ficl=ficl,
                    object=object,
                )

                # Skip objects missing feedfiles
                if not feedfile_path.exists():
                    if not runtime_settings.progress_bar:
                        logger.debug(
                            f"Object {object}: Skipping GALFIT - missing feedfile."
                        )
                    skipped += 1
                    continue

                # Get paths to product FICLO directory and GALFIT log file
                product_ficlo_path = settings.get_path(
                    name="product_ficlo",
                    path_settings=runtime_settings.roots,
                    ficl=ficl,
                    object=object,
                )
                constraints_path = DEFAULT_CONSTRAINTS_PATH
                galfit_log_path = settings.get_path(
                    name="log_galfit",
                    path_settings=runtime_settings.roots,
                    ficl=ficl,
                    object=object,
                )

                # Run GALFIT for object
                if not runtime_settings.progress_bar:
                    logger.debug(f"Object {object}: Running GALFIT.")
                run(
                    galfit_path=runtime_settings.morphology.binary,
                    product_ficlo_path=product_ficlo_path,
                    constraints_path=constraints_path,
                    feedfile_path=feedfile_path,
                    galfit_log_path=galfit_log_path,
                    galfit_model_path=model_path,
                )

                # Track object if successfully fitted
                fitted_objects.append(object)

                # Update temporary catalog every certain number of fitted
                # objects
                if (len(fitted_objects) % NUM_FITS_TO_MONITOR == 0) or (
                    len(fitted_objects) == len(objects)
                ):
                    catalog.update_temporary(
                        runtime_settings=runtime_settings,
                        ficl=ficl,
                        objects=fitted_objects[-NUM_FITS_TO_MONITOR:],
                    )

            # Catch any errors and skip to next object
            except Exception as e:
                if not runtime_settings.progress_bar:
                    logger.debug(f"Object {object}: Skipping GALFIT - {e}.")
                skipped += 1
                continue

        # Log number of skipped or failed objects
        logger.info(
            f"FICL {ficl}: Ran GALFIT - skipped {skipped}/{len(objects)} objects."
        )

    # Remove temporary catalog
    try:
        temp_catalog_path = settings.get_path(
            name="run_catalog",
            runtime_settings=runtime_settings,
            field=runtime_settings.ficls[0].field,
        )
        if temp_catalog_path.exists():
            temp_catalog_path.unlink()
    except Exception as e:
        logger.error(f"Skipping removing temporary catalog - {e}.")
