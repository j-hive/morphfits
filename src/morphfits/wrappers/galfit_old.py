"""Main program execution for the GALFIT morphology fitter wrapper package.
"""

# Imports


import gc
import logging
import shutil
from subprocess import Popen, PIPE, STDOUT
from pathlib import Path
from datetime import datetime as dt

from astropy.io import fits
from astropy.table import Table
from jinja2 import Template
from tqdm import tqdm

from . import GALFIT_DATA_ROOT
from .. import catalog, plot, products, settings
from ..utils import paths, science


# Constants


logger = logging.getLogger("GALWRAP")
"""Logging object for this module.
"""


# Functions


def generate_feedfiles(
    input_root: Path,
    product_root: Path,
    output_root: Path,
    field: str,
    image_version: str,
    catalog_version: str,
    filter: str,
    objects: list[int],
    image_sizes: list[int],
    pixscale: tuple[float, float],
    regenerate: bool = False,
    feedfile_template_path: Path = GALFIT_DATA_ROOT / "feedfile.jinja",
    constraints_path: Path = GALFIT_DATA_ROOT / "default.constraints",
    path_length: int = 64,
    float_length: int = 12,
    display_progress: bool = False,
):
    """Generate feedfiles for all objects in a FICL.

    Parameters
    ----------
    input_root : Path
        Path to root MorphFITS input directory.
    product_root : Path
        Path to root MorphFITS products directory.
    output_root : Path
        Path to root MorphFITS output directory.
    field : str
        Field of observation.
    image_version : str
        Version of image processing used on observation.
    catalog_version : str
        Version of cataloguing used for field.
    filter : str
        Filter used for observation.
    objects : list[str]
        List of object IDs in catalog for which to generate feedfiles.
    image_sizes : list[int]
        List of image sizes corresponding to each object's stamp.
    pixscale : tuple[float, float]
        Pixel scale along x and y axes, in arcseconds per pixel.
    regenerate : bool, optional
        Regenerate existing feedfiles, by default False.
    feedfile_template_path : Path, optional
        Path to jinja2 feedfile template, by default from the repository data
        directory.
    constraints_path : Path, optional
        Path to the GALFIT constraints file, by default from the repository data
        directory.
    path_length : int, optional
        Length of path strings in the template for comment alignment, by default
        64, so that comments start at column 69.
    float_length : int, optional
        Length of float strings in the template for comment alignment, by
        default 12.
    display_progress : bool, optional
        Display progress on terminal screen, by default False.
    """
    logger.info("Generating feedfiles.")

    # Define functions for comment alignment
    path_str = lambda x: str(x).ljust(path_length)
    float_str = lambda x: str(x).ljust(float_length)[:float_length]

    # Define paths to get for later
    product_path_names = ["model_galfit", "stamp", "sigma", "psf", "mask"]

    # Load in catalog
    input_catalog_path = paths.get_path(
        "input_catalog",
        input_root=input_root,
        field=field,
        image_version=image_version,
    )
    input_catalog = Table.read(input_catalog_path)

    # Get zeropoint
    science_path = paths.get_path(
        "science",
        input_root=input_root,
        field=field,
        image_version=image_version,
        filter=filter,
    )
    zeropoint = science.get_zeropoint(image_path=science_path)

    # Clear memory
    del input_catalog_path
    del science_path
    gc.collect()

    # Iterate over each object, and image_size tuple
    skipped = []
    for i in (
        tqdm(range(len(objects)), unit="feedfile", leave=False)
        if display_progress
        else range(len(objects))
    ):
        object, image_size = objects[i], image_sizes[i]

        # Skip objects which already have feedfiles
        feedfile_path = paths.get_path(
            "feedfile",
            product_root=product_root,
            field=field,
            image_version=image_version,
            catalog_version=catalog_version,
            filter=filter,
            object=object,
        )
        if feedfile_path.exists() and not regenerate:
            if not display_progress:
                logger.debug(f"Skipping object {object}, feedfile exists.")
            skipped.append(object)
            continue

        # Generate feedfile for object
        if not display_progress:
            logger.info(f"Generating feedfile for object {object}.")

        # Get paths
        product_paths = {
            path_name: paths.get_path(
                path_name,
                output_root=output_root,
                product_root=product_root,
                field=field,
                image_version=image_version,
                catalog_version=catalog_version,
                filter=filter,
                object=object,
            )
            for path_name in product_path_names
        }

        # Get constants from catalog
        # magnitude = catalog[object]["mag_auto"]
        half_light_radius = input_catalog[object]["a_image"]
        axis_ratio = input_catalog[object]["b_image"] / input_catalog[object]["a_image"]

        # Get constant from stamp
        stamp_file = fits.open(product_paths["stamp"])
        header = stamp_file["PRIMARY"].header
        magnitude = header["SURFACE_BRIGHTNESS"]

        ## Clear memory
        stamp_file.close()
        del stamp_file
        del header

        # Set configuration parameters from input
        # Note paths are all relative to ficlo_products
        feedfile_dict = {
            "stamp_path": path_str(product_paths["stamp"].name),
            "output_galfit_path": path_str(product_paths["model_galfit"].name),
            "sigma_path": path_str(product_paths["sigma"].name),
            "psf_path": path_str(product_paths["psf"].name),
            "mask_path": path_str(product_paths["mask"].name),
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
        with open(feedfile_template_path, "r") as feedfile_template:
            template = Template(feedfile_template.read())
        lines = template.render(feedfile_dict)
        with open(feedfile_path, "w") as feedfile:
            feedfile.write(lines)

        # Clear memory
        del object
        del image_size
        del feedfile_path
        del product_paths
        del magnitude
        del half_light_radius
        del axis_ratio
        del feedfile_dict
        del template
        del lines
        gc.collect()


def run_galfit(
    galfit_path: Path,
    input_root: Path,
    product_root: Path,
    output_root: Path,
    run_root: Path,
    datetime: dt,
    run_number: int,
    field: str,
    image_version: str,
    catalog_version: str,
    filter: str,
    objects: list[int],
    display_progress: bool = False,
    refit: bool = False,
):
    """Run GALFIT over all objects in a FICL.

    Parameters
    ----------
    galfit_path : Path
        Path to GALFIT binary file.
    input_root : Path
        Path to root MorphFITS input directory.
    product_root : Path
        Path to root MorphFITS products directory.
    output_root : Path
        Path to root MorphFITS output directory.
    run_root : Path
        Path to root MorphFITS runs directory.
    datetime : datetime
        Datetime of start of run.
    run_number : int
        Number of run in runs directory when other runs have the same datetime.
    field : str
        Field of observation.
    image_version : str
        Version of image processing used on observation.
    catalog_version : str
        Version of cataloguing used for field.
    filter : str
        Filter used for observation.
    objects : list[str]
        List of object IDs in catalog for which to generate feedfiles.
    display_progress : bool, optional
        Display progress on terminal screen, by default False.
    refit : bool, optional
        Rerun GALFIT on previously fitted objects, by default False.
    """
    logger.info(
        "Running GALFIT for FICL "
        + "_".join([field, image_version, catalog_version, filter])
        + "."
    )
    logger.info(
        f"Object ID range: {min(objects)} to {max(objects)} "
        + f"({len(objects)} objects)."
    )

    # Iterate over each object in FICL
    for object in (
        tqdm(objects, unit="run", leave=False) if display_progress else objects
    ):
        ## Get model path
        model_path = path.get_path(
            "model_galfit",
            output_root=output_root,
            field=field,
            image_version=image_version,
            catalog_version=catalog_version,
            filter=filter,
            object=object,
        )

        ## Skip previously fitted objects unless otherwise specified
        if (model_path.exists()) and (not refit):
            if not display_progress:
                logger.debug(f"Skipping object {object}, previously fitted.")
            continue

        ## Get feedfile path
        feedfile_path = path.get_path(
            "feedfile",
            product_root=product_root,
            field=field,
            image_version=image_version,
            catalog_version=catalog_version,
            filter=filter,
            object=object,
        )

        ## Skip object if feedfile missing
        if not feedfile_path.exists():
            if not display_progress:
                logger.debug(f"Skipping object {object}, missing feedfile.")
            continue

        ## Copy GALFIT and constraints to FICLO product directory
        ficlo_products_path = path.get_path(
            "product_ficlo",
            product_root=product_root,
            field=field,
            image_version=image_version,
            catalog_version=catalog_version,
            filter=filter,
            object=object,
        )
        shutil.copy(galfit_path, ficlo_products_path / "galfit")
        # constraints_path = GALFIT_DATA_ROOT / "default.constraints"
        # shutil.copy(constraints_path, ficlo_products_path / ".constraints")

        ## Run GALFIT via subprocess and pipe output
        if not display_progress:
            logger.info(f"Running GALFIT for object {object} in filter '{filter}'.")
        process = Popen(
            f"cd {str(ficlo_products_path)} && ./galfit {feedfile_path.name}",
            stdout=PIPE,
            shell=True,
            stderr=STDOUT,
            # bufsize=1,
            close_fds=True,
        )

        ## Capture subprocess output and close subprocess
        galfit_lines = []
        for line in iter(process.stdout.readline, b""):
            galfit_lines.append(line.rstrip().decode("utf-8"))
        process.stdout.close()
        process.wait()
        return_code = process.returncode
        if return_code != 0:
            logger.error(
                f"GALFIT did not run successfully and returned with code {return_code}."
            )

        ## Write captured output to GALFIT log file
        galfit_log_path = path.get_path(
            "log_galfit",
            output_root=output_root,
            field=field,
            image_version=image_version,
            catalog_version=catalog_version,
            filter=filter,
            object=object,
        )
        with open(galfit_log_path, mode="w") as galfit_log_file:
            for line in galfit_lines:
                galfit_log_file.write(line + "\n")

        ## Clean up GALFIT output
        for path in ficlo_products_path.iterdir():
            ### Move model to output directory
            if "galfit.fits" in path.name:
                path.rename(
                    path.get_path(
                        "model_galfit",
                        output_root=output_root,
                        field=field,
                        image_version=image_version,
                        catalog_version=catalog_version,
                        filter=filter,
                        object=object,
                    )
                )
            ### Move logs to output directory
            elif "log" in path.name:
                summary = []
                with open(path, mode="r") as summary_file:
                    for line in summary_file.readlines():
                        summary.append(line.strip())
                with open(galfit_log_path, mode="a") as galfit_log_file:
                    for line in summary:
                        galfit_log_file.write(line + "\n")
                path.unlink()
            ### Remove script, constraints, and feedfile records
            elif ("galfit" in path.name) or ("constraints" in path.name):
                path.unlink()

        ## Write parameters to catalog
        catalog.write(
            output_root=output_root,
            run_root=run_root,
            datetime=datetime,
            run_number=run_number,
            field=field,
            image_version=image_version,
            catalog_version=catalog_version,
            filter=filter,
            object=object,
            return_code=return_code,
        )


## Main


def main(
    morphfits_config: settings.MorphFITSConfig,
    regenerate_products: bool = False,
    regenerate_stamps: bool = False,
    regenerate_psfs: bool = False,
    regenerate_masks: bool = False,
    regenerate_sigmas: bool = False,
    keep_feedfiles: bool = False,
    force_refit: bool = False,
    skip_products: bool = False,
    skip_fits: bool = False,
    make_plots: bool = False,
    display_progress: bool = False,
):
    """Orchestrate GalWrap functions for passed configurations.

    Parameters
    ----------
    morphfits_config : MorphFITSConfig
        Configuration object for this program run.
    regenerate_products : bool, optional
        Regenerate all products, by default False.
    regenerate_stamps : bool, optional
        Regenerate stamps, by default False.
    regenerate_masks : bool, optional
        Regenerate masks, by default False.
    regenerate_psfs : bool, optional
        Regenerate psfs, by default False.
    regenerate_sigmas : bool, optional
        Regenerate sigmas, by default False.
    keep_feedfiles : bool, optional
        Reuse existing feedfiles, by default False.
    force_refit : bool, optional
        Run GALFIT over previously fitted objects, overwriting existing models,
        by default False.
    skip_products : bool, optional
        Skip all product generation, by default False.
    skip_fits : bool, optional
        Skip all morphology fitting via GALFIT, by default False.
    make_plots : bool, optional
        Generate model plots, by default False.
    display_progress : bool, optional
        Display progress as loading bar and suppress logging, by default False.
    """
    logger.info("Starting GalWrap.")

    # Generate products where missing, for each FICLO
    if not skip_products:
        products.generate_products(
            morphfits_config=morphfits_config,
            regenerate_products=regenerate_products,
            regenerate_stamps=regenerate_stamps,
            regenerate_psfs=regenerate_psfs,
            regenerate_masks=regenerate_masks,
            regenerate_sigmas=regenerate_sigmas,
            keep_feedfiles=keep_feedfiles,
            display_progress=display_progress,
        )

    # Run GALFIT and record parameters, for each FICLO
    if not skip_fits:
        for ficl in morphfits_config.ficls:
            run_galfit(
                galfit_path=morphfits_config.galfit_path,
                input_root=morphfits_config.input_root,
                output_root=morphfits_config.output_root,
                product_root=morphfits_config.product_root,
                run_root=morphfits_config.run_root,
                field=ficl.field,
                image_version=ficl.image_version,
                catalog_version=ficl.catalog_version,
                filter=ficl.filter,
                objects=ficl.objects,
                datetime=morphfits_config.datetime,
                run_number=morphfits_config.run_number,
                display_progress=display_progress,
                refit=force_refit,
            )

    # Plot models, for each FICLO
    if make_plots:
        for ficl in morphfits_config.ficls:
            plot.plot_model(
                output_root=morphfits_config.output_root,
                product_root=morphfits_config.product_root,
                field=ficl.field,
                image_version=ficl.image_version,
                catalog_version=ficl.catalog_version,
                filter=ficl.filter,
                objects=ficl.objects,
                wrapper="galfit",
                display_progress=display_progress,
            )

    # Plot histograms
    plot.plot_histograms(
        output_root=morphfits_config.output_root,
        run_root=morphfits_config.run_root,
        field=morphfits_config.ficls[0].field,
        datetime=morphfits_config.datetime,
        run_number=morphfits_config.run_number,
    )

    logger.info("Exiting GalWrap.")
