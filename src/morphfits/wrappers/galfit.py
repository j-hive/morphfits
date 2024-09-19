"""Main program execution for the GALFIT morphology fitter wrapper package.
"""

# Imports


import gc
import logging
import shutil
from subprocess import Popen, PIPE, STDOUT
from pathlib import Path
from datetime import datetime as dt
import re
import csv

import pandas as pd
from astropy.io import fits
from astropy.table import Table
from jinja2 import Template
from tqdm import tqdm

from . import GALFIT_DATA_ROOT
from .. import config, paths, plots, products
from ..utils import science


# Constants


logger = logging.getLogger("GALWRAP")
"""Logging object for this module.
"""


FLAGS = {
    "1": 0,
    "2": 1,
    "A-1": 2,
    "A-2": 3,
    "A-3": 4,
    "A-4": 5,
    "A-5": 6,
    "A-6": 7,
    "C-1": 8,
    "C-2": 9,
    "H-1": 10,
    "H-2": 11,
    "H-3": 12,
    "H-4": 13,
    "I-1": 14,
    "I-2": 15,
    "I-3": 16,
    "I-4": 17,
    "I-5": 18,
}
"""GALFIT flags as written in the model headers, and their corresponding
bit-mask exponent, where 2 is the base.
"""


FAIL = 495
"""GALFIT flags which indicate a failed run.

See Also
--------
README.md 
    Breakdown on binary flag values, and which flags result in failed runs. 
"""


GALFIT_LOG_REGEX = "[\*|\[]?\d{1,10}[\.]\d{1,2}[\*|\[]?"
GALFIT_LOG_FLOAT_REGEX = "\d{1,10}[\.]\d{1,2}"
"""Regex for seven .2f numbers found in GALFIT logs, which may or may not be
enveloped by * or [] characters.

See Also
--------
`record_parameters`
    Function using this expression to record the fitting parameters from logs.
"""


CATALOG_HEADERS = [
    "use",
    "field",
    "image version",
    "catalog version",
    "filter",
    "object",
    "return code",
    "flags",
    "convergence",
    "center x",
    "center y",
    "surface brightness",
    "effective radius",
    "sersic",
    "axis ratio",
    "position angle",
    "center x error",
    "center y error",
    "surface brightness error",
    "effective radius error",
    "sersic error",
    "axis ratio error",
    "position angle error",
]
"""Headers for the MorphFITS catalog CSV.
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
    product_path_names = ["galfit_model", "stamp", "sigma", "psf", "mask"]

    # Load in catalog
    catalog_path = paths.get_path(
        "catalog",
        input_root=input_root,
        field=field,
        image_version=image_version,
    )
    catalog = Table.read(catalog_path)

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
    del catalog_path
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
        half_light_radius = catalog[object]["a_image"]
        axis_ratio = catalog[object]["b_image"] / catalog[object]["a_image"]

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
            "output_galfit_path": path_str(product_paths["galfit_model"].name),
            "sigma_path": path_str(product_paths["sigma"].name),
            "psf_path": path_str(product_paths["psf"].name),
            "mask_path": path_str(product_paths["mask"].name),
            "constraints_path": path_str(".constraints"),
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
    field: str,
    image_version: str,
    catalog_version: str,
    filter: str,
    objects: list[int],
    display_progress: bool = False,
    refit: bool = False,
) -> list[str]:
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

    Returns
    -------
    list[int]
        List of GALFIT return codes for each FICLO.
    """
    logger.info(
        f"Running GALFIT for FICL {'_'.join([field, image_version, catalog_version, filter])}."
    )

    # Iterate over each object in FICL
    return_codes = []
    for object in (
        tqdm(objects, unit="run", leave=False) if display_progress else objects
    ):
        ## Get model path
        model_path = paths.get_path(
            "galfit_model",
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
            return_codes.append(0)
            continue

        ## Get feedfile path
        feedfile_path = paths.get_path(
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
                logger.debug(f"Skipping object {object}, missing products.")
            return_codes.append(2)
            continue

        ## Copy GALFIT and constraints to FICLO product directory
        constraints_path = GALFIT_DATA_ROOT / "default.constraints"
        ficlo_products_path = paths.get_path(
            "ficlo_products",
            product_root=product_root,
            field=field,
            image_version=image_version,
            catalog_version=catalog_version,
            filter=filter,
            object=object,
        )
        shutil.copy(galfit_path, ficlo_products_path / "galfit")
        shutil.copy(constraints_path, ficlo_products_path / ".constraints")

        ## Run subprocess and pipe output
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

        ## Capture output and close subprocess
        galfit_lines = []
        for line in iter(process.stdout.readline, b""):
            galfit_lines.append(line.rstrip().decode("utf-8"))
        process.stdout.close()
        process.wait()
        return_code = process.returncode
        return_codes.append(return_code)
        if return_code != 0:
            logger.error(
                f"GALFIT did not run successfully and returned with code {return_code}."
            )

        ## Write captured output to GALFIT log file
        galfit_log_path = paths.get_path(
            "galfit_log",
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
                    paths.get_path(
                        "galfit_model",
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

    return return_codes


def record_parameters(
    return_codes: list[int],
    datetime: dt,
    run_number: int,
    output_root: Path,
    run_root: Path,
    field: str,
    image_version: str,
    catalog_version: str,
    filter: str,
    objects: list[int],
    display_progress: bool = False,
):
    """Record GALFIT fitting parameters to the corresponding run directory.

    Parameters
    ----------
    return_codes : list[int]
        List of GALFIT return codes for each FICLO.
    datetime : datetime
        Datetime of start of run.
    run_number : int
        Number of run in runs directory when other runs have the same datetime.
    output_root : Path
        Path to root MorphFITS output directory.
    run_root : Path
        Path to root runs directory.
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
    for_run : bool, optional
    """
    logger.info(
        "Recording fitting parameters in catalogs for FICL "
        + f"{'_'.join([field, image_version, catalog_version, filter])}."
    )

    # Create CSV if missing and write headers
    path_catalog_run = paths.get_path(
        "parameters", run_root=run_root, datetime=datetime, run_number=run_number
    )
    path_catalog_morphfits = paths.get_path(
        "morphfits_catalog", output_root=output_root
    )
    if not path_catalog_run.exists():
        with open(path_catalog_run, mode="w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(CATALOG_HEADERS)
    if not path_catalog_morphfits.exists():
        with open(path_catalog_morphfits, mode="w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(CATALOG_HEADERS[1:])

    # Iterate over each object in FICL
    for i in (
        tqdm(range(len(objects)), unit="object", leave=False)
        if display_progress
        else range(len(objects))
    ):
        object, return_code = objects[i], return_codes[i]
        galfit_log_path = paths.get_path(
            "galfit_log",
            output_root=output_root,
            field=field,
            image_version=image_version,
            catalog_version=catalog_version,
            filter=filter,
            object=object,
        )
        galfit_model_path = paths.get_path(
            "galfit_model",
            output_root=output_root,
            field=field,
            image_version=image_version,
            catalog_version=catalog_version,
            filter=filter,
            object=object,
        )

        ## Write parameters from GALFIT log for successful runs
        if (galfit_model_path.exists()) and (galfit_log_path.exists()):
            ### Get flags from model
            galfit_model_file = fits.open(galfit_model_path)
            galfit_model_headers = galfit_model_file[2].header
            flags = 0
            for flag in galfit_model_headers["FLAGS"].split():
                flags += 2 ** FLAGS[flag]
            galfit_model_file.close()
            del galfit_model_file
            del galfit_model_headers
            gc.collect()

            ### Get parameters from GALFIT log
            with open(galfit_log_path, mode="r") as log_file:
                lines = log_file.readlines()
                while i < len(lines) - 8:
                    if (
                        ("---" in lines[i])
                        and (lines[i][0] != "#")
                        and ("Input image" in lines[i + 2])
                    ):
                        raw_parameters = re.findall(GALFIT_LOG_REGEX, lines[i + 7])
                        errors = re.findall(GALFIT_LOG_FLOAT_REGEX, lines[i + 8])
                        break
                    else:
                        i += 1

            ### Strip parameters and append to list[str]
            parameters = []
            convergence = 0
            for i in range(len(raw_parameters)):
                parameter = raw_parameters[i]

                #### Only care about convergence of size, sersic, and ratio
                if i in [3, 4, 5]:
                    for fail_indicator in ["[", "]", "*"]:
                        if fail_indicator in parameter:
                            convergence += 2 ** (i - 3)
                            parameter = parameter.replace(fail_indicator, "")
                parameters.append(parameter)

            ### Get validity ("success") from return code, flags, and convergence
            if (return_code != 0) or ((flags & FAIL) > 0) or (convergence > 0):
                use = False
            else:
                use = True

            ### Write parameters and flags to CSV
            csv_row = [
                use,
                field,
                image_version,
                catalog_version,
                filter,
                object,
                return_code,
                flags,
                convergence,
            ]
            for parameter in parameters:
                csv_row.append(parameter)
            for error in errors:
                csv_row.append(error)
            with open(path_catalog_run, mode="a", newline="") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(csv_row)

        ## Write empty row for failures
        else:
            with open(path_catalog_run, mode="a", newline="") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(
                    [
                        False,
                        field,
                        image_version,
                        catalog_version,
                        filter,
                        object,
                        return_code,
                        0,
                        0,
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                    ]
                )


## Main


def main(
    morphfits_config: config.MorphFITSConfig,
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
        for ficl in morphfits_config.get_FICLs():
            return_codes = run_galfit(
                galfit_path=morphfits_config.galfit_path,
                input_root=morphfits_config.input_root,
                output_root=morphfits_config.output_root,
                product_root=morphfits_config.product_root,
                field=ficl.field,
                image_version=ficl.image_version,
                catalog_version=ficl.catalog_version,
                filter=ficl.filter,
                objects=ficl.objects,
                display_progress=display_progress,
                refit=force_refit,
            )
            record_parameters(
                return_codes=return_codes,
                datetime=morphfits_config.datetime,
                run_number=morphfits_config.run_number,
                output_root=morphfits_config.output_root,
                run_root=morphfits_config.run_root,
                field=ficl.field,
                image_version=ficl.image_version,
                catalog_version=ficl.catalog_version,
                filter=ficl.filter,
                objects=ficl.objects,
                display_progress=display_progress,
            )

    # Plot models, for each FICLO
    if make_plots:
        for ficl in morphfits_config.get_FICLs():
            plots.plot_model(
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

    # Plot histogram for each run if the run catalog exists
    run_catalog_path = paths.get_path(
        "parameters",
        run_root=morphfits_config.run_root,
        datetime=morphfits_config.datetime,
        run_number=morphfits_config.run_number,
    )
    if run_catalog_path.exists():
        plots.plot_histogram(
            run_root=morphfits_config.run_root,
            datetime=morphfits_config.datetime,
            run_number=morphfits_config.run_number,
        )

    logger.info("Exiting GalWrap.")
