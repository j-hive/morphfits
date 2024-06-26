"""Main program execution for the GALFIT morphology fitter wrapper package.
"""

# Imports


import gc
import logging
import shutil
from subprocess import Popen, PIPE, STDOUT
from pathlib import Path
from datetime import datetime as dt
import csv

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


FAILS = [0, 1, 2, 3, 5, 6, 7, 8]
"""GALFIT flags which indicate a failed run.
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
    pixscale: float,
    regenerate: bool = False,
    apply_sigma: bool = True,
    apply_psf: bool = True,
    apply_mask: bool = True,
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
    pixscale : float
        Pixel scale of science frame.
    regenerate : bool, optional
        Regenerate existing feedfiles, by default False.
    apply_sigma : bool, optional
        Use corresponding sigma map in GALFIT, by default True.
    apply_psf : bool, optional
        Use corresponding PSF in GALFIT, by default True.
    apply_mask : bool, optional
        Use corresponding mask in GALFIT, by default True.
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
        magnitude = catalog[object]["mag_auto"]
        half_light_radius = catalog[object]["a_image"]
        axis_ratio = catalog[object]["b_image"] / catalog[object]["a_image"]

        # Set configuration parameters from input
        # Note paths are all relative to ficlo_products
        feedfile_dict = {
            "stamp_path": path_str(product_paths["stamp"].name),
            "output_galfit_path": path_str(product_paths["galfit_model"].name),
            "sigma_path": path_str(product_paths["sigma"].name if apply_sigma else ""),
            "psf_path": path_str(product_paths["psf"].name if apply_psf else ""),
            "mask_path": path_str(product_paths["mask"].name if apply_mask else ""),
            "constraints_path": path_str(".constraints"),
            "image_size": str(image_size),
            "zeropoint": float_str(zeropoint),
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
    input_root: Path,
    product_root: Path,
    output_root: Path,
    field: str,
    image_version: str,
    catalog_version: str,
    filter: str,
    objects: list[int],
    display_progress: bool = False,
) -> list[str]:
    """Run GALFIT over all objects in a FICL.

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
    display_progress : bool, optional
        Display progress on terminal screen, by default False.

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
        ## Copy GALFIT and constraints to FICLO product directory
        galfit_path = GALFIT_DATA_ROOT / "galfit"
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

        ## Get product paths
        feedfile_path = paths.get_path(
            "feedfile",
            product_root=product_root,
            field=field,
            image_version=image_version,
            catalog_version=catalog_version,
            filter=filter,
            object=object,
        )

        ## Terminate if either path missing
        if not feedfile_path.exists():
            if not display_progress:
                logger.error(f"Missing feedfile, skipping.")
            return_codes.append(-1)
            continue

        ## Run subprocess and pipe output
        if not display_progress:
            logger.info(f"Running GALFIT for object {object}.")
        process = Popen(
            f"cd {str(ficlo_products_path)} && ./galfit {feedfile_path.name}",
            stdout=PIPE,
            shell=True,
            stderr=STDOUT,
            # bufsize=1,
            close_fds=True,
        )

        ## Capture output and log and close subprocess
        sublogger = logging.getLogger("GALFIT")
        for line in iter(process.stdout.readline, b""):
            if not display_progress:
                sublogger.info(line.rstrip().decode("utf-8"))
        process.stdout.close()
        process.wait()
        return_codes.append(process.returncode)

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
                path.rename(
                    paths.get_path(
                        "galfit_log",
                        output_root=output_root,
                        field=field,
                        image_version=image_version,
                        catalog_version=catalog_version,
                        filter=filter,
                        object=object,
                    )
                )
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
    """
    logger.info(
        "Recording parameters for FICL "
        + f"{'_'.join([field, image_version, catalog_version, filter])}."
    )

    # Create CSV if missing and write headers
    parameters_path = paths.get_path(
        "parameters", run_root=run_root, datetime=datetime, run_number=run_number
    )
    if not parameters_path.exists():
        with open(parameters_path, mode="w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(
                [
                    "field",
                    "image version",
                    "catalog version",
                    "filter",
                    "object",
                    "use",
                    "status",
                    "galfit flags",
                    "center x",
                    "center y",
                    "integrated magnitude",
                    "effective radius",
                    "concentration",
                    "axis ratio",
                    "position angle",
                ]
            )

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
            flags = []
            for flag in galfit_model_headers["FLAGS"].split():
                flags.append(FLAGS[flag])
            galfit_model_file.close()
            del galfit_model_file
            del galfit_model_headers
            gc.collect()

            ### Get validity ("use") from return code and flags
            if (return_code != 0) or any([flag in FAILS for flag in flags]):
                use = 0
            else:
                use = 1

            ### Get parameters from GALFIT log
            with open(galfit_log_path, mode="r") as log_file:
                lines = log_file.readlines()
                for line in lines:
                    if "sersic" in line:
                        raw_parameters = line.split()[3:]

            ### Write parameters and flags to CSV
            csv_row = [
                field,
                image_version,
                catalog_version,
                filter,
                object,
                use,
                return_code,
                sum([2**flag for flag in flags]),
            ]
            for raw_parameter in raw_parameters:
                csv_row.append(raw_parameter.replace(")", "").replace(",", ""))
            with open(parameters_path, mode="a", newline="") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(csv_row)

        ## Write empty row for failures
        else:
            with open(parameters_path, mode="a", newline="") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(
                    [
                        field,
                        image_version,
                        catalog_version,
                        filter,
                        object,
                        0,
                        return_code,
                        0,
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
    regenerate_stamp: bool = False,
    regenerate_psf: bool = False,
    regenerate_mask: bool = False,
    regenerate_sigma: bool = False,
    regenerate_feedfile: bool = True,
    apply_mask: bool = True,
    apply_psf: bool = True,
    apply_sigma: bool = True,
    kron_factor: int = 3,
    display_progress: bool = False,
):
    """Orchestrate GalWrap functions for passed configurations.

    Parameters
    ----------
    morphfits_config : MorphFITSConfig
        Configuration object for this program run.
    regenerate_products : bool, optional
        Regenerate all products, by default False.
    regenerate_stamp : bool, optional
        Regenerate stamps, by default False.
    regenerate_mask : bool, optional
        Regenerate masks, by default False.
    regenerate_psf : bool, optional
        Regenerate psfs, by default False.
    regenerate_sigma : bool, optional
        Regenerate sigmas, by default False.
    regenerate_feedfile : bool, optional
        Regenerate feedfile, by default False.
    apply_mask : bool, optional
        Apply generated mask product in GALFIT run, by default False.
    apply_psf : bool, optional
        Apply generated psf product in GALFIT run, by default False.
    apply_sigma : bool, optional
        Apply generated sigma product in GALFIT run, by default False.
    kron_factor : int, optional
        Multiplicative factor for image size, by default 3.
    """
    logger.info("Starting GalWrap.")

    # Generate products where missing, for each FICLO
    products.generate_products(
        morphfits_config=morphfits_config,
        regenerate_products=regenerate_products,
        regenerate_stamp=regenerate_stamp,
        regenerate_psf=regenerate_psf,
        regenerate_mask=regenerate_mask,
        regenerate_sigma=regenerate_sigma,
        regenerate_feedfile=regenerate_feedfile,
        apply_mask=apply_mask,
        apply_psf=apply_psf,
        apply_sigma=apply_sigma,
        kron_factor=kron_factor,
        display_progress=display_progress,
    )

    # Run GALFIT and record parameters, for each FICLO
    for ficl in morphfits_config.get_FICLs():
        return_codes = run_galfit(
            input_root=morphfits_config.input_root,
            output_root=morphfits_config.output_root,
            product_root=morphfits_config.product_root,
            field=ficl.field,
            image_version=ficl.image_version,
            catalog_version=ficl.catalog_version,
            filter=ficl.filter,
            objects=ficl.objects,
            display_progress=display_progress,
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

    logger.info("Exiting GalWrap.")
