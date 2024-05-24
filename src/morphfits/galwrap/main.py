"""Main program execution for the GalWrap morphology fitter package.
"""

# Imports


import logging
from subprocess import Popen, PIPE, STDOUT
from pathlib import Path

from . import config, paths, products, GALWRAP_DATA_ROOT
from .setup import FICLO, GalWrapConfig
from ..utils import plots


# Constants


logger = logging.getLogger("GALWRAP")


# Functions


def run_galfit(ficlo: FICLO, galwrap_config: GalWrapConfig):
    # Get paths
    feedfile_path = paths.get_path(
        "feedfile", galwrap_config=galwrap_config, ficlo=ficlo
    )
    model_path = paths.get_path("model", galwrap_config=galwrap_config, ficlo=ficlo)

    # Run subprocess and pipe output
    logger.info(f"Running GALFIT.")
    process = Popen(
        f"cd {str(GALWRAP_DATA_ROOT)} && ./galfit {str(feedfile_path)}",
        stdout=PIPE,
        shell=True,
        stderr=STDOUT,
        bufsize=1,
        close_fds=True,
    )

    # Capture output and log and close subprocess
    sublogger = logging.getLogger("GALFIT")
    process_log = []
    for line in iter(process.stdout.readline, b""):
        process_log.append(line.rstrip().decode("utf-8"))
    process.stdout.close()
    process.wait()
    for line in process_log:
        sublogger.info(line)

    # Clean up GALFIT output
    if (process.returncode == 0) and (model_path.exists()):
        for path in GALWRAP_DATA_ROOT.iterdir():
            # Move logs to output directory
            if "log" in path.name:
                path.rename(
                    paths.get_path("fitlog", galwrap_config=galwrap_config, ficlo=ficlo)
                )
            # Remove records
            elif "galfit." in path.name:
                path.unlink()


def main(
    config_path: str | Path | None = None,
    galwrap_root: str | Path | None = None,
    input_root: str | Path | None = None,
    product_root: str | Path | None = None,
    output_root: str | Path | None = None,
    field: str | None = None,
    fields: list[str] | None = None,
    image_version: str | None = None,
    image_versions: list[str] | None = None,
    catalog_version: str | None = None,
    catalog_versions: list[str] | None = None,
    filter: str | None = None,
    filters: list[str] | None = None,
    object: int | None = None,
    objects: list[int] | None = None,
    pixscale: float | None = None,
    pixscales: list[float] | None = None,
    morphology_version: str | None = None,
    morphology_versions: list[str] | None = None,
    regenerate: bool = False,
    regenerate_stamp: bool = False,
    regenerate_psf: bool = False,
    regenerate_mask: bool = False,
    regenerate_sigma: bool = False,
    regenerate_feedfile: bool = True,
    use_mask: bool = True,
    use_psf: bool = True,
    use_sigma: bool = True,
):
    logger.info("Starting GalWrap.")

    # Setup
    ## Create configuration object
    if config_path is not None:
        logger.info(f"Creating configuration object from config file.")
    else:
        logger.info(f"Creating configuration object from passed values.")
    galwrap_config = config.create_config(
        config_path=config_path,
        galwrap_root=galwrap_root,
        input_root=input_root,
        product_root=product_root,
        output_root=output_root,
        field=field,
        fields=fields,
        image_version=image_version,
        image_versions=image_versions,
        catalog_version=catalog_version,
        catalog_versions=catalog_versions,
        filter=filter,
        filters=filters,
        object=object,
        objects=objects,
        pixscale=pixscale,
        pixscales=pixscales,
        morphology_version=morphology_version,
        morphology_versions=morphology_versions,
    )

    ## Setup product and output directories if nonexistent
    logger.info("Setting up product and output directories where missing.")
    paths.setup_galwrap_paths(galwrap_config=galwrap_config)

    # Create products if nonexistent, for each FICLO
    logger.info("Generating products from input data, where nonexistent.")
    for ficlo in galwrap_config.get_ficlos():
        products.generate_products(
            ficlo=ficlo,
            galwrap_config=galwrap_config,
            regenerate=regenerate,
            regenerate_stamp=regenerate_stamp,
            regenerate_psf=regenerate_psf,
            regenerate_mask=regenerate_mask,
            regenerate_sigma=regenerate_sigma,
            regenerate_feedfile=regenerate_feedfile,
            use_mask=use_mask,
            use_psf=use_psf,
            use_sigma=use_sigma,
        )

    # Run GALFIT, for each FICLO
    logger.info("Running GALFIT for each configuration.")
    for ficlo in galwrap_config.get_ficlos():
        run_galfit(ficlo=ficlo, galwrap_config=galwrap_config)

    # Make plots
    logger.info("Plotting models.")
    for ficlo in galwrap_config.get_ficlos():
        stamp_path = paths.get_path("stamp", galwrap_config=galwrap_config, ficlo=ficlo)
        model_path = paths.get_path("model", galwrap_config=galwrap_config, ficlo=ficlo)
        output_path = paths.get_path(
            "comparison", galwrap_config=galwrap_config, ficlo=ficlo
        )
        plots.plot_comparison(
            stamp_path=stamp_path, model_path=model_path, output_path=output_path
        )

    logger.info("Exiting GalWrap.")
