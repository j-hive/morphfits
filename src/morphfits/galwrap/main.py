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


def run_galfit(galwrap_config: GalWrapConfig):
    # Iterate over each FICL in configuration
    for ficl in galwrap_config.get_FICLs():
        logger.info(f"Running GALFIT for FICL {ficl}.")

        # Iterate over each object in FICL
        for object in ficl.objects:
            # Get product paths
            feedfile_path = paths.get_path(
                "feedfile",
                product_root=galwrap_config.product_root,
                field=ficl.field,
                image_version=ficl.image_version,
                catalog_version=ficl.catalog_version,
                filter=ficl.filter,
                object=object,
            )
            model_path = paths.get_path(
                "model",
                output_root=galwrap_config.output_root,
                field=ficl.field,
                image_version=ficl.image_version,
                catalog_version=ficl.catalog_version,
                filter=ficl.filter,
                object=object,
            )

            # Terminate if either path missing
            if not feedfile_path.exists():
                logger.error(f"Missing feedfile, skipping.")
                continue

            # Run subprocess and pipe output
            logger.info(f"Running GALFIT for object {object}.")
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
            for line in iter(process.stdout.readline, b""):
                sublogger.info(line.rstrip().decode("utf-8"))
            process.stdout.close()
            process.wait()

            # Clean up GALFIT output
            if (process.returncode == 0) and (model_path.exists()):
                for path in GALWRAP_DATA_ROOT.iterdir():
                    # Move logs to output directory
                    if "log" in path.name:
                        path.rename(
                            paths.get_path(
                                "fitlog",
                                output_root=galwrap_config.output_root,
                                field=ficl.field,
                                image_version=ficl.image_version,
                                catalog_version=ficl.catalog_version,
                                filter=ficl.filter,
                                object=object,
                            )
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
    fields: list[str] | None = None,
    image_versions: list[str] | None = None,
    catalog_versions: list[str] | None = None,
    filters: list[str] | None = None,
    objects: list[int] | None = None,
    regenerate_products: bool = False,
    regenerate_stamp: bool = False,
    regenerate_psf: bool = False,
    regenerate_mask: bool = False,
    regenerate_sigma: bool = False,
    regenerate_feedfile: bool = True,
    apply_mask: bool = True,
    apply_psf: bool = True,
    apply_sigma: bool = True,
):
    logger.info("Starting GalWrap.")

    # Setup
    ## Create configuration object
    galwrap_config = config.create_config(
        config_path=config_path,
        galwrap_root=galwrap_root,
        input_root=input_root,
        product_root=product_root,
        output_root=output_root,
        fields=fields,
        image_versions=image_versions,
        catalog_versions=catalog_versions,
        filters=filters,
        objects=objects,
    )

    ## Setup product and output directories if nonexistent
    paths.setup_galwrap_paths(galwrap_config=galwrap_config)

    # Create products if nonexistent, for each FICLO
    products.generate_products(
        galwrap_config=galwrap_config,
        regenerate_products=regenerate_products,
        regenerate_stamp=regenerate_stamp,
        regenerate_psf=regenerate_psf,
        regenerate_mask=regenerate_mask,
        regenerate_sigma=regenerate_sigma,
        regenerate_feedfile=regenerate_feedfile,
        apply_mask=apply_mask,
        apply_psf=apply_psf,
        apply_sigma=apply_sigma,
    )

    # Run GALFIT, for each FICLO
    run_galfit(galwrap_config=galwrap_config)

    import sys

    sys.exit()
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
