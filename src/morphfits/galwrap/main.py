"""Main program execution for the GalWrap morphology fitter package.
"""

# Imports


import logging
import shutil
from subprocess import Popen, PIPE, STDOUT
from pathlib import Path

from . import config, paths, products, GALWRAP_DATA_ROOT
from .setup import FICLO, GalWrapConfig
from ..utils import plots


# Constants


logger = logging.getLogger("GALWRAP")


# Functions


## GALFIT


def run_galfit(galwrap_config: GalWrapConfig):
    # Iterate over each FICL in configuration
    for ficl in galwrap_config.get_FICLs():
        logger.info(f"Running GALFIT for FICL {ficl}.")

        # Iterate over each object in FICL
        for object in ficl.objects:
            # Copy GALFIT and constraints to FICLO product directory
            galfit_path = GALWRAP_DATA_ROOT / "galfit"
            constraints_path = GALWRAP_DATA_ROOT / "default.constraints"
            product_ficlo_path = paths.get_path(
                "product_ficlo",
                product_root=galwrap_config.product_root,
                field=ficl.field,
                image_version=ficl.image_version,
                catalog_version=ficl.catalog_version,
                filter=ficl.filter,
                object=object,
            )
            shutil.copy(galfit_path, product_ficlo_path / "galfit")
            shutil.copy(constraints_path, product_ficlo_path / ".constraints")

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
                f"cd {str(product_ficlo_path)} && ./galfit {feedfile_path.name}",
                stdout=PIPE,
                shell=True,
                stderr=STDOUT,
                # bufsize=1,
                close_fds=True,
            )

            # Capture output and log and close subprocess
            sublogger = logging.getLogger("GALFIT")
            for line in iter(process.stdout.readline, b""):
                sublogger.info(line.rstrip().decode("utf-8"))
            process.stdout.close()
            process.wait()

            # Clean up GALFIT output
            if process.returncode == 0:
                for path in product_ficlo_path.iterdir():
                    # Move model to output directory
                    if "model" in path.name:
                        path.rename(
                            paths.get_path(
                                "model",
                                output_root=galwrap_config.output_root,
                                field=ficl.field,
                                image_version=ficl.image_version,
                                catalog_version=ficl.catalog_version,
                                filter=ficl.filter,
                                object=object,
                            )
                        )
                    # Move logs to output directory
                    elif "log" in path.name:
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
                    # Remove script, constraints, and feedfile records
                    elif ("galfit" in path.name) or ("constraints" in path.name):
                        path.unlink()


## Main


def stamp(
    input_root: str,
    field: str,
    image_version: str,
    catalog_version: str,
    filter: str,
    product_root: str | None = None,
    output_root: str | None = None,
    kron_factor: int = 3,
):
    """Generate stamp cutouts of all objects for a given FICL.

    Parameters
    ----------
    input_root : str
        Path to root directory of input products, e.g. catalogs, science images,
        and PSFs.
    field : str
        Field over which to generate stamps.
    image_version : str
        Version of image processing over which to generate stamps.
    catalog_version : str
        Version of cataloguing from which to get objects.
    filter : str
        Filter of observation.
    product_root : str | None, optional
        Path to root directory of products generated by this program to execute
        GALFIT, e.g. cutouts/stamps, masks, and feedfiles, by default found from
        input root.
    output_root : str | None, optional
        Path to root directory of GALFIT output products, e.g. morphology model
        and plots), by default found from input root.
    kron_factor : int, optional
        Multiplicative factor for image size, by default 3. The higher this is,
        the larger the image, and the smaller the object appears in the image.
    """
    logger.info(f"Starting GalWrap stamps.")

    # Create configuration object
    galwrap_config = config.create_config(
        input_root=input_root,
        product_root=product_root,
        output_root=output_root,
        fields=[field],
        image_versions=[image_version],
        catalog_versions=[catalog_version],
        filters=[filter],
    )

    # Setup product and output directories if nonexistent
    paths.setup_galwrap_paths(galwrap_config=galwrap_config, display_progress=True)

    # Regenerate stamps for each object in FICL
    products.generate_stamps(
        input_root=galwrap_config.input_root,
        product_root=galwrap_config.product_root,
        image_version=image_version,
        field=field,
        catalog_version=catalog_version,
        filter=filter,
        objects=galwrap_config.objects,
        kron_factor=kron_factor,
        display_progress=True,
    )

    # Plot objects, 50 at a time
    for ficl in galwrap_config.get_FICLs():
        plots.plot_objects(
            field=ficl.field,
            image_version=ficl.image_version,
            catalog_version=ficl.catalog_version,
            filter=ficl.filter,
            objects=ficl.objects,
            product_root=galwrap_config.product_root,
            output_root=galwrap_config.output_root,
        )

    logger.info("Exiting GalWrap.")


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
    kron_factor: int = 3,
    psf_factor: int = 4,
):
    """Orchestrate GalWrap functions for passed configurations.

    Parameters
    ----------
    config_path : str | Path | None, optional
        Path to user config yaml file, by default None (no user config file
        provided).
    galwrap_root : str | Path | None, optional
        Path to root directory of GalWrap filesystem, by default None (not
        passed through CLI).
    input_root : str | Path | None, optional
        Path to root directory of input products, e.g. catalogs, science images,
        and PSFs, by default None (not passed through CLI).
    product_root : str | Path | None, optional
        Path to root directory of products generated by this program to execute
        GALFIT, e.g. cutouts/stamps, masks, and feedfiles, by default None (not
        passed through CLI).
    output_root : str | Path | None, optional
        Path to root directory of GALFIT output products, e.g. morphology model
        and plots, by default None (not passed through CLI).
    fields : list[str] | None, optional
        List of fields over which to execute GALFIT, by default None (not passed
        through CLI).
    image_versions : list[str] | None, optional
        List of image versions over which to execute GALFIT, by default None
        (not passed through CLI).
    catalog_versions : list[str] | None, optional
        List of catalog versions over which to execute GALFIT, by default None
        (not passed through CLI).
    filters : list[str] | None, optional
        List of filter bands over which to execute GALFIT, by default None (not
        passed through CLI).
    objects : list[int] | None, optional
        List of target IDs over which to execute GALFIT, for each catalog, by
        default None (not passed through CLI).
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
    psf_factor : int, optional
        Division factor for PSF crop size, by default 4.
    """
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
        kron_factor=kron_factor,
        psf_factor=psf_factor,
    )

    # Run GALFIT, for each FICLO
    run_galfit(galwrap_config=galwrap_config)

    # Plot models, for each FICLO
    for ficl in galwrap_config.get_FICLs():
        plots.plot_products(
            field=ficl.field,
            image_version=ficl.image_version,
            catalog_version=ficl.catalog_version,
            filter=ficl.filter,
            objects=ficl.objects,
            product_root=galwrap_config.product_root,
            output_root=galwrap_config.output_root,
        )
        for object in ficl.objects:
            stamp_path = paths.get_path(
                "stamp",
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
            output_path = paths.get_path(
                "comparison",
                output_root=galwrap_config.output_root,
                field=ficl.field,
                image_version=ficl.image_version,
                catalog_version=ficl.catalog_version,
                filter=ficl.filter,
                object=object,
            )

            if not stamp_path.exists():
                logger.debug(f"Missing stamp {stamp_path.name}, skipping.")
                continue
            if not model_path.exists():
                logger.debug(f"Missing model {model_path.name}, skipping.")
                continue

            plots.plot_comparison(
                stamp_path=stamp_path, model_path=model_path, output_path=output_path
            )

    logger.info("Exiting GalWrap.")
