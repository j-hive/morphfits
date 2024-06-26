"""Main program execution for morphology fitting.
"""

# Imports


import logging
from typing import Optional
from datetime import datetime as dt

import typer

from . import config, paths, plots, products
from .wrappers import galfit
from .utils import logs


# App Instantiation
app = typer.Typer()


# Morphology Fitters


@app.command()
def galwrap(
    config_path: Optional[str] = None,
    morphfits_root: Optional[str] = None,
    input_root: Optional[str] = None,
    product_root: Optional[str] = None,
    output_root: Optional[str] = None,
    fields: Optional[list[str]] = None,
    image_versions: Optional[list[str]] = None,
    catalog_versions: Optional[list[str]] = None,
    filters: Optional[list[str]] = None,
    objects: Optional[list[int]] = None,
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
    """Run GALFIT over given FICLOs.

    Parameters
    ----------
    config_path : str | Path | None, optional
        Path to user config yaml file, by default None (no user config file
        provided).
    morphfits_root : str | Path | None, optional
        Path to root directory of MorphFITS filesystem, by default None (not
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
    display_progress : bool, optional
        Display progress on terminal screen via tqdm, by default False.
    """
    # Create configuration object
    morphfits_config = config.create_config(
        config_path=config_path,
        morphfits_root=morphfits_root,
        input_root=input_root,
        product_root=product_root,
        output_root=output_root,
        fields=fields,
        image_versions=image_versions,
        catalog_versions=catalog_versions,
        filters=filters,
        objects=objects,
        display_progress=display_progress,
    )

    # Create program and module logger
    logs.create_logger(
        filename=paths.get_path(
            "morphfits_log",
            output_root=morphfits_config.output_root,
            datetime=morphfits_config.datetime,
        )
    )
    logger = logging.getLogger("MORPHFITS")
    logger.info("Starting MorphFITS.")

    # Write configuration to file in run directory
    morphfits_config.write()
    return

    # Call wrapper
    galfit.main(
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

    # Write configuration to file in run directory
    morphfits_config.write()

    # Exit
    logger.info("Exiting MorphFITS.")


@app.command()
def imcascade():
    """Run imcascade over given FICLOs. NOT IMPLEMENTED"""
    raise NotImplementedError


@app.command()
def pysersic():
    """Run pysersic over given FICLOs. NOT IMPLEMENTED"""
    raise NotImplementedError


# Other Apps


@app.command()
def stamp(
    input_root: str,
    field: str,
    image_version: str,
    catalog_version: str,
    filter: str,
    product_root: Optional[str] = None,
    output_root: Optional[str] = None,
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
    product_root : str, optional
        Path to root directory of products generated by this program to execute
        GALFIT, e.g. cutouts/stamps, masks, and feedfiles, by default found from
        input root.
    output_root : str, optional
        Path to root directory of GALFIT output products, e.g. morphology model
        and plots), by default found from input root.
    kron_factor : int, optional
        Multiplicative factor for image size, by default 3. The higher this is,
        the larger the image, and the smaller the object appears in the image.
    """
    """Generate stamp cutouts of all objects for a given FICL.

    Parameters
    ----------
    morphfits_config : MorphFITSConfig
        Configuration object for this program run.
    kron_factor : int, optional
        Multiplicative factor for image size, by default 3. The higher this is,
        the larger the image, and the smaller the object appears in the image.
    """
    # Create configuration object
    morphfits_config = config.create_config(
        input_root=input_root,
        product_root=product_root,
        output_root=output_root,
        fields=[field],
        image_versions=[image_version],
        catalog_versions=[catalog_version],
        filters=[filter],
        display_progress=True,
    )

    # Create program and module logger
    logs.create_logger(
        filename=paths.get_path(
            "morphfits_log",
            output_root=morphfits_config.output_root,
            datetime=morphfits_config.datetime,
        )
    )
    logger = logging.getLogger("MORPHFITS")
    logger.info("Starting MorphFITS.")

    # Display progress
    ficl = next(morphfits_config.get_FICLs())
    logger.info(f"Starting MorphFITS stamps for FICL {ficl}.")

    # Regenerate stamps for each object in FICL
    products.generate_stamps(
        input_root=morphfits_config.input_root,
        product_root=morphfits_config.product_root,
        image_version=ficl.image_version,
        field=ficl.field,
        catalog_version=ficl.catalog_version,
        filter=ficl.filter,
        objects=ficl.objects,
        kron_factor=kron_factor,
        display_progress=True,
    )

    # Plot all objects
    plots.plot_objects(
        product_root=morphfits_config.product_root,
        output_root=morphfits_config.output_root,
        field=ficl.field,
        image_version=ficl.image_version,
        catalog_version=ficl.catalog_version,
        filter=ficl.filter,
        objects=ficl.objects,
        display_progress=True,
    )

    # Exit
    logger.info("Exiting MorphFITS.")


# Main Program


def main():
    app()


if __name__ == "__main__":
    main()


# TODO
# want to save logs to each run
# can't save it there until directory is created
# directory created after something has already been logged
