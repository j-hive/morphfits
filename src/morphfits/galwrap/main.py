"""Main program execution for the GalWrap morphology fitter package.
"""

# Imports


import logging
from pathlib import Path

from . import config, paths, products


# Constants


logger = logging.getLogger("GALWRAP")


# Functions


def run_galfit():
    pass


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
):
    logger.info("Starting GalWrap.")

    # Setup
    ## Create configuration object
    if config_path is not None:
        logger.info(f"Creating configuration object from config file at {config_path}")
    else:
        logger.info(f"Creating configuration object with input root at {input_root}")
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
    logger.info(
        "Setting up product and output directories "
        + f"where missing at {galwrap_config.galwrap_root}"
    )
    paths.setup_galwrap_paths(galwrap_config=galwrap_config)

    # Create products if nonexistent, for each FICLO
    logger.info("Generating products from input data, where nonexistent.")
    # products.generate_all_products(galwrap_config=galwrap_config)
    for ficlo in galwrap_config.get_ficlos():
        products.generate_products(ficlo=ficlo, galwrap_config=galwrap_config)

    # Run GALFIT
    ## For each FICLO

    # Make plots
    ## For each FICLO
