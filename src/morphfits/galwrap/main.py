"""Main program execution for the GalWrap morphology fitter package.
"""

# Imports


import logging
from pathlib import Path

from . import config, paths


# Constants


logger = logging.getLogger("GALWRAP")


# Main


def main(
    config_path: str | Path | None = None,
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
    # Setup
    ## Create configuration object
    galwrap_config = config.create_config(
        config_path=config_path,
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

    # Create products if nonexistent
    ## Mask
    ## PSF
    ## Sigma
    ## For each FICLO
    ### Stamp
    ### Feedfile

    # Run GALFIT
    ## For each FICLO

    # Make plots
    ## For each FICLO


main(input_root="./data/galwrap_root/input")
