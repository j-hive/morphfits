"""Main program execution for morphology fitting.
"""

# Imports


import logging
from pathlib import Path
from typing import Union, Optional

import typer

from .utils import logs
from .galwrap.main import main as galwrap_main


# Main


# Create logger for program and module
main_logger = logs.create_logger()
logger = logging.getLogger("MORPHFITS")

# Create typer app
app = typer.Typer()


# Morphology versions


@app.command()
def galwrap(
    config_path: Optional[str] = None,
    galwrap_root: Optional[str] = None,
    input_root: Optional[str] = None,
    product_root: Optional[str] = None,
    output_root: Optional[str] = None,
    field: Optional[str] = None,
    fields: Optional[list[str]] = None,
    image_version: Optional[str] = None,
    image_versions: Optional[list[str]] = None,
    catalog_version: Optional[str] = None,
    catalog_versions: Optional[list[str]] = None,
    filter: Optional[str] = None,
    filters: Optional[list[str]] = None,
    object: Optional[int] = None,
    objects: Optional[list[int]] | None = None,
    pixscale: Optional[float] = None,
    pixscales: Optional[list[float]] | None = None,
    morphology_version: Optional[str] = None,
    morphology_versions: Optional[list[str]] = None,
):
    galwrap_main(
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


@app.command()
def pyfits(to_be_implemented: int):
    pass


if __name__ == "__main__":
    logger.info("Starting MorphFITS.")
    app()
