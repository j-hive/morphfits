"""Main program execution for morphology fitting.
"""

# Imports


import logging
from typing import Optional, List
from typing_extensions import Annotated
from pathlib import Path

import typer

from . import config, paths, plots, products
from .wrappers import galfit
from .utils import logs


# App Instantiation
app = typer.Typer()


# Morphology Fitters


@app.command()
def galwrap(
    config_path: Annotated[
        typer.FileText,
        typer.Option(
            help="Path to configuration settings YAML file.",
            rich_help_panel="Paths",
            exists=True,
            dir_okay=False,
            show_default=False,
            resolve_path=True,
        ),
    ] = None,
    morphfits_root: Annotated[
        Path,
        typer.Option(
            help="Path to MorphFITS filesystem root.",
            rich_help_panel="Paths",
            exists=True,
            file_okay=False,
            show_default=False,
            resolve_path=True,
        ),
    ] = None,
    input_root: Annotated[
        Path,
        typer.Option(
            help="Path to root input directory. Must be set here or in --config-path.",
            rich_help_panel="Paths",
            exists=True,
            file_okay=False,
            show_default=False,
            resolve_path=True,
        ),
    ] = None,
    output_root: Annotated[
        Path,
        typer.Option(
            help="Path to root output directory.",
            rich_help_panel="Paths",
            exists=True,
            file_okay=False,
            show_default=False,
            resolve_path=True,
            writable=True,
        ),
    ] = None,
    product_root: Annotated[
        Path,
        typer.Option(
            help="Path to root products directory.",
            rich_help_panel="Paths",
            exists=True,
            file_okay=False,
            show_default=False,
            resolve_path=True,
            writable=True,
        ),
    ] = None,
    run_root: Annotated[
        Path,
        typer.Option(
            help="Path to root runs directory.",
            rich_help_panel="Paths",
            exists=True,
            file_okay=False,
            show_default=False,
            resolve_path=True,
            writable=True,
        ),
    ] = None,
    fields: Annotated[
        Optional[List[str]],
        typer.Option(
            "--fields",
            "-F",
            help="List of fields over which to run MorphFITS.",
            rich_help_panel="FICLOs",
            show_default=False,
        ),
    ] = None,
    image_versions: Annotated[
        Optional[List[str]],
        typer.Option(
            "--image-versions",
            "-I",
            help="List of image versions over which to run MorphFITS.",
            rich_help_panel="FICLOs",
            show_default=False,
        ),
    ] = None,
    catalog_versions: Annotated[
        Optional[List[str]],
        typer.Option(
            "--catalog-versions",
            "-C",
            help="List of catalog versions over which to run MorphFITS.",
            rich_help_panel="FICLOs",
            show_default=False,
        ),
    ] = None,
    filters: Annotated[
        Optional[List[str]],
        typer.Option(
            "--filters",
            "-L",
            help="List of filters over which to run MorphFITS.",
            rich_help_panel="FICLOs",
            show_default=False,
        ),
    ] = None,
    objects: Annotated[
        Optional[List[int]],
        typer.Option(
            "--objects",
            "-O",
            help="List of object IDs over which to run MorphFITS.",
            rich_help_panel="FICLOs",
            show_default=False,
        ),
    ] = None,
    regenerate_products: Annotated[
        bool,
        typer.Option(
            "--regenerate-products",
            help="Regenerate all products. Overrides other flags.",
            rich_help_panel="Products",
            is_flag=True,
        ),
    ] = False,
    regenerate_stamps: Annotated[
        bool,
        typer.Option(
            "--regenerate-stamps",
            help="Regenerate all stamps. Must be set for other products to regenerate.",
            rich_help_panel="Products",
            is_flag=True,
        ),
    ] = False,
    regenerate_psfs: Annotated[
        bool,
        typer.Option(
            "--regenerate-psfs",
            help="Regenerate all PSF crops.",
            rich_help_panel="Products",
            is_flag=True,
        ),
    ] = False,
    regenerate_masks: Annotated[
        bool,
        typer.Option(
            "--regenerate-masks",
            help="Regenerate all bad pixel masks.",
            rich_help_panel="Products",
            is_flag=True,
        ),
    ] = False,
    regenerate_sigmas: Annotated[
        bool,
        typer.Option(
            "--regenerate-sigmas",
            help="Regenerate all sigma maps.",
            rich_help_panel="Products",
            is_flag=True,
        ),
    ] = False,
    keep_feedfiles: Annotated[
        bool,
        typer.Option(
            "--keep-feedfiles",
            help="Use existing GALFIT feedfiles.",
            rich_help_panel="Products",
            is_flag=True,
        ),
    ] = False,
    skip_products: Annotated[
        bool,
        typer.Option(
            "--skip-products",
            help="Skip all product generation.",
            rich_help_panel="Stages",
            is_flag=True,
        ),
    ] = False,
    skip_fits: Annotated[
        bool,
        typer.Option(
            "--skip-fits",
            help="Skip all fitting via GALFIT.",
            rich_help_panel="Stages",
            is_flag=True,
        ),
    ] = False,
    skip_plots: Annotated[
        bool,
        typer.Option(
            "--skip-plots",
            help="Skip all model plotting and visualizations.",
            rich_help_panel="Stages",
            is_flag=True,
        ),
    ] = False,
    display_progress: Annotated[
        bool,
        typer.Option(
            "--display-progress",
            "-d",
            help="Display progress as a loading bar and suppress per-object logging.",
            is_flag=True,
        ),
    ] = False,
):
    """Model objects from given FICLs using GALFIT."""
    # Create configuration object
    morphfits_config = config.create_config(
        config_path=config_path,
        morphfits_root=morphfits_root,
        input_root=input_root,
        output_root=output_root,
        product_root=product_root,
        run_root=run_root,
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
            run_root=morphfits_config.run_root,
            datetime=morphfits_config.datetime,
            run_number=morphfits_config.run_number,
        )
    )
    logger = logging.getLogger("MORPHFITS")
    logger.info("Starting MorphFITS.")

    # Call wrapper
    galfit.main(
        morphfits_config=morphfits_config,
        regenerate_products=regenerate_products,
        regenerate_stamp=regenerate_stamps,
        regenerate_psf=regenerate_psfs,
        regenerate_mask=regenerate_masks,
        regenerate_sigma=regenerate_sigmas,
        keep_feedfiles=keep_feedfiles,
        skip_products=skip_products,
        skip_fits=skip_fits,
        skip_plots=skip_plots,
        display_progress=display_progress,
    )

    # Remove empty directories
    morphfits_config.clean_paths(display_progress=display_progress)

    # Write configuration to file in run directory
    morphfits_config.write()

    # Exit
    logger.info("Exiting MorphFITS.")


@app.command()
def imcascade():
    """Model objects from given FICLs using imcascade. NOT IMPLEMENTED"""
    raise NotImplementedError


@app.command()
def pysersic():
    """Model objects from given FICLs using pysersic. NOT IMPLEMENTED"""
    raise NotImplementedError


# Other Apps


@app.command()
def stamp(
    input_root: Annotated[
        Path,
        typer.Option(
            help="Path to root input directory.",
            show_default=False,
            exists=True,
            file_okay=False,
            rich_help_panel="Paths",
            resolve_path=True,
        ),
    ],
    field: Annotated[
        str,
        typer.Option(
            "--field",
            "-F",
            show_default=False,
            help="Field over which to generate stamps.",
            rich_help_panel="FICL",
        ),
    ],
    image_version: Annotated[
        str,
        typer.Option(
            "--image-version",
            "-I",
            show_default=False,
            help="Version of image processing over which to generate stamps.",
            rich_help_panel="FICL",
        ),
    ],
    catalog_version: Annotated[
        str,
        typer.Option(
            "--catalog-version",
            "-C",
            show_default=False,
            help="Version of cataloguing from which object IDs are taken.",
            rich_help_panel="FICL",
        ),
    ],
    filter: Annotated[
        str,
        typer.Option(
            "--filter",
            "-L",
            show_default=False,
            help="Filter over which to generate stamps.",
            rich_help_panel="FICL",
        ),
    ],
    product_root: Annotated[
        Path,
        typer.Option(
            help="Path to root product directory.",
            show_default='"products" directory created at --input-root level',
            exists=True,
            file_okay=False,
            rich_help_panel="Paths",
            resolve_path=True,
            writable=True,
        ),
    ] = None,
    output_root: Annotated[
        Path,
        typer.Option(
            help="Path to root output directory.",
            show_default='"output" directory created at --input-root level',
            exists=True,
            file_okay=False,
            rich_help_panel="Paths",
            resolve_path=True,
            writable=True,
        ),
    ] = None,
):
    """[IN DEVELOPMENT] Generate stamp cutouts of all objects from a given FICL."""
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
            run_root=morphfits_config.run_root,
            datetime=morphfits_config.datetime,
            run_number=morphfits_config.run_number,
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
        display_progress=True,
    )

    # Plot all objects
    plots.plot_objects(
        output_root=morphfits_config.output_root,
        product_root=morphfits_config.product_root,
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
