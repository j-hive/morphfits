"""Main program execution for morphology fitting.
"""

# Imports


import logging
from typing import Optional, List
from typing_extensions import Annotated
from pathlib import Path

import typer

from . import config, input, paths, plots, products, ROOT
from .wrappers import galfit
from .utils import logs


# App Instantiation
app = typer.Typer()


# Morphology Fitters


@app.command()
def galwrap(
    galfit_path: Annotated[
        typer.FileBinaryRead,
        typer.Option(
            help="Path to GALFIT binary file.",
            rich_help_panel="Paths",
            exists=True,
            dir_okay=False,
            show_default=False,
            resolve_path=True,
        ),
    ] = None,
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
            "--field",
            "-F",
            help="Fields over which to run MorphFITS.",
            rich_help_panel="FICLOs",
            show_default=False,
        ),
    ] = None,
    image_versions: Annotated[
        Optional[List[str]],
        typer.Option(
            "--image-version",
            "-I",
            help="Image versions over which to run MorphFITS.",
            rich_help_panel="FICLOs",
            show_default=False,
        ),
    ] = None,
    catalog_versions: Annotated[
        Optional[List[str]],
        typer.Option(
            "--catalog-version",
            "-C",
            help="Catalog versions over which to run MorphFITS.",
            rich_help_panel="FICLOs",
            show_default=False,
        ),
    ] = None,
    filters: Annotated[
        Optional[List[str]],
        typer.Option(
            "--filter",
            "-L",
            help="Filters over which to run MorphFITS.",
            rich_help_panel="FICLOs",
            show_default=False,
        ),
    ] = None,
    objects: Annotated[
        Optional[List[int]],
        typer.Option(
            "--object",
            "-O",
            help="Object IDs over which to run MorphFITS.",
            rich_help_panel="FICLOs",
            show_default=False,
        ),
    ] = None,
    object_first: Annotated[
        Optional[int],
        typer.Option(
            "--first-object",
            help="ID of first object over which to run MorphFITS.",
            rich_help_panel="Batch Runs",
            show_default=False,
        ),
    ] = None,
    object_last: Annotated[
        Optional[int],
        typer.Option(
            "--last-object",
            help="ID of last object over which to run MorphFITS.",
            rich_help_panel="Batch Runs",
            show_default=False,
        ),
    ] = None,
    batch_n_process: Annotated[
        int,
        typer.Option(
            "--batch-n-process",
            help="Number of cores over which to divide a program run.",
            rich_help_panel="Batch Runs",
            show_default=False,
        ),
    ] = 1,
    batch_process_id: Annotated[
        int,
        typer.Option(
            "--batch-process-id",
            help="Process number in batch run (out of batch_n_process).",
            rich_help_panel="Batch Runs",
            show_default=False,
        ),
    ] = 0,
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
    force_refit: Annotated[
        bool,
        typer.Option(
            "--force-refit",
            help="Run GALFIT over previously fitted objects and overwrite existing models.",
            rich_help_panel="Stages",
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
    make_plots: Annotated[
        bool,
        typer.Option(
            "--make-plots",
            help="Generate model visualizations.",
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
        object_first=object_first,
        object_last=object_last,
        batch_n_process=batch_n_process,
        batch_process_id=batch_process_id,
        galfit_path=galfit_path,
        display_progress=display_progress,
    )

    # Display status
    logger = logging.getLogger("MORPHFITS")

    # Call wrapper
    galfit.main(
        morphfits_config=morphfits_config,
        regenerate_products=regenerate_products,
        regenerate_stamps=regenerate_stamps,
        regenerate_psfs=regenerate_psfs,
        regenerate_masks=regenerate_masks,
        regenerate_sigmas=regenerate_sigmas,
        keep_feedfiles=keep_feedfiles,
        force_refit=force_refit,
        skip_products=skip_products,
        skip_fits=skip_fits,
        make_plots=make_plots,
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
    """[NOT IMPLEMENTED] Model objects from given FICLs using imcascade."""
    raise NotImplementedError


@app.command()
def pysersic():
    """[NOT IMPLEMENTED] Model objects from given FICLs using pysersic."""
    raise NotImplementedError


# Other Commands


@app.command()
def download(
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
    input_root: Annotated[
        Path,
        typer.Option(
            help="Path to root input directory. Must be set here or in --config-path.",
            rich_help_panel="Paths",
            file_okay=False,
            show_default=False,
            resolve_path=True,
        ),
    ] = None,
    fields: Annotated[
        Optional[List[str]],
        typer.Option(
            "--field",
            "-F",
            help="Fields over which to run MorphFITS.",
            rich_help_panel="FICLOs",
            show_default=False,
        ),
    ] = None,
    image_versions: Annotated[
        Optional[List[str]],
        typer.Option(
            "--image-version",
            "-I",
            help="Image versions over which to run MorphFITS.",
            rich_help_panel="FICLOs",
            show_default=False,
        ),
    ] = None,
    filters: Annotated[
        Optional[List[str]],
        typer.Option(
            "--filter",
            "-L",
            help="Filters over which to run MorphFITS.",
            rich_help_panel="FICLOs",
            show_default=False,
        ),
    ] = None,
    skip_download: Annotated[
        bool,
        typer.Option(
            "--skip-download",
            help="Skip downloading files (only unzip existing files).",
            rich_help_panel="Stages",
            show_default=False,
        ),
    ] = False,
    skip_unzip: Annotated[
        bool,
        typer.Option(
            "--skip-unzip",
            help="Skip unzipping files (only download files).",
            rich_help_panel="Stages",
            show_default=False,
        ),
    ] = False,
    overwrite: Annotated[
        bool,
        typer.Option(
            "--overwrite",
            help="Overwrite existing downloads.",
            rich_help_panel="Stages",
            show_default=False,
        ),
    ] = False,
):
    """Download and unzip input files from the DJA archive."""

    # Create configuration object
    morphfits_config = config.create_config(
        config_path=config_path,
        input_root=input_root,
        fields=fields,
        image_versions=image_versions,
        filters=filters,
        wrappers=[""],
        galfit_path="",
        display_progress=False,
        download=True,
    )

    # Create program and module logger
    logger = logging.getLogger("MORPHFITS")

    # Download and unzip files
    input.main(
        morphfits_config=morphfits_config,
        skip_download=skip_download,
        skip_unzip=skip_unzip,
        overwrite=overwrite,
    )

    # Exit
    logger.info("Exiting MorphFITS.")


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
