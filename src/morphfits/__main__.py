"""Main program execution for morphology fitting.
"""

# Imports


import logging
from pathlib import Path
from typing import Annotated, Optional, List

import typer

from . import download, plot, products, settings
from .wrappers import galfit
from .utils import logs, paths


# App Instantiation


app = typer.Typer(add_completion=False, no_args_is_help=True)
"""Primary app of program.
"""


# Commands


## Morphology


@app.command(
    short_help="Model galaxy and cluster objects using GALFIT.",
    help="Model the morphology of identified galaxy and cluster objects "
    + "from JWST observations using GALFIT. "
    + "Creates cutouts, deviations, PSF crops, masks, and models "
    + "of each object, as well as catalogs and histograms of each fitting, "
    + "and optional plots.",
    rich_help_panel="Morphology",
)
def galwrap(
    galfit_path: Annotated[
        typer.FileBinaryRead,
        typer.Option(
            "--galfit",
            "-g",
            help="Path to GALFIT binary executable file.",
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
            "--config",
            "-c",
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
            "--root",
            "-r",
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
            "--input",
            "-i",
            help="Path to root input directory.",
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
            "--output",
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
            "--product",
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
            "--run",
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
            "-n",
            help="Number of cores over which to divide a program run.",
            rich_help_panel="Batch Runs",
            show_default=False,
        ),
    ] = 1,
    batch_process_id: Annotated[
        int,
        typer.Option(
            "--batch-process-id",
            "-p",
            help="Process number in batch run (out of batch_n_process).",
            rich_help_panel="Batch Runs",
            show_default=False,
        ),
    ] = 0,
    remake_all: Annotated[
        bool,
        typer.Option(
            "--remake-all",
            help="Remake all products and overwrite existing. Overrides other 'remake' flags.",
            rich_help_panel="Products",
            is_flag=True,
        ),
    ] = False,
    remake_stamps: Annotated[
        bool,
        typer.Option(
            "--remake-stamps",
            help="Remake stamps and overwrite existing.",
            rich_help_panel="Products",
            is_flag=True,
        ),
    ] = False,
    remake_sigmas: Annotated[
        bool,
        typer.Option(
            "--remake-sigmas",
            help="Remake sigma maps and overwrite existing.",
            rich_help_panel="Products",
            is_flag=True,
        ),
    ] = False,
    remake_psfs: Annotated[
        bool,
        typer.Option(
            "--remake-psfs",
            help="Remake PSF crops and overwrite existing.",
            rich_help_panel="Products",
            is_flag=True,
        ),
    ] = False,
    remake_masks: Annotated[
        bool,
        typer.Option(
            "--remake-masks",
            help="Remake masks and overwrite existing.",
            rich_help_panel="Products",
            is_flag=True,
        ),
    ] = False,
    remake_feedfiles: Annotated[
        bool,
        typer.Option(
            "--remake-feedfiles",
            help="Remake GALFIT feedfiles and overwrite existing.",
            rich_help_panel="Products",
            is_flag=True,
        ),
    ] = False,
    refit: Annotated[
        bool,
        typer.Option(
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
    progress_bar: Annotated[
        bool,
        typer.Option(
            "--progress",
            help="Display progress as a loading bar and suppress per-object logging.",
            is_flag=True,
        ),
    ] = False,
):
    # Create configuration object
    morphfits_config = settings.create_config(
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
        display_progress=progress_bar,
    )

    # Display status
    logger = logging.getLogger("MORPHFITS")

    # Unzip zipped files
    download.unzip_files(morphfits_config=morphfits_config)

    # Create product files
    products.make_all(
        morphfits_config=morphfits_config,
        remake_all=remake_all,
        remake_stamps=remake_stamps,
        remake_sigmas=remake_sigmas,
        remake_psfs=remake_psfs,
        remake_masks=remake_masks,
        progress_bar=progress_bar,
    )

    # Create feedfiles

    # Run GALFIT

    # Write catalogs

    # Plot histograms

    # Plot models

    # Remove empty directories
    morphfits_config.clean_paths(display_progress=progress_bar)

    # Write configuration to file in run directory
    morphfits_config.write()

    # Exit
    logger.info("Exiting MorphFITS.")


@app.command(
    short_help="Model galaxy and cluster objects using imcascade.",
    help="Model the morphology of identified galaxy and cluster objects "
    + "from JWST observations using imcascade. "
    + "Creates cutouts, deviations, PSF crops, masks, and models "
    + "of each object, as well as catalogs and histograms of each fitting, "
    + "and optional plots.",
    rich_help_panel="Morphology",
    hidden=True,
)
def imcascade():
    raise NotImplementedError


@app.command(
    short_help="Model galaxy and cluster objects using pysersic.",
    help="Model the morphology of identified galaxy and cluster objects "
    + "from JWST observations using pysersic. "
    + "Creates cutouts, deviations, PSF crops, masks, and models "
    + "of each object, as well as catalogs and histograms of each fitting, "
    + "and optional plots.",
    rich_help_panel="Morphology",
    hidden=True,
)
def pysersic():
    raise NotImplementedError


## Tools


# TODO rename
@app.command(
    short_help="Download and unzip input files from the DJA archive.",
    help="Download input files required for a morphology fitting run, "
    + "unzip them, and organize them as per the MorphFITS directory structure. "
    + "Required files downloaded are the input catalog, segmentation map, "
    + "science image, and exposure and weights maps. "
    + "Note the simulated PSF must be manually downloaded from STSci's "
    + "drop box and moved to the appropriate location. "
    + "For further details, consult the README.",
    rich_help_panel="Tools",
)
def get(
    config_path: Annotated[
        typer.FileText,
        typer.Option(
            "--config",
            "-c",
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
            "--input",
            "-i",
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
    # Create configuration object
    morphfits_config = settings.create_config(
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
    download.main(
        morphfits_config=morphfits_config,
        skip_download=skip_download,
        skip_unzip=skip_unzip,
        overwrite=overwrite,
    )

    # Exit
    logger.info("Exiting MorphFITS.")


# TODO rename
@app.command(
    short_help="Create FICLO products (cutouts for each object).",
    help="Create products for each FICLO. A product is an intermediate "
    + "FITS file required for a morphology fitting algorithm. "
    + "This includes stamps, sigma maps, PSF crops, and masks "
    + "for each object in a FICL.",
    rich_help_panel="Tools",
    hidden=True,
)
def get_products():
    raise NotImplementedError


# TODO rename
@app.command(
    short_help="Visualize each object in a field.",
    help="Create a plot visualizing each object in a FICL.",
    rich_help_panel="Tools",
    hidden=True,
)
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
    # Create configuration object
    morphfits_config = settings.create_config(
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
            "run_log",
            run_root=morphfits_config.run_root,
            datetime=morphfits_config.datetime,
            run_number=morphfits_config.run_number,
        )
    )
    logger = logging.getLogger("MORPHFITS")
    logger.info("Starting MorphFITS.")

    # Display progress
    ficl = next(morphfits_config.ficls)
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
    plot.plot_objects(
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


## Main


def main():
    app()


if __name__ == "__main__":
    main()
