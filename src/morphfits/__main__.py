"""Main program execution for morphology fitting.
"""

# Imports


import logging
from pathlib import Path
from typing import Annotated, Optional, List

import typer

from . import catalog, initialize, plot, products, settings
from .wrappers import galfit


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
        Optional[typer.FileBinaryRead],
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
    settings_path: Annotated[
        Optional[typer.FileText],
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
        Optional[Path],
        typer.Option(
            "--root",
            "-r",
            help="Path to MorphFITS filesystem root.",
            rich_help_panel="Paths",
            file_okay=False,
            show_default=False,
            resolve_path=True,
        ),
    ] = None,
    input_root: Annotated[
        Optional[Path],
        typer.Option(
            "--input",
            "-i",
            help="Path to root input directory.",
            rich_help_panel="Paths",
            file_okay=False,
            show_default=False,
            resolve_path=True,
        ),
    ] = None,
    output_root: Annotated[
        Optional[Path],
        typer.Option(
            "--output",
            help="Path to root output directory.",
            rich_help_panel="Paths",
            file_okay=False,
            show_default=False,
            resolve_path=True,
            writable=True,
        ),
    ] = None,
    product_root: Annotated[
        Optional[Path],
        typer.Option(
            "--product",
            help="Path to root products directory.",
            rich_help_panel="Paths",
            file_okay=False,
            show_default=False,
            resolve_path=True,
            writable=True,
        ),
    ] = None,
    run_root: Annotated[
        Optional[Path],
        typer.Option(
            "--run",
            help="Path to root runs directory.",
            rich_help_panel="Paths",
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
    first_object: Annotated[
        Optional[int],
        typer.Option(
            "--first-object",
            help="ID of first object over which to run MorphFITS.",
            rich_help_panel="Batch Runs",
            show_default=False,
        ),
    ] = None,
    last_object: Annotated[
        Optional[int],
        typer.Option(
            "--last-object",
            help="ID of last object over which to run MorphFITS.",
            rich_help_panel="Batch Runs",
            show_default=False,
        ),
    ] = None,
    batch_n_process: Annotated[
        Optional[int],
        typer.Option(
            "--batch-n-process",
            "-n",
            help="Number of cores over which to divide a program run.",
            rich_help_panel="Batch Runs",
            show_default=False,
        ),
    ] = None,
    batch_process_id: Annotated[
        Optional[int],
        typer.Option(
            "--batch-process-id",
            "-p",
            help="Process number in batch run (out of batch_n_process).",
            rich_help_panel="Batch Runs",
            show_default=False,
        ),
    ] = None,
    skip_unzip: Annotated[
        Optional[bool],
        typer.Option(
            "--skip-unzip",
            help="Skip unzipping any zipped observation files.",
            rich_help_panel="Stages",
            is_flag=True,
        ),
    ] = None,
    skip_product: Annotated[
        Optional[bool],
        typer.Option(
            "--skip-product",
            help="Skip making any product files.",
            rich_help_panel="Stages",
            is_flag=True,
        ),
    ] = None,
    skip_morphology: Annotated[
        Optional[bool],
        typer.Option(
            "--skip-morphology",
            help="Skip running any morphology fitting.",
            rich_help_panel="Stages",
            is_flag=True,
        ),
    ] = None,
    skip_catalog: Annotated[
        Optional[bool],
        typer.Option(
            "--skip-catalog",
            help="Skip writing any catalog files.",
            rich_help_panel="Stages",
            is_flag=True,
        ),
    ] = None,
    skip_histogram: Annotated[
        Optional[bool],
        typer.Option(
            "--skip-histogram",
            help="Skip making any histogram plots.",
            rich_help_panel="Stages",
            is_flag=True,
        ),
    ] = None,
    skip_plot: Annotated[
        Optional[bool],
        typer.Option(
            "--skip-plot",
            help="Skip making any model plots.",
            rich_help_panel="Stages",
            is_flag=True,
        ),
    ] = None,
    skip_cleanup: Annotated[
        Optional[bool],
        typer.Option(
            "--skip-cleanup",
            help="Skip cleaning up the directory structure.",
            rich_help_panel="Stages",
            is_flag=True,
        ),
    ] = None,
    remake_all: Annotated[
        Optional[bool],
        typer.Option(
            "--remake-all",
            help="Remake all products and overwrite existing. Overrides other 'remake' flags.",
            rich_help_panel="Remake",
            is_flag=True,
        ),
    ] = None,
    remake_stamps: Annotated[
        Optional[bool],
        typer.Option(
            "--remake-stamps",
            help="Remake stamps and overwrite existing.",
            rich_help_panel="Remake",
            is_flag=True,
        ),
    ] = None,
    remake_sigmas: Annotated[
        Optional[bool],
        typer.Option(
            "--remake-sigmas",
            help="Remake sigma maps and overwrite existing.",
            rich_help_panel="Remake",
            is_flag=True,
        ),
    ] = None,
    remake_psfs: Annotated[
        Optional[bool],
        typer.Option(
            "--remake-psfs",
            help="Remake PSF crops and overwrite existing.",
            rich_help_panel="Remake",
            is_flag=True,
        ),
    ] = None,
    remake_masks: Annotated[
        Optional[bool],
        typer.Option(
            "--remake-masks",
            help="Remake masks and overwrite existing.",
            rich_help_panel="Remake",
            is_flag=True,
        ),
    ] = None,
    remake_morphology: Annotated[
        Optional[bool],
        typer.Option(
            "--remake-morphology",
            help="Remake morphology fit models and overwrite existing.",
            rich_help_panel="Remake",
            is_flag=True,
        ),
    ] = None,
    remake_plots: Annotated[
        Optional[bool],
        typer.Option(
            "--remake-plots",
            help="Remake plots and overwrite existing.",
            rich_help_panel="Remake",
            is_flag=True,
        ),
    ] = None,
    remake_feedfiles: Annotated[
        Optional[bool],
        typer.Option(
            "--remake-feedfiles",
            help="Remake GALFIT feedfiles and overwrite existing.",
            rich_help_panel="Remake",
            is_flag=True,
        ),
    ] = None,
    scale: Annotated[
        Optional[float],
        typer.Option(
            "--scale",
            help="Increase the image size of stamps by this scale factor.",
            rich_help_panel="Science",
            show_default="20",
        ),
    ] = None,
    psf_copy: Annotated[
        Optional[bool],
        typer.Option(
            "--copy-psfs",
            help="Make one PSF per filter rather than per object.",
            rich_help_panel="Science",
            is_flag=True,
            show_default="True",
        ),
    ] = None,
    psf_size: Annotated[
        Optional[int],
        typer.Option(
            "--psf-size",
            help="Size of per-filter PSF crop (only used when --copy-psfs raised).",
            rich_help_panel="Science",
            show_default="80",
        ),
    ] = None,
    boost: Annotated[
        Optional[float],
        typer.Option(
            "--boost",
            help="Boost the initial estimate by this amount.",
            rich_help_panel="Science",
            show_default="1.0",
        ),
    ] = None,
    log_level: Annotated[
        Optional[str],
        typer.Option(
            "--log-level",
            help="Logging level at which to write logs to file, "
            + "one of the standard Python levels.",
            show_default=False,
        ),
    ] = None,
    progress_bar: Annotated[
        Optional[bool],
        typer.Option(
            "--progress",
            help="Display progress as a loading bar and suppress per-object logging.",
            is_flag=True,
        ),
    ] = None,
):
    # Create settings objects
    runtime_settings, science_settings = settings.get_settings(
        settings_path=settings_path,
        morphfits_root=morphfits_root,
        input_root=input_root,
        output_root=output_root,
        product_root=product_root,
        run_root=run_root,
        batch_n_process=batch_n_process,
        batch_process_id=batch_process_id,
        fields=fields,
        image_versions=image_versions,
        catalog_versions=catalog_versions,
        filters=filters,
        objects=objects,
        first_object=first_object,
        last_object=last_object,
        progress_bar=progress_bar,
        log_level=log_level,
        skip_unzip=skip_unzip,
        skip_product=skip_product,
        skip_morphology=skip_morphology,
        skip_catalog=skip_catalog,
        skip_histogram=skip_histogram,
        skip_plot=skip_plot,
        skip_cleanup=skip_cleanup,
        remake_all=remake_all,
        remake_stamps=remake_stamps,
        remake_sigmas=remake_sigmas,
        remake_psfs=remake_psfs,
        remake_masks=remake_masks,
        remake_morphology=remake_morphology,
        remake_plots=remake_plots,
        remake_others=remake_feedfiles,
        morphology="galfit",
        galfit_path=galfit_path,
        scale=scale,
        psf_copy=psf_copy,
        psf_size=psf_size,
        boost=boost,
        initialized=True,
    )

    # Display status
    logger = logging.getLogger("MORPHFITS")
    logger.info("Starting MorphFITS.")
    if runtime_settings.process_count > 1:
        logger.info("Running in batch mode.")
        logger.info(
            f"Batch process: {runtime_settings.process_id} "
            + f"/ {runtime_settings.process_count-1}"
        )

    # Unzip zipped files
    if runtime_settings.stages.unzip:
        initialize.unzip_all(runtime_settings=runtime_settings)

    # Create product files
    if runtime_settings.stages.product:
        products.make_all(
            runtime_settings=runtime_settings, science_settings=science_settings
        )

    # Create feedfiles
    if runtime_settings.stages.product:
        galfit.make_all_feedfiles(
            runtime_settings=runtime_settings, science_settings=science_settings
        )

    # Run GALFIT
    if runtime_settings.stages.morphology:
        galfit.run_all(runtime_settings=runtime_settings)

    # Write catalogs
    if runtime_settings.stages.catalog:
        catalog.make_all(runtime_settings=runtime_settings)

    # Plot histograms
    if runtime_settings.stages.histogram:
        plot.all_histograms(runtime_settings=runtime_settings)

    # Plot models
    if runtime_settings.stages.plot:
        plot.all_models(runtime_settings=runtime_settings)

    # Remove empty directories
    if runtime_settings.stages.cleanup:
        runtime_settings.cleanup_directories()

    # Write settings to file in run directory
    if runtime_settings.stages.cleanup:
        science_settings.write(runtime_settings=runtime_settings)
        runtime_settings.write()

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


@app.command(
    name="initialize",
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
def initialize_command(
    settings_path: Annotated[
        Optional[typer.FileText],
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
        Optional[Path],
        typer.Option(
            "--root",
            "-r",
            help="Path to MorphFITS filesystem root.",
            rich_help_panel="Paths",
            file_okay=False,
            show_default=False,
            resolve_path=True,
        ),
    ] = None,
    input_root: Annotated[
        Optional[Path],
        typer.Option(
            "--input",
            "-i",
            help="Path to root input directory.",
            rich_help_panel="Paths",
            file_okay=False,
            show_default=False,
            resolve_path=True,
        ),
    ] = None,
    run_root: Annotated[
        Optional[Path],
        typer.Option(
            "--run",
            help="Path to root runs directory.",
            rich_help_panel="Paths",
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
    skip_morphology: Annotated[
        Optional[bool],
        typer.Option(
            "--skip-download",
            help="Skip downloading any files from DJA.",
            rich_help_panel="Stages",
            is_flag=True,
        ),
    ] = None,
    skip_unzip: Annotated[
        Optional[bool],
        typer.Option(
            "--skip-unzip",
            help="Skip unzipping any zipped observation files.",
            rich_help_panel="Stages",
            is_flag=True,
        ),
    ] = None,
    log_level: Annotated[
        Optional[str],
        typer.Option(
            "--log-level",
            help="Logging level at which to write logs to file, "
            + "one of the standard Python levels.",
            show_default=False,
        ),
    ] = None,
):
    # Create settings objects
    runtime_settings, science_settings = settings.get_settings(
        settings_path=settings_path,
        morphfits_root=morphfits_root,
        input_root=input_root,
        fields=fields,
        image_versions=image_versions,
        catalog_versions=catalog_versions,
        filters=filters,
        skip_morphology=skip_morphology,
        skip_unzip=skip_unzip,
        log_level=log_level,
    )

    # Display status
    logger = logging.getLogger("MORPHFITS")
    logger.info("Starting MorphFITS.")

    # Get list of all available files for download from DJA
    dja_catalog = initialize.get_dja_catalog()

    # Get dict of all src to dest for download
    src_dest = initialize.get_src_dest(runtime_settings, dja_catalog)

    # Download files
    if runtime_settings.stages.morphology:
        initialize.get_input(src_dest)

    # Unzip zipped files
    if runtime_settings.stages.unzip:
        initialize.unzip_all(runtime_settings)

    # Write settings to file in run directory
    runtime_settings.write()

    # Exit
    logger.info("Exiting MorphFITS.")


@app.command(
    name="product",
    short_help="Create FICLO products (cutouts for each object).",
    help="Create products for each FICLO. A product is an intermediate "
    + "FITS file required for a morphology fitting algorithm. "
    + "This includes stamps, sigma maps, PSF crops, and masks "
    + "for each object in a FICL.",
    rich_help_panel="Tools",
    hidden=True,
)
def product_command():
    raise NotImplementedError


@app.command(
    name="catalog",
    short_help="Create morphology catalogs.",
    help="Update the full merge catalog at output/catalogs/merge "
    + "and per-FIC catalogs at output/catalogs/morphology.",
    rich_help_panel="Tools",
    hidden=True,
)
def catalog_command():
    raise NotImplementedError


## Main


def main():
    app()


if __name__ == "__main__":
    main()
