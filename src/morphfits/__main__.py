"""Main program execution for morphology fitting.
"""

# Imports


import logging
from pathlib import Path
from typing import Union, Optional

import typer

from .utils import logs
from .galwrap.main import main as galwrap_main


# Instantiations


## Create logger for program and module
main_logger = logs.create_logger()
logger = logging.getLogger("MORPHFITS")

## Create typer app
app = typer.Typer()


# Morphology Fitters


@app.command()
def galwrap(
    config_path: Optional[str] = None,
    galwrap_root: Optional[str] = None,
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
    psf_factor: int = 4,
):
    """Command to invoke GalWrap program for MorphFITS.

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
    galwrap_main(
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
    logger.info("Exiting MorphFITS.")


@app.command()
def pyfits(to_be_implemented: int):
    pass


# Main Program


if __name__ == "__main__":
    logger.info("Starting MorphFITS.")
    app()
