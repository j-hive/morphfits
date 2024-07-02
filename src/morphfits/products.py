"""Create J-HIVE MorphFITS GalWrap products for GALFIT usage.
"""

# Imports


import gc
import logging
import warnings
from pathlib import Path

import numpy as np
from scipy import ndimage
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata.utils import Cutout2D
from astropy.table import Table
from astropy.wcs import WCS
from jinja2 import Template
from tqdm import tqdm

from . import paths
from .config import MorphFITSConfig
from .wrappers import galfit
from .utils import science


# Constants


logger = logging.getLogger("PRODUCTS")
"""Logger object for this module.
"""


np.seterr(divide="ignore", invalid="ignore")
warnings.filterwarnings("ignore")
"""Ignore zero division warnings encountered in generate_sigmas.
"""


# Functions


## Products


def generate_stamps(
    input_root: Path,
    product_root: Path,
    field: str,
    image_version: str,
    catalog_version: str,
    filter: str,
    objects: list[str],
    minimum_image_size: int = 32,
    kron_factor: int = 3,
    regenerate: bool = False,
    display_progress: bool = False,
) -> tuple[list[int], list[SkyCoord], list[int]]:
    """Generate stamps (science frame cutouts) for all objects in a FICL.

    Parameters
    ----------
    input_root : Path
        Path to root GalWrap input directory.
    product_root : Path
        Path to root GalWrap products directory.
    field : str
        Field of observation.
    image_version : str
        Version of image processing used on observation.
    catalog_version : str
        Version of cataloguing used for field.
    filter : str
        Filter used for observation.
    objects : list[str]
        List of object IDs in catalog to stamp from observation.
    minimum_image_size : int, optional
        Minimum square stamp pixel dimensions, by default 32.
    kron_factor : int, optional
        Multiplicative factor to apply to Kron radius for each object to
        determine image size, by default 3.
    regenerate : bool, optional
        Regenerate existing stamps, by default False.
    display_progress : bool, optional
        Display progress on terminal screen via tqdm, by default False.

    Returns
    -------
    tuple[list[int], list[SkyCoord], list[int]]
        List of object IDs, positions, and image sizes for successful stamps.
    """
    logger.info("Generating stamps.")

    # Load in catalog
    catalog_path = paths.get_path(
        "catalog", input_root=input_root, field=field, image_version=image_version
    )
    catalog = Table.read(catalog_path)

    # Load in image and header data
    science_path = paths.get_path(
        "science",
        input_root=input_root,
        field=field,
        image_version=image_version,
        filter=filter,
    )
    science_file = fits.open(science_path)
    image, wcs = science_file["PRIMARY"].data, WCS(science_file["PRIMARY"].header)

    # Clear file from memory
    science_file.close()
    del science_file
    del science_path
    del catalog_path
    gc.collect()

    # Iterate over each object
    generated, skipped = ([], [], []), []
    for object in (
        tqdm(objects, unit="stamp", leave=False) if display_progress else objects
    ):
        # Record object position from catalog
        position = SkyCoord(
            ra=catalog[object]["ra"], dec=catalog[object]["dec"], unit="deg"
        )

        # Determine image size
        kron_radius = catalog[object][
            (
                "kron_radius_circ"
                if "kron_radius_circ" in catalog.keys()
                else "kron_radius"
            )
        ]
        image_size = np.nanmax(
            [
                int(kron_radius / 0.04 * kron_factor),
                minimum_image_size,
            ]
        )

        # Skip objects that have already been stamped
        stamp_path = paths.get_path(
            "stamp",
            product_root=product_root,
            field=field,
            image_version=image_version,
            catalog_version=catalog_version,
            filter=filter,
            object=object,
        )
        if stamp_path.exists() and not regenerate:
            if not display_progress:
                logger.debug(f"Skipping object {object}, stamp exists.")
            generated[0].append(object)
            generated[1].append(position)
            generated[2].append(image_size)
            skipped.append(object)
            continue

        # Try generating stamp for object
        try:
            if not display_progress:
                logger.info(f"Generating stamp for object {object}.")

            # Generate stamp
            stamp = Cutout2D(data=image, position=position, size=image_size, wcs=wcs)

            # Write stamp to disk if image nonzero and of correct shape
            if (np.amax(stamp.data) > 0) and (
                stamp.data.shape == (image_size, image_size)
            ):
                stamp_headers = stamp.wcs.to_header()
                stamp_headers["EXPTIME"] = 1

                stamp_hdul = fits.PrimaryHDU(data=stamp.data, header=stamp_headers)
                stamp_hdul.writeto(stamp_path, overwrite=True)

                # Store object ID, position, and image size for other products
                generated[0].append(object)
                generated[1].append(position)
                generated[2].append(image_size)
                del stamp_headers
                del stamp_hdul
            else:
                if np.amax(stamp.data) <= 0:
                    logger.debug(f"Skipping object {object}, missing nonzero data.")
                else:
                    logger.debug(
                        f"Skipping object {object}, dimensions "
                        + f"{stamp.data.shape} don't match expected image size {image_size}."
                    )
                skipped.append(object)

            # Clear memory
            del position
            del kron_radius
            del image_size
            del stamp_path
            del stamp
            gc.collect()

        # Catch skipped objects
        except Exception as e:
            logger.error(e)
            skipped.append(object)

    # Display skipped objects
    if len(skipped) > 0:
        logger.debug(
            f"Skipped generating stamps for {len(skipped)} objects: {skipped}."
        )

    # Return stored information from each object
    return generated


def generate_sigmas(
    input_root: Path,
    product_root: Path,
    field: str,
    image_version: str,
    catalog_version: str,
    filter: str,
    objects: list[int],
    positions: list[SkyCoord],
    image_sizes: list[int],
    regenerate: bool = False,
    display_progress: bool = False,
):
    """Generate sigma maps for all objects in a FICL.

    Parameters
    ----------
    input_root : Path
        Path to root GalWrap input directory.
    product_root : Path
        Path to root GalWrap products directory.
    field : str
        Field of observation.
    image_version : str
        Version of image processing used on observation.
    catalog_version : str
        Version of cataloguing used for field.
    filter : str
        Filter used for observation.
    objects : list[str]
        List of object IDs in catalog for which to generate sigma maps.
    positions : list[SkyCoord]
        List of positions of objects in the sky, from catalog.
    image_sizes : list[int]
        List of image sizes corresponding to each object's stamp.
    regenerate : bool, optional
        Regenerate existing sigma maps, by default False.
    display_progress : bool, optional
        Display progress on terminal screen via tqdm, by default False.

    See Also
    --------
    This algorithm was taken from `the DJA blog <https://dawn-cph.github.io/dja/blog/2023/07/18/image-data-products/>`_.
    """
    logger.info("Generating sigma maps.")

    # Load in exposure and weights maps
    exposure_path = paths.get_path(
        "exposure",
        input_root=input_root,
        field=field,
        image_version=image_version,
        filter=filter,
    )
    exposure_file = fits.open(exposure_path)
    exposure_data = exposure_file["PRIMARY"].data
    exposure_headers = exposure_file["PRIMARY"].header
    exposure_wcs = WCS(exposure_headers)

    weights_path = paths.get_path(
        "weights",
        input_root=input_root,
        field=field,
        image_version=image_version,
        filter=filter,
    )
    weights_file = fits.open(weights_path)
    weights_data = weights_file["PRIMARY"].data
    weights_wcs = WCS(weights_file["PRIMARY"].header)

    # Close and clear files from memory
    exposure_file.close()
    weights_file.close()
    del exposure_path
    del weights_path
    del exposure_file
    del weights_file
    gc.collect()

    # Iterate over each object, position, and image_size tuple
    skipped = []
    for i in (
        tqdm(range(len(objects)), unit="sigma", leave=False)
        if display_progress
        else range(len(objects))
    ):
        object, position, image_size = objects[i], positions[i], image_sizes[i]

        # Skip objects which already have sigma maps
        sigma_path = paths.get_path(
            "sigma",
            product_root=product_root,
            field=field,
            image_version=image_version,
            catalog_version=catalog_version,
            filter=filter,
            object=object,
        )
        if sigma_path.exists() and not regenerate:
            if not display_progress:
                logger.debug(f"Skipping object {object}, sigma map exists.")
            skipped.append(object)
            continue

        # Try generating sigma map for object
        try:
            if not display_progress:
                logger.info(f"Generating sigma map for object {object}.")

            # Load stamp for this object and filter over minimum of 0
            stamp_path = paths.get_path(
                "stamp",
                product_root=product_root,
                field=field,
                image_version=image_version,
                catalog_version=catalog_version,
                filter=filter,
                object=object,
            )
            stamp_file = fits.open(stamp_path)
            stamp_data = stamp_file["PRIMARY"].data
            maximized_stamp = np.maximum(stamp_data, 0)

            # Clear memory
            stamp_file.close()
            del stamp_path
            del stamp_file
            del stamp_data
            gc.collect()

            # Generate cutout for exposure map, which is 4x smaller than science
            quarter_image_size = int(image_size / 4)
            exposure_image_size = (
                quarter_image_size + 1
                if int((image_size + 1) / 4) > quarter_image_size
                else quarter_image_size
            )
            exposure_cutout = Cutout2D(
                data=exposure_data,
                position=position,
                size=exposure_image_size,
                wcs=exposure_wcs,
            )

            # Grow exposure cutout to same size as science -> s
            zeroes = np.zeros(shape=(image_size, image_size), dtype=int)
            zeroes[2::4, 2::4] += exposure_cutout.data
            full_exposure = ndimage.maximum_filter(input=zeroes, size=4)
            del zeroes
            del exposure_cutout
            gc.collect()

            # Find multiplicative factors applied to original count rate data
            scale_factor = 1.0
            for header in ["PHOTMJSR", "PHOTSCAL", "OPHOTFNU"]:
                if header in exposure_headers:
                    scale_factor /= exposure_headers[header]
                if header == "OPHOTFNU":
                    scale_factor *= exposure_headers["PHOTFNU"]

            # Calculate effective gain - electrons per DN of the mosaic
            effective_gain = scale_factor * full_exposure
            del scale_factor
            del full_exposure
            gc.collect()

            # Calculate Poisson variance in mosaic DN -> 10nJy / s
            poisson_variance = maximized_stamp / effective_gain
            del maximized_stamp
            del effective_gain
            gc.collect()

            # Generate cutout for weights
            weights_cutout = Cutout2D(
                data=weights_data,
                position=position,
                size=image_size,
                wcs=weights_wcs,
            )

            # Calculate original variance from weights map -> 1 / 10nJy
            weights_variance = 1 / weights_cutout.data
            del weights_cutout
            gc.collect()

            # Calculate total variance -> electrons / s
            variance = weights_variance + poisson_variance
            del poisson_variance
            del weights_variance
            gc.collect()

            # Calculate sigma from variance
            sigma = np.sqrt(variance)

            # Correct for NaNs in first row and column
            corrected_sigma = np.copy(sigma)
            corrected_sigma[0, :] = corrected_sigma[1, :]
            corrected_sigma[:, 0] = corrected_sigma[:, 1]
            pmax = np.nanpercentile(sigma, 99)
            corrected_sigma = np.nan_to_num(corrected_sigma, nan=pmax, posinf=pmax)

            # Write sigma to file
            sigma_hdul = fits.PrimaryHDU(
                data=corrected_sigma,
                header=exposure_wcs.to_header(),
            )
            sigma_hdul.writeto(sigma_path, overwrite=True)

            # Clear memory
            del sigma_path
            del object
            del position
            del image_size
            del variance
            del sigma
            del corrected_sigma
            del pmax
            del sigma_hdul
            gc.collect()

        # Catch skipped objects
        except Exception as e:
            if not display_progress:
                logger.error(e)
            skipped.append(object)

    # Display skipped objects
    if len(skipped) > 0:
        logger.debug(
            f"Skipped generating sigma maps for {len(skipped)} objects: {skipped}."
        )


def generate_psfs(
    input_root: Path,
    product_root: Path,
    field: str,
    image_version: str,
    catalog_version: str,
    filter: str,
    objects: list[int],
    image_sizes: list[int],
    pixscale: float = 0.04,
    regenerate: bool = False,
    display_progress: bool = False,
):
    """Generate PSF crops for all frames in a filter.

    Parameters
    ----------
    input_root : Path
        Path to root GalWrap input directory.
    product_root : Path
        Path to root GalWrap products directory.
    field : str
        Field of observation.
    image_version : str
        Version of image processing used on observation.
    catalog_version : str
        Version of cataloguing used for field.
    filter : str
        Filter used for observation.
    objects : list[str]
        List of object IDs in catalog for which to generate masks.
    image_sizes : list[int]
        List of image sizes corresponding to each object's stamp.
    pixscale : float, optional
        Pixel scale of science frame, by default 0.04''/px.
    regenerate : bool, optional
        Regenerate existing crops, by default False.
    display_progress : bool, optional
        Display progress on terminal screen via tqdm, by default False.
    """
    logger.info("Generating PSF crops.")

    # Load in PSF and clear memory
    original_psf_path = paths.get_path(
        "original_psf",
        input_root=input_root,
        filter=filter,
    )
    original_psf_file = fits.open(original_psf_path)
    original_psf_data = original_psf_file["PRIMARY"].data
    original_psf_headers = original_psf_file["PRIMARY"].header
    psf_pixscale = original_psf_headers["PIXELSCL"]
    psf_length = original_psf_headers["NAXIS1"]
    original_psf_file.close
    del original_psf_file
    del original_psf_headers
    gc.collect()

    # Iterate over each object
    skipped = []
    for i in (
        tqdm(range(len(objects)), unit="psf", leave=False)
        if display_progress
        else range(len(objects))
    ):
        object, image_size = objects[i], image_sizes[i]

        # Skip existing PSF cutouts
        psf_path = paths.get_path(
            "psf",
            product_root=product_root,
            field=field,
            image_version=image_version,
            catalog_version=catalog_version,
            filter=filter,
            object=object,
        )
        if psf_path.exists() and not regenerate:
            if not display_progress:
                logger.debug(f"Skipping, PSF crop exists.")
            return

        # Calculate PSF size from ratio of PSF pixscale to science pixscale
        psf_image_size = int(image_size * psf_pixscale / pixscale)
        center = int(psf_length / 2)

        try:
            # Cutout square of calculated size, centered at PSF center
            psf = Cutout2D(
                data=original_psf_data, position=(center, center), size=psf_image_size
            )

            # Write to file
            psf_hdul = fits.PrimaryHDU(data=psf.data)
            psf_hdul.writeto(psf_path, overwrite=True)

        except Exception as e:
            if not display_progress:
                logger.error(e)
            skipped.append(object)

    # Display skipped objects
    if len(skipped) > 0:
        logger.debug(
            f"Skipped generating sigma maps for {len(skipped)} objects: {skipped}."
        )


def generate_masks(
    input_root: Path,
    product_root: Path,
    field: str,
    image_version: str,
    catalog_version: str,
    filter: str,
    objects: list[int],
    positions: list[SkyCoord],
    image_sizes: list[int],
    regenerate: bool = False,
    display_progress: bool = False,
):
    """Generate masks for all objects in a FICL.

    Parameters
    ----------
    input_root : Path
        Path to root GalWrap input directory.
    product_root : Path
        Path to root GalWrap products directory.
    field : str
        Field of observation.
    image_version : str
        Version of image processing used on observation.
    catalog_version : str
        Version of cataloguing used for field.
    filter : str
        Filter used for observation.
    objects : list[str]
        List of object IDs in catalog for which to generate masks.
    positions : list[SkyCoord]
        List of positions of objects in the sky, from catalog.
    image_sizes : list[int]
        List of image sizes corresponding to each object's stamp.
    regenerate : bool, optional
        Regenerate existing masks, by default False.
    display_progress : bool, optional
        Display progress on terminal screen via tqdm, by default False.
    """
    logger.info("Generating masks.")

    # Load in segmentation map
    segmap_path = paths.get_path(
        "segmap",
        input_root=input_root,
        field=field,
        image_version=image_version,
        filter=filter,
    )
    segmap_file = fits.open(segmap_path)
    segmap_data = segmap_file["PRIMARY"].data
    segmap_wcs = WCS(segmap_file["PRIMARY"].header)

    # Close and clear files from memory
    segmap_file.close()
    del segmap_path
    del segmap_file
    gc.collect()

    # Iterate over each object, position, and image_size tuple
    skipped = []
    for i in (
        tqdm(range(len(objects)), unit="mask", leave=False)
        if display_progress
        else range(len(objects))
    ):
        object, position, image_size = objects[i], positions[i], image_sizes[i]

        # Skip objects which already have masks
        mask_path = paths.get_path(
            "mask",
            product_root=product_root,
            field=field,
            image_version=image_version,
            catalog_version=catalog_version,
            filter=filter,
            object=object,
        )
        if mask_path.exists() and not regenerate:
            if not display_progress:
                logger.debug(f"Skipping object {object}, mask exists.")
            skipped.append(object)
            continue

        # Try generating mask for object
        try:
            if not display_progress:
                logger.info(f"Generating mask for object {object}.")

            # Create cutout from segmap
            segmap_cutout = Cutout2D(
                data=segmap_data,
                position=position,
                size=image_size,
                wcs=segmap_wcs,
            )

            # Set array of ones to zero where object and sky are
            mask = np.ones(shape=(image_size, image_size))
            object_location = np.where(segmap_cutout.data == object + 1)
            sky_location = np.where(segmap_cutout.data == 0)
            mask[object_location] = 0
            mask[sky_location] = 0

            # Write to disk
            mask_hdul = fits.PrimaryHDU(data=mask, header=segmap_cutout.wcs.to_header())
            mask_hdul.writeto(mask_path, overwrite=True)

            # Clear memory
            del object
            del position
            del image_size
            del mask_path
            del segmap_cutout
            del mask
            del object_location
            del sky_location
            del mask_hdul
            gc.collect()

        # Catch skipped objects
        except Exception as e:
            if not display_progress:
                logger.error(e)
            skipped.append(object)

    # Display skipped objects
    if len(skipped) > 0:
        logger.debug(f"Skipped generating masks for {len(skipped)} objects: {skipped}.")


## Main


def generate_products(
    morphfits_config: MorphFITSConfig,
    regenerate_products: bool = False,
    regenerate_stamp: bool = False,
    regenerate_sigma: bool = False,
    regenerate_psf: bool = False,
    regenerate_mask: bool = False,
    regenerate_feedfile: bool = True,
    minimum_image_size: int = 32,
    kron_factor: int = 3,
    display_progress: bool = False,
):
    """Generate all products for a given configuration.

    Parameters
    ----------
    morphfits_config : MorphFITSConfig
        Configuration object for this MorphFITS run.
    regenerate_products : bool, optional
        Regenerate all products, by default False.
    regenerate_stamp : bool, optional
        Regenerate stamps, by default False.
    regenerate_sigma : bool, optional
        Regenerate sigma maps, by default False.
    regenerate_psf : bool, optional
        Regenerate PSF crops, by default False.
    regenerate_mask : bool, optional
        Regenerate masks, by default False.
    regenerate_feedfile : bool, optional
        Regenerate feedfiles, by default True.
    apply_sigma : bool, optional
        Use sigma maps in the GALFIT run, by default True.
    apply_psf : bool, optional
        Use PSFs in the GALFIT run, by default True.
    apply_mask : bool, optional
        Use masks in the GALFIT run, by default True.
    minimum_image_size : int, optional
        Minimum pixel length of square stamp, by default 32.
    kron_factor : int, optional
        Multiplicative factor to apply to Kron radius for each object to
        determine image size of stamp, by default 3.
    display_progress : bool, optional
        Display progress on terminal screen via tqdm, by default False.
    """
    # Iterate over each FICL in configuration
    for ficl in morphfits_config.get_FICLs():
        logger.info(f"Generating products for FICL {ficl}.")

        # Generate science cutouts if missing or requested
        objects, positions, image_sizes = generate_stamps(
            input_root=morphfits_config.input_root,
            product_root=morphfits_config.product_root,
            field=ficl.field,
            image_version=ficl.image_version,
            catalog_version=ficl.catalog_version,
            filter=ficl.filter,
            objects=ficl.objects,
            minimum_image_size=minimum_image_size,
            kron_factor=kron_factor,
            regenerate=regenerate_products
            or regenerate_stamp
            or regenerate_sigma
            or regenerate_psf
            or regenerate_mask
            or regenerate_feedfile,
            display_progress=display_progress,
        )

        # Generate sigma maps if missing or requested
        generate_sigmas(
            input_root=morphfits_config.input_root,
            product_root=morphfits_config.product_root,
            field=ficl.field,
            image_version=ficl.image_version,
            catalog_version=ficl.catalog_version,
            filter=ficl.filter,
            objects=objects,
            positions=positions,
            image_sizes=image_sizes,
            regenerate=regenerate_products or regenerate_sigma,
            display_progress=display_progress,
        )

        # Generate PSF if missing or requested
        generate_psfs(
            input_root=morphfits_config.input_root,
            product_root=morphfits_config.product_root,
            field=ficl.field,
            image_version=ficl.image_version,
            catalog_version=ficl.catalog_version,
            filter=ficl.filter,
            objects=objects,
            image_sizes=image_sizes,
            regenerate=regenerate_products or regenerate_psf,
            display_progress=display_progress,
        )

        # Generate masks if missing or requested
        generate_masks(
            input_root=morphfits_config.input_root,
            product_root=morphfits_config.product_root,
            field=ficl.field,
            image_version=ficl.image_version,
            catalog_version=ficl.catalog_version,
            filter=ficl.filter,
            objects=objects,
            positions=positions,
            image_sizes=image_sizes,
            regenerate=regenerate_products or regenerate_mask,
            display_progress=display_progress,
        )

        # Generate wrapper-specific files
        for wrapper in morphfits_config.wrappers:
            match wrapper:
                # GALFIT requires a feedfile to run
                case "galfit":
                    galfit.generate_feedfiles(
                        input_root=morphfits_config.input_root,
                        product_root=morphfits_config.product_root,
                        output_root=morphfits_config.output_root,
                        field=ficl.field,
                        image_version=ficl.image_version,
                        catalog_version=ficl.catalog_version,
                        filter=ficl.filter,
                        objects=objects,
                        image_sizes=image_sizes,
                        pixscale=ficl.pixscale,
                        regenerate=regenerate_products or regenerate_feedfile,
                        display_progress=display_progress,
                    )
                # Other wrappers unrecognized
                case _:
                    raise NotImplementedError(f"Wrapper for {wrapper} not implemented.")
