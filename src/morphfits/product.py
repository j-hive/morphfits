"""Create J-HIVE MorphFITS GalWrap products for GALFIT usage.
"""

# Imports


import logging
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.nddata.utils import Cutout2D
from astropy.table import Table
from astropy.wcs import WCS
from scipy import ndimage

from tqdm import tqdm

from .config import MorphFITSConfig, FICL
from .utils import path, science


# Constants


logger = logging.getLogger("PRODUCT")
"""Logging object for this module.
"""


np.seterr(divide="ignore", invalid="ignore")
"""Ignore zero division warnings in making sigma maps.
"""


# Functions


## Single Object


def make_stamp(
    morphfits_config: MorphFITSConfig,
    ficl: FICL,
    object: int,
    input_catalog: Table,
    image: np.ndarray,
    wcs: WCS,
    zeropoint: float,
    remake: bool = False,
):
    # Get path to which to write stamp
    stamp_path = path.get_path(
        name="stamp", morphfits_config=morphfits_config, ficl=ficl, object=object
    )

    # Skip existing stamps unless in remake mode
    if stamp_path.exists() and not remake:
        raise FileExistsError(f"Skipping object {object}, stamp exists.")

    # Get object position and image size from catalog
    position = science.get_position(input_catalog=input_catalog, object=object)
    image_size = science.get_image_size(
        input_catalog=input_catalog,
        catalog_version=ficl.catalog_version,
        object=object,
        pixscale=ficl.pixscale,
    )

    # Generate stamp
    stamp = Cutout2D(data=image, position=position, size=image_size, wcs=wcs)

    # Get nonzero and shape correctness of stamp
    stamp_has_nonzero_data = np.amax(stamp.data) > 0
    stamp_is_correct_shape = stamp.data.shape == (image_size, image_size)

    # Write stamp to disk if image nonzero and of correct shape
    if stamp_has_nonzero_data and stamp_is_correct_shape:
        stamp_headers = stamp.wcs.to_header()
        stamp_headers["EXPTIME"] = 1
        stamp_headers["ZP"] = zeropoint
        stamp_headers["SB"] = science.get_surface_brightness(
            image=stamp.data, pixscale=ficl.pixscale, zeropoint=zeropoint
        )

        # Wrote stamp to FITS file
        stamp_hdul = fits.PrimaryHDU(data=stamp.data, header=stamp_headers)
        stamp_hdul.writeto(stamp_path, overwrite=True)

    # Otherwise raise error to skip object
    else:
        if not stamp_has_nonzero_data:
            raise ValueError(f"Skipping object {object}, missing nonzero data.")
        else:
            raise ValueError(
                f"Skipping object {object}, dimensions "
                + f"{stamp.data.shape} don't match expected image size {image_size}."
            )


def make_sigma(
    morphfits_config: MorphFITSConfig,
    ficl: FICL,
    object: int,
    input_catalog: Table,
    exposure_image: np.ndarray,
    exposure_headers: fits.Header,
    exposure_wcs: WCS,
    weights_image: np.ndarray,
    weights_wcs: WCS,
    remake: bool = False,
):
    # Get path to which to write sigma map
    sigma_path = path.get_path(
        name="sigma", morphfits_config=morphfits_config, ficl=ficl, object=object
    )

    # Skip existing sigmas unless in remake mode
    if sigma_path.exists() and not remake:
        raise FileExistsError(f"Skipping object {object}, sigma map exists.")

    # Get position and image size of this object
    position = science.get_position(input_catalog=input_catalog, object=object)
    image_size = science.get_image_size(
        input_catalog=input_catalog,
        catalog_version=ficl.catalog_version,
        object=object,
        pixscale=ficl.pixscale,
    )

    # Load stamp for this object and filter over minimum of 0
    stamp_image, stamp_headers = science.get_fits_data(
        name="stamp", morphfits_config=morphfits_config, ficl=ficl, object=object
    )
    maximized_stamp = np.maximum(stamp_image, 0)

    # Generate cutout for exposure map, which is 4x smaller than science
    quarter_image_size = int(image_size / 4)
    exposure_image_size = (
        quarter_image_size + 1
        if int((image_size + 1) / 4) > quarter_image_size
        else quarter_image_size
    )
    exposure_cutout = Cutout2D(
        data=exposure_image,
        position=position,
        size=exposure_image_size,
        wcs=exposure_wcs,
    )

    # Grow exposure cutout to same size as science -> s
    zeroes = np.zeros(shape=(image_size, image_size), dtype=int)
    zeroes[2::4, 2::4] += exposure_cutout.data
    full_exposure = ndimage.maximum_filter(input=zeroes, size=4)

    # Find multiplicative factors applied to original count rate data
    scale_factor = 1.0
    for header in ["PHOTMJSR", "PHOTSCAL", "OPHOTFNU"]:
        if header in exposure_headers:
            scale_factor /= exposure_headers[header]
        if header == "OPHOTFNU":
            scale_factor *= exposure_headers["PHOTFNU"]

    # Calculate effective gain - electrons per DN of the mosaic
    effective_gain = scale_factor * full_exposure

    # Calculate Poisson variance in mosaic DN -> 10nJy / s
    poisson_variance = maximized_stamp / effective_gain

    # Generate cutout for weights
    weights_cutout = Cutout2D(
        data=weights_image,
        position=position,
        size=image_size,
        wcs=weights_wcs,
    )

    # Calculate original variance from weights map -> 1 / 10nJy
    weights_variance = 1 / weights_cutout.data

    # Calculate total variance -> electrons / s
    variance = weights_variance + poisson_variance

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


def make_psf(
    morphfits_config: MorphFITSConfig,
    ficl: FICL,
    object: int,
    input_catalog: Table,
    input_psf_image: np.ndarray,
    input_psf_pixscale: float,
    input_psf_length: float,
    remake: bool = False,
):
    # Get path to which to write PSF crop
    psf_path = path.get_path(
        name="psf", morphfits_config=morphfits_config, ficl=ficl, object=object
    )

    # Skip existing PSF crops unless in remake mode
    if psf_path.exists() and not remake:
        raise FileExistsError(f"Skipping object {object}, PSF crop exists.")

    # Get image size of this object
    image_size = science.get_image_size(
        input_catalog=input_catalog,
        catalog_version=ficl.catalog_version,
        object=object,
        pixscale=ficl.pixscale,
    )

    # Calculate PSF size from ratio of PSF pixscale to science pixscale
    psf_image_size = int(image_size * input_psf_pixscale / max(ficl.pixscale))
    center = int(input_psf_length / 2)

    # Cutout square of calculated size, centered at PSF center
    psf = Cutout2D(data=input_psf_image, position=(center, center), size=psf_image_size)

    # Write to file
    psf_hdul = fits.PrimaryHDU(data=psf.data)
    psf_hdul.writeto(psf_path, overwrite=True)


def make_mask(
    morphfits_config: MorphFITSConfig,
    ficl: FICL,
    object: int,
    input_catalog: Table,
    segmap_image: np.ndarray,
    segmap_wcs: WCS,
    expand_segmap: bool,
    remake: bool = False,
):
    # Get path to which to write mask
    mask_path = path.get_path(
        name="mask",
        morphfits_config=morphfits_config,
        ficl=ficl,
        object=object,
    )
    if mask_path.exists() and not remake:
        raise FileExistsError(f"Skipping object {object}, mask exists.")

    # Get position and image size of this object
    position = science.get_position(input_catalog=input_catalog, object=object)
    image_size = science.get_image_size(
        input_catalog=input_catalog,
        catalog_version=ficl.catalog_version,
        object=object,
        pixscale=ficl.pixscale,
    )

    # Create cutout from segmap
    segmap_cutout = Cutout2D(
        data=segmap_image,
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

    # Expand (re-bin) data if incorrect pixscale
    if expand_segmap:
        zeroes = np.zeros(shape=(image_size, image_size))
        i_start = int(image_size / 4)
        mask_length = int(np.ceil(image_size / 2))
        mask_cutout = mask[
            i_start : i_start + mask_length, i_start : i_start + mask_length
        ]
        zeroes[::2, ::2] += mask_cutout
        mask = ndimage.maximum_filter(input=zeroes, size=2)

    # Write to disk
    mask_hdul = fits.PrimaryHDU(data=mask, header=segmap_cutout.wcs.to_header())
    mask_hdul.writeto(mask_path, overwrite=True)


## All Objects


def make_stamps(
    morphfits_config: MorphFITSConfig,
    ficl: FICL,
    input_catalog: Table,
    remake: bool = False,
    progress_bar: bool = False,
):
    logger.info(f"Making stamps for FICL {ficl}.")

    # Try loading image and header data
    try:
        image, headers = science.get_fits_data(
            name="science", morphfits_config=morphfits_config, ficl=ficl
        )
        zeropoint = science.get_zeropoint(headers=headers)
        wcs = WCS(header=headers)
    except Exception as e:
        logger.error(f"Skipping making stamps for FICL {ficl}.")
        logger.error(e)
        return

    # Get iterable object list, displaying progress bar if flagged
    if progress_bar:
        objects = tqdm(iterable=ficl.objects, unit="obj", leave=False)
    else:
        objects = ficl.objects

    # Iterate over each object
    skipped = 0
    for object in objects:
        # Try making stamp for object
        if not progress_bar:
            logger.debug(f"Making stamp for object {object}.")
        try:
            make_stamp(
                morphfits_config=morphfits_config,
                ficl=ficl,
                object=object,
                input_catalog=input_catalog,
                image=image,
                wcs=wcs,
                zeropoint=zeropoint,
                remake=remake,
            )

        # Catch any errors and skip to next object
        except Exception as e:
            if not progress_bar:
                logger.debug(e)
            skipped += 1
            continue

    # Log number of skipped or failed objects
    if skipped > 0:
        logger.debug(f"Skipped making stamps for {skipped} objects.")


def make_sigmas(
    morphfits_config: MorphFITSConfig,
    ficl: FICL,
    input_catalog: Table,
    remake: bool = False,
    progress_bar: bool = False,
):
    logger.info(f"Making sigma maps for FICL {ficl}.")

    # Try loading image and header data
    try:
        exposure_image, exposure_headers = science.get_fits_data(
            name="exposure", morphfits_config=morphfits_config, ficl=ficl
        )
        exposure_wcs = WCS(header=exposure_headers)
        weights_image, weights_headers = science.get_fits_data(
            name="weights", morphfits_config=morphfits_config, ficl=ficl
        )
        weights_wcs = WCS(header=weights_headers)
    except Exception as e:
        logger.error(f"Skipping making sigma maps for FICL {ficl}.")
        logger.error(e)
        return

    # Get iterable object list, displaying progress bar if flagged
    if progress_bar:
        objects = tqdm(iterable=ficl.objects, unit="obj", leave=False)
    else:
        objects = ficl.objects

    # Iterate over each object
    skipped = 0
    for object in objects:
        # Try making sigma map for object
        if not progress_bar:
            logger.debug(f"Making sigma map for object {object}.")
        try:
            make_sigma(
                morphfits_config=morphfits_config,
                ficl=ficl,
                object=object,
                input_catalog=input_catalog,
                exposure_image=exposure_image,
                exposure_headers=exposure_headers,
                exposure_wcs=exposure_wcs,
                weights_image=weights_image,
                weights_wcs=weights_wcs,
                remake=remake,
            )

        # Catch any errors and skip to next object
        except Exception as e:
            if not progress_bar:
                logger.debug(e)
            skipped += 1
            continue

    # Log number of skipped or failed objects
    if skipped > 0:
        logger.debug(f"Skipped making sigma maps for {skipped} objects.")


def make_psfs(
    morphfits_config: MorphFITSConfig,
    ficl: FICL,
    input_catalog: Table,
    remake: bool = False,
    progress_bar: bool = False,
):
    logger.info(f"Making PSF crops for FICL {ficl}.")

    # Try opening PSF and getting information from headers
    try:
        input_psf_image, input_psf_headers = science.get_fits_data(
            name="input_psf", morphfits_config=morphfits_config, ficl=ficl
        )
        input_psf_pixscale = input_psf_headers["PIXELSCL"]
        input_psf_length = input_psf_headers["NAXIS1"]
    except Exception as e:
        logger.error(f"Skipping making PSF crops for FICL {ficl}.")
        logger.error(e)
        return

    # Get iterable object list, displaying progress bar if flagged
    if progress_bar:
        objects = tqdm(iterable=ficl.objects, unit="obj", leave=False)
    else:
        objects = ficl.objects

    # Iterate over each object
    skipped = 0
    for object in objects:
        # Try making PSF crop for object
        if not progress_bar:
            logger.debug(f"Making PSF crop for object {object}.")
        try:
            make_psf(
                morphfits_config=morphfits_config,
                ficl=ficl,
                object=object,
                input_catalog=input_catalog,
                input_psf_image=input_psf_image,
                input_psf_pixscale=input_psf_pixscale,
                input_psf_length=input_psf_length,
                remake=remake,
            )

        # Catch any errors and skip to next object
        except Exception as e:
            if not progress_bar:
                logger.debug(e)
            skipped += 1
            continue

    # Log number of skipped or failed objects
    if skipped > 0:
        logger.debug(f"Skipped making PSF crops for {skipped} objects.")


def make_masks(
    morphfits_config: MorphFITSConfig,
    ficl: FICL,
    input_catalog: Table,
    remake: bool = False,
    progress_bar: bool = False,
):
    logger.info(f"Making masks for FICL {ficl}.")

    # Try opening segmentation map and getting information from headers
    try:
        segmap_image, segmap_headers = science.get_fits_data(
            name="input_segmap",
            morphfits_config=morphfits_config,
            ficl=ficl,
        )
        segmap_wcs = WCS(segmap_headers)
        segmap_path = path.get_path(
            name="input_segmap",
            morphfits_config=morphfits_config,
            ficl=ficl,
        )
        segmap_pixscale = science.get_pixscale(science_path=segmap_path)

        # Set flag to expand data if pixscales are not equal
        expand_segmap = max(segmap_pixscale) / max(ficl.pixscale) == 2
    except Exception as e:
        logger.error(f"Skipping making masks for FICL {ficl}.")
        logger.error(e)
        return

    # Get iterable object list, displaying progress bar if flagged
    if progress_bar:
        objects = tqdm(iterable=ficl.objects, unit="obj", leave=False)
    else:
        objects = ficl.objects

    # Iterate over each object
    skipped = 0
    for object in objects:
        # Try making mask for object
        if not progress_bar:
            logger.debug(f"Making mask for object {object}.")
        try:
            make_mask(
                morphfits_config=morphfits_config,
                ficl=ficl,
                object=object,
                input_catalog=input_catalog,
                segmap_image=segmap_image,
                segmap_wcs=segmap_wcs,
                expand_segmap=expand_segmap,
                remake=remake,
            )

        # Catch any errors and skip to next object
        except Exception as e:
            if not progress_bar:
                logger.debug(e)
            skipped += 1
            continue

    # Log number of skipped or failed objects
    if skipped > 0:
        logger.debug(f"Skipped making PSF crops for {skipped} objects.")


## Main


# create product files
# for each ficl:
#     open catalog
#     open image
#     for each object:
#         create stamp file
#     close image
#     open exposure map
#     open weights map
#     for each object:
#         create sigma map file
#     close exposure map
#     close weights map
#     open psf
#     for each object:
#         create psf crop file
#     close psf
#     open segmap
#     for each object:
#         create mask file
#     close segmap
#     close catalog
def make_all(
    morphfits_config: MorphFITSConfig,
    remake_all: bool = False,
    remake_stamps: bool = False,
    remake_sigmas: bool = False,
    remake_psfs: bool = False,
    remake_masks: bool = False,
    progress_bar: bool = False,
):
    # Iterate over each FICL in this run
    for ficl in morphfits_config.ficls:
        logger.info(f"Making products for FICL {ficl}.")
        logger.info(
            f"Object ID range: {min(ficl.objects)} to {max(ficl.objects)} "
            + f"({len(ficl.objects)} objects)."
        )

        # Open catalog
        input_catalog_path = path.get_path(
            "input_catalog",
            input_root=morphfits_config.input_root,
            field=ficl.field,
            image_version=ficl.image_version,
        )
        input_catalog = Table.read(input_catalog_path)

        # Make stamps for all objects in FICL
        make_stamps(
            morphfits_config=morphfits_config,
            ficl=ficl,
            input_catalog=input_catalog,
            remake=remake_all or remake_stamps,
            progress_bar=progress_bar,
        )

        # Make sigma map stamps for all objects in FICL
        make_sigmas(
            morphfits_config=morphfits_config,
            ficl=ficl,
            input_catalog=input_catalog,
            remake=remake_all or remake_sigmas,
            progress_bar=progress_bar,
        )

        # Make PSF crops for all objects in FICL
        make_psfs(
            morphfits_config=morphfits_config,
            ficl=ficl,
            input_catalog=input_catalog,
            remake=remake_all or remake_psfs,
            progress_bar=progress_bar,
        )

        # Make mask stamps for all objects in FICL
        make_masks(
            morphfits_config=morphfits_config,
            ficl=ficl,
            input_catalog=input_catalog,
            remake=remake_all or remake_masks,
            progress_bar=progress_bar,
        )
