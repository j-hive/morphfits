"""Create MorphFITS products.

A product is distinct from an output in the sense that it is, in the MorphFITS
pipeline, an intermediate stage. The following products are generated by this
module.
1. Stamp (square cutout of an object in its field)
2. Sigma Map (deviation corresponding to stamp)
3. PSF Crop (square cutout of the corresponding filter's simulated PSF)
4. Mask (binary mask identifying the object and sky)
"""

# Imports


import logging
from pathlib import Path

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata.utils import Cutout2D
from astropy.table import Table
from astropy.wcs import WCS
from scipy import ndimage

from tqdm import tqdm

from . import settings
from .settings import RuntimeSettings, FICL
from .utils import science


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
    path: Path,
    image: np.ndarray,
    wcs: WCS,
    zeropoint: float,
    pixscale: tuple[int, int],
    position: SkyCoord,
    image_size: int,
):
    """Create the stamp for a single object.

    A stamp is a 2D square cutout of an object (galaxy or cluster) from its
    original observation image.

    When successful, the stamp is written to disk.

    Parameters
    ----------
    path : Path
        Path to which to write stamp FITS file.
    image : ndarray
        2D float image data array from input science file.
    wcs : WCS
        Coordinate system object from input science file.
    zeropoint : float
        Brightness zeropoint from input science file.
    pixscale : tuple[int, int]
        Pixel scales along each axis from input science file, in "/px.
    position : SkyCoord
        Coordinates of object in sky.
    image_size : int
        Number of pixels along one dimension of square stamp.

    Raises
    ------
    ValueError
        Cutout missing nonzero data or non-square dimensions (likely on the edge
        of the original image).
    """
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
            image=stamp.data, pixscale=pixscale, zeropoint=zeropoint
        )

        # Wrote stamp to FITS file
        stamp_hdul = fits.PrimaryHDU(data=stamp.data, header=stamp_headers)
        stamp_hdul.writeto(path, overwrite=True)

    # Otherwise raise error to skip object
    elif not stamp_has_nonzero_data:
        raise ValueError(f"missing nonzero data")
    else:
        raise ValueError(
            f"got dimensions {stamp.data.shape}, "
            + f"expected ({image_size}, {image_size})"
        )


def make_sigma(
    path: Path,
    stamp_image: np.ndarray,
    exposure_image: np.ndarray,
    exposure_headers: fits.Header,
    exposure_wcs: WCS,
    weights_image: np.ndarray,
    weights_wcs: WCS,
    position: SkyCoord,
    image_size: int,
):
    """Create the sigma map for a single object.

    A sigma map is the square root of the variance in the data of the
    corresponding stamp.

    When successful, the sigma is written to disk.

    Parameters
    ----------
    path : Path
        Path to which to write sigma map FITS file.
    stamp_image : np.ndarray
        2D float image data array from product stamp file.
    exposure_image : np.ndarray
        2D float image data array from input exposure map file.
    exposure_headers : fits.Header
        FITS headers from input exposure map file.
    exposure_wcs : WCS
        Coordinate system object from input exposure map file.
    weights_image : np.ndarray
        2D float image data array from input weights map file.
    weights_wcs : WCS
        Coordinate system object from input weights map file.
    position : SkyCoord
        Coordinates of object in sky.
    image_size : int
        Number of pixels along one dimension of square image.
    """
    # Filter stamp over minimum of 0
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
    sigma_hdul.writeto(path, overwrite=True)


def make_psf(path: Path, image: np.ndarray, position: tuple[int, int], image_size: int):
    """Create the PSF crop for a single object.

    A PSF crop is a cutout of the simulated PSF for the corresponding filter, at
    a scale appropriate to the stamp and pixel scale of the original
    observation.

    When successful, the PSF is written to disk.

    Parameters
    ----------
    path : Path
        Path to which to write PSF crop FITS file.
    image : np.ndarray
        2D float image data array from input PSF file.
    position : SkyCoord
        Coordinates of object in sky.
    image_size : int
        Number of pixels along one dimension of square stamp.
    """
    # Cutout square of calculated size, centered at PSF center
    psf = Cutout2D(data=image, position=position, size=image_size)

    # Write to file
    psf_hdul = fits.PrimaryHDU(data=psf.data)
    psf_hdul.writeto(path, overwrite=True)


def make_mask(
    path: Path,
    image: np.ndarray,
    wcs: WCS,
    expand: bool,
    object: int,
    position: SkyCoord,
    image_size: int,
):
    """Create the mask for a single object.

    A mask is a binary switch image to determine if the same pixel in the
    corresponding stamp should be considered 'on'. A pixel is 'on' if it is
    identified as part of the object, or background. A pixel is 'off' if it is
    identified as part of another object. The mask prevents morphology fitting
    to irrelevant pixels.

    When successful, the mask is written to disk.

    Parameters
    ----------
    path : Path
        Path to which to write mask FITS file.
    image : ndarray
        2D float image data array from input segmentation map file.
    wcs : WCS
        Coordinate system object from input segmentation map file.
    expand : bool
        Expand the segmentation map to twice its size. Used for shorter
        wavelength observations, due to them having half the pixel scale of
        longer wavelength observations.
    object : int
        Integer ID of object in catalog.
    position : SkyCoord
        Coordinates of object in sky.
    image_size : int
        Number of pixels along one dimension of square image.

    Raises
    ------
    FileExistsError
        Mask already exists and remaking not requested.
    """
    # Create cutout from segmap
    segmap_cutout = Cutout2D(
        data=image,
        position=position,
        size=image_size,
        wcs=wcs,
    )

    # Set array of ones to zero where object and sky are
    mask = np.ones(shape=(image_size, image_size))
    object_location = np.where(segmap_cutout.data == object + 1)
    sky_location = np.where(segmap_cutout.data == 0)
    mask[object_location] = 0
    mask[sky_location] = 0

    # Expand (re-bin) data if incorrect pixscale
    if expand:
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
    mask_hdul.writeto(path, overwrite=True)


## All Objects


def make_ficl_stamps(
    runtime_settings: RuntimeSettings, ficl: FICL, input_catalog: Table
):
    """Create the stamps for every object in a FICL.

    Skips the FICL if any error occurs during opening the corresponding
    observation. Skips an object if any error occurs during stamp creation.

    When successful, all created stamps are written to disk.

    Parameters
    ----------
    runtime_settings : RuntimeSettings
        Settings for the runtime of this program execution.
    ficl : FICL
        Settings for a single observation, e.g. field and image version.
    input_catalog : Table
        Catalog detailing each object in a field.
    """
    logger.info(f"FICL {ficl}: Making stamps.")

    # Try loading image and header data
    try:
        science_path = settings.get_path(
            name="science", path_settings=runtime_settings.roots, ficl=ficl
        )
        image, headers = science.get_fits_data(path=science_path)
        # zeropoint = science.get_zeropoint(headers=headers)
        zeropoint = 28.9
        wcs = WCS(header=headers)
    except Exception as e:
        logger.error(f"FICL {ficl}: Skipping stamps - failed loading input.")
        logger.error(e)
        return

    # Get iterable object list, displaying progress bar if flagged
    if runtime_settings.progress_bar:
        objects = tqdm(iterable=ficl.objects, unit="obj", leave=False)
    else:
        objects = ficl.objects

    # Iterate over each object
    skipped = 0
    for object in objects:
        # Try making stamp for object
        try:
            # Get path to stamp
            stamp_path = settings.get_path(
                name="stamp",
                path_settings=runtime_settings.roots,
                ficl=ficl,
                object=object,
            )

            # Skip existing stamps unless requested
            if stamp_path.exists() and not runtime_settings.remake.stamps:
                if not runtime_settings.progress_bar:
                    logger.debug(f"Object {object}: Skipping stamp - exists.")
                skipped += 1
                continue

            # Get object position and image size from catalog
            position = science.get_position(input_catalog=input_catalog, object=object)
            image_size = science.get_image_size(
                input_catalog=input_catalog,
                catalog_version=ficl.catalog_version,
                object=object,
                pixscale=ficl.pixscale,
            )

            # Make stamp for object
            if not runtime_settings.progress_bar:
                logger.debug(f"Object {object}: Making stamp.")
            make_stamp(
                path=stamp_path,
                image=image,
                wcs=wcs,
                zeropoint=zeropoint,
                pixscale=ficl.pixscale,
                position=position,
                image_size=image_size,
            )

        # Catch any errors and skip to next object
        except Exception as e:
            if not runtime_settings.progress_bar:
                logger.debug(f"Object {object}: Skipping stamp - {e}.")
            skipped += 1
            continue

    # Log number of skipped or failed objects
    logger.info(f"FICL {ficl}: Made stamps - skipped {skipped} objects.")


def make_ficl_sigmas(
    runtime_settings: RuntimeSettings, ficl: FICL, input_catalog: Table
):
    """Create the sigma maps for every object in a FICL.

    Skips the FICL if any error occurs during opening the corresponding exposure
    or weights maps. Skips an object if any error occurs during sigma creation.

    When successful, all created sigma maps are written to disk.

    Parameters
    ----------
    runtime_settings : RuntimeSettings
        Settings for the runtime of this program execution.
    ficl : FICL
        Settings for a single observation, e.g. field and image version.
    input_catalog : Table
        Catalog detailing each object in a field.
    """
    logger.info(f"FICL {ficl}: Making sigma maps.")

    # Try loading image and header data
    try:
        exposure_path = settings.get_path(
            name="exposure", path_settings=runtime_settings.roots, ficl=ficl
        )
        exposure_image, exposure_headers = science.get_fits_data(exposure_path)
        exposure_wcs = WCS(header=exposure_headers)
        weights_path = settings.get_path(
            name="weights", path_settings=runtime_settings.roots, ficl=ficl
        )
        weights_image, weights_headers = science.get_fits_data(weights_path)
        weights_wcs = WCS(header=weights_headers)
    except Exception as e:
        logger.error(f"FICL {ficl}: Skipping sigma maps - failed loading input.")
        logger.error(e)
        return

    # Get iterable object list, displaying progress bar if flagged
    if runtime_settings.progress_bar:
        objects = tqdm(iterable=ficl.objects, unit="obj", leave=False)
    else:
        objects = ficl.objects

    # Iterate over each object
    skipped = 0
    for object in objects:
        # Try making sigma map for object
        try:
            # Get path to sigma map
            sigma_path = settings.get_path(
                name="sigma",
                path_settings=runtime_settings.roots,
                ficl=ficl,
                object=object,
            )

            # Skip existing sigmas unless requested
            if sigma_path.exists() and not runtime_settings.remake.sigmas:
                if not runtime_settings.progress_bar:
                    logger.debug(f"Object {object}: Skipping sigma map - exists.")
                skipped += 1
                continue

            # Get path to corresponding stamp
            stamp_path = settings.get_path(
                name="stamp",
                path_settings=runtime_settings.roots,
                ficl=ficl,
                object=object,
            )

            # Skip FICLOs without stamps
            if not stamp_path.exists():
                if not runtime_settings.progress_bar:
                    logger.debug(
                        f"Object {object}: Skipping sigma map - missing stamp."
                    )
                skipped += 1
                continue

            # Get stamp image
            stamp_image, stamp_headers = science.get_fits_data(stamp_path)

            # Get object position and image size from catalog
            position = science.get_position(input_catalog=input_catalog, object=object)
            image_size = science.get_image_size(
                input_catalog=input_catalog,
                catalog_version=ficl.catalog_version,
                object=object,
                pixscale=ficl.pixscale,
            )

            # Make sigma map for object
            if not runtime_settings.progress_bar:
                logger.debug(f"Object {object}: Making sigma map.")
            make_sigma(
                path=sigma_path,
                stamp_image=stamp_image,
                exposure_image=exposure_image,
                exposure_headers=exposure_headers,
                exposure_wcs=exposure_wcs,
                weights_image=weights_image,
                weights_wcs=weights_wcs,
                position=position,
                image_size=image_size,
            )

        # Catch any errors and skip to next object
        except Exception as e:
            if not runtime_settings.progress_bar:
                logger.debug(f"Object {object}: Skipping sigma map - {e}.")
            skipped += 1
            continue

    # Log number of skipped or failed objects
    logger.info(f"FICL {ficl}: Made sigma maps - skipped {skipped} objects.")


def make_ficl_psfs(runtime_settings: RuntimeSettings, ficl: FICL, input_catalog: Table):
    """Create the PSF crops for every object in a FICL.

    Skips the FICL if any error occurs during opening the corresponding
    simulated PSF. Skips an object if any error occurs during PSF crop
    creation.

    When successful, all created PSf crops are written to disk.

    Parameters
    ----------
    runtime_settings : RuntimeSettings
        Settings for the runtime of this program execution.
    ficl : FICL
        Settings for a single observation, e.g. field and image version.
    input_catalog : Table
        Catalog detailing each object in a field.
    """
    logger.info(f"FICL {ficl}: Making PSF crops.")

    # Try opening PSF and getting information from headers
    try:
        input_psf_path = settings.get_path(
            name="input_psf", path_settings=runtime_settings.roots, ficl=ficl
        )
        input_psf_image, input_psf_headers = science.get_fits_data(input_psf_path)
        input_psf_pixscale = input_psf_headers["PIXELSCL"]
        input_psf_length = input_psf_headers["NAXIS1"]
    except Exception as e:
        logger.error(f"FICL {ficl}: Skipping PSF crops - failed loading input.")
        logger.error(e)
        return

    # Get iterable object list, displaying progress bar if flagged
    if runtime_settings.progress_bar:
        objects = tqdm(iterable=ficl.objects, unit="obj", leave=False)
    else:
        objects = ficl.objects

    # Iterate over each object
    skipped = 0
    for object in objects:
        # Try making PSF crop for object
        try:
            # Get path to PSF crop
            psf_path = settings.get_path(
                name="psf",
                path_settings=runtime_settings.roots,
                ficl=ficl,
                object=object,
            )

            # Skip existing PSF crops unless requested
            if psf_path.exists() and not runtime_settings.remake.psfs:
                if not runtime_settings.progress_bar:
                    logger.debug(f"Object {object}: Skipping PSF crop - exists.")
                skipped += 1
                continue

            # Get object image size from catalog
            image_size = science.get_image_size(
                input_catalog=input_catalog,
                catalog_version=ficl.catalog_version,
                object=object,
                pixscale=ficl.pixscale,
            )

            # Calculate PSF size from ratio of PSF pixscale to science pixscale
            psf_image_size = int(image_size * input_psf_pixscale / max(ficl.pixscale))
            center = int(input_psf_length / 2)

            # Make PSF crop for object
            if not runtime_settings.progress_bar:
                logger.debug(f"Object {object}: Making PSF crop.")
            make_psf(
                path=psf_path,
                image=input_psf_image,
                position=(center, center),
                image_size=psf_image_size,
            )

        # Catch any errors and skip to next object
        except Exception as e:
            if not runtime_settings.progress_bar:
                logger.debug(f"Object {object}: Skipping PSF crop - {e}.")
            skipped += 1
            continue

    # Log number of skipped or failed objects
    logger.info(f"FICL {ficl}: Made PSF crops - skipped {skipped} objects.")


def make_ficl_masks(
    runtime_settings: RuntimeSettings, ficl: FICL, input_catalog: Table
):
    """Create the masks for every object in a FICL.

    Skips the FICL if any error occurs during opening the corresponding
    segmentation map. Skips an object if any error occurs during mask creation.

    When successful, all created masks are written to disk.

    Parameters
    ----------
    runtime_settings : RuntimeSettings
        Settings for the runtime of this program execution.
    ficl : FICL
        Settings for a single observation, e.g. field and image version.
    input_catalog : Table
        Catalog detailing each object in a field.
    """
    logger.info(f"FICL {ficl}: Making masks.")

    # Try opening segmentation map and getting information from headers
    try:
        segmap_path = settings.get_path(
            name="input_segmap", path_settings=runtime_settings.roots, ficl=ficl
        )
        segmap_image, segmap_headers = science.get_fits_data(segmap_path)
        segmap_wcs = WCS(segmap_headers)
        segmap_pixscale = science.get_pixscale(segmap_path)

        # Set flag to expand data if pixscales are not equal
        # As a result of short wavelength observations having half the pixscale
        # of long wavelength observations
        expand_segmap = max(segmap_pixscale) / max(ficl.pixscale) == 2
    except Exception as e:
        logger.error(f"FICL {ficl}: Skipping masks - failed loading input.")
        logger.error(e)
        return

    # Get iterable object list, displaying progress bar if flagged
    if runtime_settings.progress_bar:
        objects = tqdm(iterable=ficl.objects, unit="obj", leave=False)
    else:
        objects = ficl.objects

    # Iterate over each object
    skipped = 0
    for object in objects:
        # Try making mask for object
        try:
            # Get path to mask
            mask_path = settings.get_path(
                name="mask",
                path_settings=runtime_settings.roots,
                ficl=ficl,
                object=object,
            )

            # Skip existing masks unless requested
            if mask_path.exists() and not runtime_settings.remake.masks:
                if not runtime_settings.progress_bar:
                    logger.debug(f"Object {object}: Skipping mask - exists.")
                skipped += 1
                continue

            # Get position and image size of this object
            position = science.get_position(input_catalog=input_catalog, object=object)
            image_size = science.get_image_size(
                input_catalog=input_catalog,
                catalog_version=ficl.catalog_version,
                object=object,
                pixscale=ficl.pixscale,
            )

            # Make mask for object
            if not runtime_settings.progress_bar:
                logger.debug(f"Object {object}: Making mask.")
            make_mask(
                path=mask_path,
                image=segmap_image,
                wcs=segmap_wcs,
                expand=expand_segmap,
                object=object,
                position=position,
                image_size=image_size,
            )

        # Catch any errors and skip to next object
        except Exception as e:
            if not runtime_settings.progress_bar:
                logger.debug(f"Object {object}: Skipping mask - {e}.")
            skipped += 1
            continue

    # Log number of skipped or failed objects
    logger.info(f"FICL {ficl}: Made masks - skipped {skipped} objects.")


## Primary


def make_all(runtime_settings: RuntimeSettings):
    """Create each product for each FICL in this program run.

    A product is an intermediate FITS file isolating an object within its field.
    When successful, all created products are written to disk.

    Creates products in the following order.
    1. Stamp
    2. Sigma Map
    3. PSF Crop
    4. Mask
    Note that for each FICL (observation), all of its corresponding stamps will
    be created and written, then all of its corresponding sigma maps, and so on.

    Parameters
    ----------
    runtime_settings : RuntimeSettings
        Settings for the runtime of this program execution.
    """
    # Iterate over each FICL in this run
    for ficl in runtime_settings.ficls:
        # Try to make all products for FICL
        try:
            logger.info(f"FICL {ficl}: Making products.")
            logger.info(
                f"Objects: {min(ficl.objects)} to {max(ficl.objects)} "
                + f"({len(ficl.objects)} objects)."
            )

            # Open input catalog
            input_catalog_path = settings.get_path(
                name="input_catalog", path_settings=runtime_settings.roots, ficl=ficl
            )
            input_catalog = Table.read(input_catalog_path)

            # Make stamps for all objects in FICL
            make_ficl_stamps(runtime_settings, ficl, input_catalog)

            # Make sigma map stamps for all objects in FICL
            make_ficl_sigmas(runtime_settings, ficl, input_catalog)

            # Make PSF crops for all objects in FICL
            make_ficl_psfs(runtime_settings, ficl, input_catalog)

            # Make mask stamps for all objects in FICL
            make_ficl_masks(runtime_settings, ficl, input_catalog)

        # Catch any error opening FICL or input catalog
        except Exception as e:
            logger.error(f"FICL {ficl}: Skipping making products - {e}.")
            continue
