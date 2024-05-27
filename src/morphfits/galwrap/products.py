"""Create J-HIVE MorphFITS GalWrap products for GALFIT usage.
"""

# Imports


import gc
import logging
from pathlib import Path

import numpy as np
from scipy import ndimage
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata.utils import Cutout2D
from astropy.table import Table
from astropy.wcs import WCS
from jinja2 import Template

from . import paths, GALWRAP_DATA_ROOT
from .setup import FICLO, FICL, GalWrapConfig
from ..utils import science


# Constants


logger = logging.getLogger("PRODUCTS")
"""Logger object for this module.
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

    Returns
    -------
    tuple[list[int], list[SkyCoord], list[int]]
        List of object IDs, positions, and image sizes for successful stamps.
    """
    logger.info(
        "Generating stamps for FICL "
        + "_".join(field, image_version, catalog_version, filter)
        + "."
    )

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
    gc.collect()

    # Iterate over each object
    generated, skipped = ([], [], []), []
    for object in objects:
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
            skipped.append(object)
            continue

        # Try generating stamp for object
        try:
            logger.info(f"Generating stamp for object {object}.")

            # Record object position from catalog
            position = SkyCoord(
                ra=catalog[object]["ra"], dec=catalog[object]["dec"], unit="deg"
            )

            # Determine image size
            kron_radius = catalog[object]["kron_radius_circ"]
            image_size = np.nanmax(
                [
                    int(kron_radius / 0.04 * kron_factor),
                    minimum_image_size,
                ]
            )

            # Generate stamp
            stamp = Cutout2D(data=image, position=position, size=image_size, wcs=wcs)

            # Write stamp to disk if image nonzero and of correct shape
            if (np.amax(stamp.data) > 0) and (
                stamp.data.shape == (image_size, image_size)
            ):
                fits.PrimaryHDU(data=stamp.data, header=stamp.wcs.to_header()).writeto(
                    stamp_path, overwrite=True
                )

                # Store object ID, position, and image size for other products
                generated[0].append(object)
                generated[1].append(position)
                generated[2].append(image_size)
            else:
                skipped.append(object)

            # Clear memory
            del stamp_path
            del position
            del kron_radius
            del image_size
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

    See Also
    --------
    This algorithm was taken from `the DJA blog <https://dawn-cph.github.io/dja/blog/2023/07/18/image-data-products/>`_.
    """
    logger.info(
        "Generating sigma maps for FICL "
        + "_".join(field, image_version, catalog_version, filter)
        + "."
    )

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
    for i in range(len(objects)):
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
            skipped.append(object)
            continue

        # Try generating sigma map for object
        try:
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

            # Generate cutout for exposure, which is 4x smaller than science
            exposure_cutout = Cutout2D(
                data=exposure_data,
                position=position,
                size=int(image_size / 4),
                wcs=exposure_wcs,
            )

            # Grow exposure cutout to same size as science
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

            # Calculate Poisson variance in mosaic DN
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

            # Calculate original variance from weights map
            weights_variance = 1 / weights_cutout.data
            del weights_cutout
            gc.collect()

            # Calculate total variance
            variance = weights_variance + poisson_variance
            del poisson_variance
            del weights_variance
            gc.collect()

            # Calculate sigma
            sigma = np.sqrt(variance)
            fits.PrimaryHDU(data=sigma, header=weights_wcs.to_header()).writeto(
                sigma_path, overwrite=True
            )

            # Clear memory
            del sigma_path
            del object
            del position
            del image_size
            del sigma
            gc.collect()

        # Catch skipped objects
        except Exception as e:
            logger.error(e)
            skipped.append(object)

    # Display skipped objects
    if len(skipped) > 0:
        logger.debug(
            f"Skipped generating sigma maps for {len(skipped)} objects: {skipped}."
        )


def generate_psf(
    input_root: Path,
    product_root: Path,
    filter: str,
    pixscale: float,
    regenerate: bool = False,
    size_factor: int = 6,
):
    """Generate PSF crops for all frames in a filter.

    Parameters
    ----------
    input_root : Path
        Path to root GalWrap input directory.
    product_root : Path
        Path to root GalWrap products directory.
    filter : str
        Filter used for observation.
    pixscale : float
        Pixel scale of science frame.
    regenerate : bool, optional
        Regenerate existing crops, by default False.
    size_factor : int, optional
        PSF size divisor for image size, by default 6.
    """
    pixname = science.get_pixname(pixscale)
    logger.info("Generating a PSF cutout for LP " + "_".join(filter, pixname) + ".")

    # Skip filters which already have PSF cutouts
    psf_path = paths.get_path(
        "psf",
        product_root=product_root,
        filter=filter,
        pixscale=pixscale,
    )
    if psf_path.exists() and not regenerate:
        logger.info(f"PSF already exists, skipping.")
        return

    # Load in PSF and clear memory
    rawpsf_path = paths.get_path(
        "rawpsf",
        input_root=input_root,
        filter=filter,
    )
    rawpsf_file = fits.open(rawpsf_path)
    rawpsf_data = rawpsf_file["PRIMARY"].data
    rawpsf_headers = rawpsf_file["PRIMARY"].header
    rawpsf_file.close
    del rawpsf_file
    gc.collect()

    # Calculate PSF size from ratio of PSF pixscale to science pixscale
    image_size = int(
        rawpsf_headers["NAXIS1"] * rawpsf_headers["PIXELSCL"] / pixscale / size_factor
    )
    center = int(rawpsf_headers["NAXIS1"] / 2)
    del rawpsf_headers
    gc.collect()

    # Cutout square of calculated size, centered at PSF center
    psf = Cutout2D(data=rawpsf_data, position=(center, center), size=image_size)

    # Write to file
    fits.PrimaryHDU(data=psf.data).writeto(psf_path, overwrite=True)


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
):
    """Generate masks for all objects in a FIC.

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
    """
    logger.info(
        "Generating masks for FIC "
        + "_".join(field, image_version, catalog_version)
        + "."
    )

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
    for i in range(len(objects)):
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
            skipped.append(object)
            continue

        # Try generating mask for object
        try:
            logger.info(f"Generating mask for object {object}.")

            # Generate mask from segmap
            mask = Cutout2D(
                data=segmap_data,
                position=position,
                size=image_size,
                wcs=segmap_wcs,
            )

            # Write to disk
            fits.PrimaryHDU(data=mask.data, header=mask.wcs.to_header()).writeto(
                mask_path, overwrite=True
            )

            # Clear memory
            del mask_path
            del object
            del position
            del image_size
            del mask
            gc.collect()

        # Catch skipped objects
        except Exception as e:
            logger.error(e)
            skipped.append(object)

    # Display skipped objects
    if len(skipped) > 0:
        logger.debug(f"Skipped generating masks for {len(skipped)} objects: {skipped}.")


def generate_feedfiles(
    input_root: Path,
    product_root: Path,
    output_root: Path,
    field: str,
    image_version: str,
    catalog_version: str,
    filter: str,
    objects: list[int],
    image_sizes: list[int],
    pixscale: float,
    regenerate: bool = False,
    apply_sigma: bool = True,
    apply_psf: bool = True,
    apply_mask: bool = True,
    feedfile_template_path: Path = GALWRAP_DATA_ROOT / "feedfile.jinja",
    constraints_path: Path = GALWRAP_DATA_ROOT / "default.constraints",
    path_length: int = 181,
    float_length: int = 12,
):
    """Generate feedfiles for all objects in a FICL.

    Parameters
    ----------
    input_root : Path
        Path to root GalWrap input directory.
    product_root : Path
        Path to root GalWrap products directory.
    output_root : Path
        Path to root Galwrap output directory.
    field : str
        Field of observation.
    image_version : str
        Version of image processing used on observation.
    catalog_version : str
        Version of cataloguing used for field.
    filter : str
        Filter used for observation.
    objects : list[str]
        List of object IDs in catalog for which to generate feedfiles.
    image_sizes : list[int]
        List of image sizes corresponding to each object's stamp.
    pixscale : float
        Pixel scale of science frame.
    regenerate : bool, optional
        Regenerate existing feedfiles, by default False.
    apply_sigma : bool, optional
        Use corresponding sigma map in GALFIT, by default True.
    apply_psf : bool, optional
        Use corresponding PSF in GALFIT, by default True.
    apply_mask : bool, optional
        Use corresponding mask in GALFIT, by default True.
    feedfile_template_path : Path, optional
        Path to jinja2 feedfile template, by default from the repository data
        directory.
    constraints_path : Path, optional
        Path to the GALFIT constraints file, by default from the repository data
        directory.
    path_length : int, optional
        Length of path strings in the template for comment alignment, by default
        181, so that comments start on column 185.
    float_length : int, optional
        Length of float strings in the template for comment alignment, by
        default 12.
    """
    logger.info(
        "Generating feedfiles for FICL "
        + "_".join(field, image_version, catalog_version, filter)
        + "."
    )

    # Define functions for comment alignment
    path_str = lambda x: str(x).ljust(path_length)[:path_length]
    float_str = lambda x: str(x).ljust(float_length)[:path_length]

    # Load in catalog
    catalog_path = paths.get_path(
        "catalog",
        input_root=input_root,
        field=field,
        image_version=image_version,
    )
    catalog = Table.read(catalog_path)

    # Get zeropoint
    science_path = paths.get_path(
        "science",
        input_root=input_root,
        field=field,
        image_version=image_version,
        filter=filter,
    )
    zeropoint = science.get_zeropoint(image_path=science_path)

    # Clear memory
    del catalog_path
    del science_path
    gc.collect()

    # Iterate over each object, and image_size tuple
    skipped = []
    for i in range(len(objects)):
        object, image_size = objects[i], image_sizes[i]

        # Skip objects which already have feedfiles
        feedfile_path = paths.get_path(
            "feedfile",
            product_root=product_root,
            field=field,
            image_version=image_version,
            catalog_version=catalog_version,
            filter=filter,
            object=object,
        )
        if feedfile_path.exists() and not regenerate:
            skipped.append(object)
            continue

        # Generate feedfile for object
        logger.info(f"Generating feedfile for object {object}.")

        # Get paths
        output_galfit_path = paths.get_path(
            "output_galfit",
            output_root=output_root,
            field=field,
            image_version=image_version,
            catalog_version=catalog_version,
            filter=filter,
            object=object,
        )
        stamp_path = paths.get_path(
            "stamp",
            product_root=product_root,
            field=field,
            image_version=image_version,
            catalog_version=catalog_version,
            filter=filter,
            object=object,
        )
        sigma_path = paths.get_path(
            "stamp",
            product_root=product_root,
            field=field,
            image_version=image_version,
            catalog_version=catalog_version,
            filter=filter,
            object=object,
        )
        psf_path = paths.get_path(
            "stamp",
            product_root=product_root,
            filter=filter,
            pixscale=pixscale,
        )
        mask_path = paths.get_path(
            "stamp",
            product_root=product_root,
            field=field,
            image_version=image_version,
            catalog_version=catalog_version,
            filter=filter,
            object=object,
        )

        magnitude = catalog[object]["mag_auto"]
        half_light_radius = catalog[object]["a_image"]
        axis_ratio = catalog[object]["b_image"] / catalog[object]["a_image"]

        # Set configuration parameters from input
        feedfile_dict = {
            "stamp_path": path_str(stamp_path),
            "output_galfit_path": path_str(output_galfit_path),
            "sigma_path": path_str(sigma_path if apply_sigma else ""),
            "psf_path": path_str(psf_path if apply_psf else ""),
            "mask_path": path_str(mask_path if apply_mask else ""),
            "constraints_path": path_str(constraints_path),
            "image_size": float_str(image_size),
            "zeropoint": float_str(zeropoint),
            "position": float_str(image_size / 2),
            "magnitude": float_str(magnitude),
            "half_light_radius": float_str(half_light_radius),
            "axis_ratio": float_str(axis_ratio),
        }

        # Write new feedfile from template and save to output directory
        with open(feedfile_template_path, "r") as feedfile_template:
            template = Template(feedfile_template.read())
        lines = template.render(feedfile_dict)
        with open(feedfile_path, "w") as feedfile:
            feedfile.write(lines)

        # Clear memory
        del object
        del image_size
        del feedfile_path
        del output_galfit_path
        del stamp_path
        del sigma_path
        del psf_path
        del mask_path
        del magnitude
        del half_light_radius
        del axis_ratio
        del feedfile_dict
        del template
        del lines
        gc.collect()


## Main


def generate_product_ficlo(
    galwrap_config: GalWrapConfig,
    ficlo: FICLO,
    regenerate_products: bool = False,
    regenerate_stamp: bool = False,
    regenerate_sigma: bool = False,
    regenerate_psf: bool = False,
    regenerate_mask: bool = False,
    regenerate_feedfile: bool = True,
    apply_sigma: bool = True,
    apply_psf: bool = True,
    apply_mask: bool = True,
    minimum_image_size: int = 32,
):
    raise NotImplementedError()


def generate_products(
    galwrap_config: GalWrapConfig,
    regenerate_products: bool = False,
    regenerate_stamp: bool = False,
    regenerate_sigma: bool = False,
    regenerate_psf: bool = False,
    regenerate_mask: bool = False,
    regenerate_feedfile: bool = True,
    apply_sigma: bool = True,
    apply_psf: bool = True,
    apply_mask: bool = True,
    minimum_image_size: int = 32,
    kron_factor: int = 3,
):
    """Generate all products for a given configuration.

    Parameters
    ----------
    galwrap_config : GalWrapConfig
        Configuration object for this GalWrap run.
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
    """
    # Iterate over each FICL in configuration
    for ficl in galwrap_config.get_FICLs():
        # Generate science cutouts if missing or requested
        objects, positions, image_sizes = generate_stamps(
            input_root=galwrap_config.input_root,
            product_root=galwrap_config.product_root,
            field=ficl.field,
            image_version=ficl.image_version,
            catalog_version=ficl.catalog_version,
            filter=ficl.filter,
            objects=ficl.objects,
            minimum_image_size=minimum_image_size,
            kron_factor=kron_factor,
            regenerate=regenerate_products or regenerate_stamp,
        )

        # Generate sigma maps if missing or requested
        generate_sigmas(
            input_root=galwrap_config.input_root,
            product_root=galwrap_config.product_root,
            field=ficl.field,
            image_version=ficl.image_version,
            catalog_version=ficl.catalog_version,
            filter=ficl.filter,
            objects=objects,
            positions=positions,
            image_sizes=image_sizes,
            regenerate=regenerate_products or regenerate_sigma,
        )

        # Generate PSF if missing or requested
        generate_psf(
            input_root=galwrap_config.input_root,
            product_root=galwrap_config.product_root,
            filter=ficl.filter,
            pixscale=ficl.pixscale,
            regenerate=regenerate_products or regenerate_psf,
        )

        # Generate masks if missing or requested
        generate_masks(
            input_root=galwrap_config.input_root,
            product_root=galwrap_config.product_root,
            field=ficl.field,
            image_version=ficl.image_version,
            catalog_version=ficl.catalog_version,
            filter=ficl.filter,
            objects=objects,
            positions=positions,
            image_sizes=image_sizes,
            regenerate=regenerate_products or regenerate_mask,
        )

        # Generate feedfiles if missing or requested
        generate_feedfiles(
            input_root=galwrap_config.input_root,
            product_root=galwrap_config.product_root,
            output_root=galwrap_config.output_root,
            field=ficl.field,
            image_version=ficl.image_version,
            catalog_version=ficl.catalog_version,
            filter=ficl.filter,
            objects=objects,
            image_sizes=image_sizes,
            pixscale=ficl.pixscale,
            regenerate=regenerate_products or regenerate_feedfile,
            apply_sigma=apply_sigma,
            apply_psf=apply_psf,
            apply_mask=apply_mask,
        )
