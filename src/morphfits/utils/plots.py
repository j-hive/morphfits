"""Visualize output data from MorphFITS.
"""

# Imports


import gc
import logging
from pathlib import Path

import numpy as np
from astropy.io import fits
from matplotlib import pyplot as plt

from ..galwrap import paths


# Constants


logger = logging.getLogger("PLOTS")
"""Logger object for processes from this module.
"""


logging.getLogger("matplotlib").setLevel(100)
logging.getLogger("PIL").setLevel(100)
"""Ignore matplotlib and PIL logs."""


# Functions


def plot_products(
    field: str,
    image_version: str,
    catalog_version: str,
    filter: str,
    objects: list[int],
    product_root: Path | None = None,
    output_root: Path | None = None,
    morphology_version: str = "galwrap",
):
    if morphology_version == "galwrap":
        logger.info(
            f"Plotting products for FICL {'_'.join([field,image_version,catalog_version,filter])}."
        )

        # Iterate over each object in FICL
        for object in objects:
            # Get paths
            object_paths = {
                name: paths.get_path(
                    name,
                    product_root=product_root,
                    output_root=output_root,
                    field=field,
                    image_version=image_version,
                    catalog_version=catalog_version,
                    filter=filter,
                    object=object,
                    pixscale=0.04,
                )
                for name in ["stamp", "sigma", "psf", "mask", "model", "prodplot"]
            }

            # Skip if any files missing
            skip_object = False
            for path_name, path in object_paths.items():
                if (path_name != "prodplot") and (not path.exists()):
                    logger.debug(f"Skipping object {object}, missing {path_name}.")
                    skip_object = True
                    break
            if skip_object:
                continue

            logger.info(f"Plotting products for object {object}.")

            # Load in data
            stamp_file = fits.open(object_paths["stamp"])
            sigma_file = fits.open(object_paths["sigma"])
            psf_file = fits.open(object_paths["psf"])
            mask_file = fits.open(object_paths["mask"])
            model_file = fits.open(object_paths["model"])
            stamp = stamp_file["PRIMARY"].data
            sigma = sigma_file["PRIMARY"].data
            psf = psf_file["PRIMARY"].data
            mask = mask_file["PRIMARY"].data
            model = model_file[2].data
            norm_model = np.copy(model)
            norm_model -= np.min(model)
            norm_model /= np.max(model)
            norm_model *= np.max(stamp) - np.min(stamp)
            norm_model += np.min(stamp)
            residual = norm_model - stamp

            # Clear memory
            stamp_file.close()
            sigma_file.close()
            psf_file.close()
            mask_file.close()
            model_file.close()
            del stamp_file
            del sigma_file
            del psf_file
            del mask_file
            del model_file
            del model

            # Plot each product
            plt.subplots(2, 3, figsize=(12, 8))
            plt.subplots_adjust(hspace=0, wspace=0)
            plt.title(
                f"{'_'.join([field,image_version,catalog_version,filter])} Products"
            )

            plt.subplot(2, 3, 1)
            plt.imshow(stamp, cmap="magma")
            plt.title("stamp", y=0, fontsize=20, color="white")
            plt.axis("off")

            plt.subplot(2, 3, 2)
            plt.imshow(sigma, cmap="magma")
            plt.title("sigma", y=0, fontsize=20, color="white")
            plt.axis("off")

            plt.subplot(2, 3, 3)
            plt.imshow(psf, cmap="magma")
            plt.title("psf", y=0, fontsize=20, color="white")
            plt.axis("off")

            plt.subplot(2, 3, 4)
            plt.imshow(norm_model, cmap="magma")
            plt.title("model", y=0, fontsize=20, color="white")
            plt.axis("off")

            plt.subplot(2, 3, 5)
            plt.imshow(residual, cmap="magma")
            plt.title("residual", y=0, fontsize=20, color="white")
            plt.axis("off")

            plt.subplot(2, 3, 6)
            plt.imshow(mask, cmap="magma")
            plt.title("mask", y=0, fontsize=20, color="white")
            plt.axis("off")

            # Save plot
            plt.savefig(object_paths["prodplot"], bbox_inches="tight")
            plt.close()

            # Clear memory
            del stamp
            del sigma
            del psf
            del mask
            del norm_model
            del residual
            del object_paths
            gc.collect()


def plot_comparison(stamp_path: Path, model_path: Path, output_path: Path):
    logger.info(f"Plotting model comparisons for {model_path.name}.")

    # Load in data
    stamp = fits.open(stamp_path)["PRIMARY"].data
    model = fits.open(model_path)[2].data
    norm_model = np.copy(model)
    norm_model -= np.min(model)
    norm_model /= np.max(model)
    norm_model *= np.max(stamp) - np.min(stamp)
    norm_model += np.min(stamp)

    # Plot data
    plt.subplots(1, 3, figsize=(20, 6))
    plt.title(stamp_path.name[:-4])

    plt.subplot(1, 3, 1)
    plt.imshow(stamp, cmap="magma")
    plt.axis("off")
    plt.title("Original")

    plt.subplot(1, 3, 2)
    plt.imshow(norm_model, cmap="magma")
    plt.axis("off")
    plt.title("Model")

    plt.subplot(1, 3, 3)
    plt.imshow(norm_model - stamp, cmap="magma")
    plt.axis("off")
    plt.title("Residuals")

    # Save data
    plt.savefig(output_path, bbox_inches="tight")
