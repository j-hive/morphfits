"""Visualize output data from MorphFITS.
"""

# Imports


import gc
import logging
from pathlib import Path

import numpy as np
from astropy.io import fits
from matplotlib import pyplot as plt
from matplotlib import colors as mplc

from ..galwrap import paths


# Constants


logger = logging.getLogger("PLOTS")
"""Logger object for processes from this module.
"""


logging.getLogger("matplotlib").setLevel(100)
logging.getLogger("PIL").setLevel(100)
"""Ignore matplotlib and PIL logs."""


pos = [0.0, 0.008, 0.3, 0.5, 0.7, 1.0]
colours = [
    [0, 0, 0],
    [0, 0, 0],
    np.array([103, 111, 122]) / 255,
    np.array([132, 156, 186]) / 255,
    np.array([250, 203, 115]) / 255,
    [1, 1, 1],
]
colour_names = ["red", "green", "blue"]
JHIVE_CMAP = mplc.LinearSegmentedColormap(
    "jhive_cmap",
    {
        colour_names[i]: [
            (pos[j], colours[j][i], colours[j][i]) for j in range(len(pos))
        ]
        for i in range(3)
    },
    1024,
)
"""Colormap using J-HIVE colors.
"""


# Functions


def plot_objects(
    field: str,
    image_version: str,
    catalog_version: str,
    filter: str,
    objects: list[int],
    rows: int = 10,
    columns: int = 5,
    product_root: Path | None = None,
    output_root: Path | None = None,
    morphology_version: str = "galwrap",
):
    if morphology_version == "galwrap":
        logger.info(
            f"Plotting all objects in FICL {'_'.join([field,image_version,catalog_version,filter])}."
        )

        # Iterate over all objects
        i = 0
        all_objects_plotted = False
        while i < len(objects):
            plt.subplots(rows, columns, figsize=(2 * columns, 2 * rows))
            plt.subplots_adjust(hspace=0, wspace=0)

            # Iterate over (rows * columns) objects
            for j in range(rows * columns):
                # Get object with existing stamp
                object = objects[i]
                stamp_path = paths.get_path(
                    "stamp",
                    product_root=product_root,
                    field=field,
                    image_version=image_version,
                    catalog_version=catalog_version,
                    filter=filter,
                    object=object,
                )

                # Iterate over objects until all have been plotted
                while not stamp_path.exists():
                    del object
                    del stamp_path
                    gc.collect()

                    i += 1
                    if i >= len(objects):
                        all_objects_plotted = True
                        break
                    object = objects[i]
                    stamp_path = paths.get_path(
                        "stamp",
                        product_root=product_root,
                        field=field,
                        image_version=image_version,
                        catalog_version=catalog_version,
                        filter=filter,
                        object=object,
                    )

                # Terminate if all objects plotted
                if all_objects_plotted:
                    break

                # Store first and last object of plot
                if j == 0:
                    first_object = object
                else:
                    last_object = object

                # Load stamp and clear memory
                stamp_file = fits.open(stamp_path)
                stamp_data = stamp_file["PRIMARY"].data
                stamp_file.close()
                del stamp_path
                del stamp_file
                gc.collect()

                # Plot object stamp and clear memory
                plt.subplot(rows, columns, j + 1)
                plt.imshow(stamp_data, cmap=JHIVE_CMAP)
                plt.title(i, y=1, color="white", fontsize=20)
                plt.axis("off")
                del stamp_data
                gc.collect()

                # Pick next object
                i += 1

            # Get path, title and save plot
            objects_path = paths.get_path(
                "objects",
                field=field,
                image_version=image_version,
                catalog_version=catalog_version,
                filter=filter,
                object=first_object,
            )
            plt.suptitle(
                "_".join(
                    [field, image_version, catalog_version, filter]
                    + f" Objects {first_object} to {last_object}"
                )
            )
            plt.savefig(objects_path, bbox_inches="tight", pad_inches=0.0, dpi=60)
            plt.close()

            # Clear memory
            del all_objects_plotted
            del object
            del first_object
            del last_object
            del objects_path
            gc.collect()


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
            plt.imshow(stamp, cmap=JHIVE_CMAP)
            plt.title("stamp", y=0, fontsize=20, color="white")
            plt.axis("off")

            plt.subplot(2, 3, 2)
            plt.imshow(sigma, cmap=JHIVE_CMAP)
            plt.title("sigma", y=0, fontsize=20, color="white")
            plt.axis("off")

            plt.subplot(2, 3, 3)
            plt.imshow(psf, cmap=JHIVE_CMAP)
            plt.title("psf", y=0, fontsize=20, color="white")
            plt.axis("off")

            plt.subplot(2, 3, 4)
            plt.imshow(norm_model, cmap=JHIVE_CMAP)
            plt.title("model", y=0, fontsize=20, color="white")
            plt.axis("off")

            plt.subplot(2, 3, 5)
            plt.imshow(residual, cmap=JHIVE_CMAP)
            plt.title("residual", y=0, fontsize=20, color="white")
            plt.axis("off")

            plt.subplot(2, 3, 6)
            plt.imshow(mask, cmap=JHIVE_CMAP)
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
    # TODO docs
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
    plt.imshow(stamp, cmap=JHIVE_CMAP)
    plt.axis("off")
    plt.title("Original")

    plt.subplot(1, 3, 2)
    plt.imshow(norm_model, cmap=JHIVE_CMAP)
    plt.axis("off")
    plt.title("Model")

    plt.subplot(1, 3, 3)
    plt.imshow(norm_model - stamp, cmap=JHIVE_CMAP)
    plt.axis("off")
    plt.title("Residuals")

    # Save data
    plt.savefig(output_path, bbox_inches="tight")
