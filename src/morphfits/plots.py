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
from tqdm import tqdm

from . import paths


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
    product_root: Path,
    output_root: Path,
    field: str,
    image_version: str,
    catalog_version: str,
    filter: str,
    objects: list[int],
    columns: int = 25,
    display_progress: bool = False,
):
    """Plot all objects for a given FICL.

    Parameters
    ----------
    product_root : Path
        Path to root products directory.
    output_root : Path
        Path to root output directory.
    field : str
        Field of observation.
    image_version : str
        Image processing version of observation data.
    catalog_version : str
        Cataloguing version of objects in observation.
    filter : str
        Filter band of observation.
    objects : list[int]
        Objects to be plotted.
    columns : int, optional
        Number of columns in visualization, by default 12.
    display_progress : bool, optional
        Display progress via tqdm, by default False.
    """
    logger.info(
        f"Plotting all objects in FICL {'_'.join([field,image_version,catalog_version,filter])}."
    )

    # Get stamp paths for each object
    stamp_paths = {}
    for object in objects:
        stamp_path = paths.get_path(
            "stamp",
            product_root=product_root,
            field=field,
            image_version=image_version,
            catalog_version=catalog_version,
            filter=filter,
            object=object,
        )
        if not stamp_path.exists():
            pass
        else:
            stamp_paths[object] = stamp_path
        del stamp_path
        gc.collect()

    # Create new plot
    num_stamps = len(list(stamp_paths.keys()))
    rows = int(num_stamps / columns) + 1
    fig, axes = plt.subplots(rows, columns, figsize=(columns, rows), facecolor="black")
    plt.subplots_adjust(hspace=0, wspace=0)

    # Remove extra spots
    for i in range(columns - 1, columns - num_stamps % columns - 2, -1):
        fig.delaxes(axes[rows - 1, i])

    # Plot all objects
    for i in tqdm(range(len(objects))) if display_progress else range(len(objects)):
        object = objects[i]

        # Plot object stamp in current spot
        stamp_file = fits.open(stamp_paths[object])
        plt.subplot(rows, columns, i)
        plt.imshow(stamp_file["PRIMARY"].data, cmap=JHIVE_CMAP)
        plt.title(object, y=1, color="white", fontsize=16)
        plt.axis("off")

        # Clear memory
        stamp_file.close()
        del stamp_file
        gc.collect()

    # Save plot
    objects_path = paths.get_path(
        "ficl_objects",
        output_root=output_root,
        field=field,
        image_version=image_version,
        catalog_version=catalog_version,
        filter=filter,
    )
    plt.suptitle(
        "_".join([field, image_version, catalog_version, filter]) + " objects",
        color="white",
        fontsize=20,
    )
    plt.savefig(objects_path, bbox_inches="tight", pad_inches=0.0)
    plt.close()


def plot_model(
    product_root: Path,
    output_root: Path,
    field: str,
    image_version: str,
    catalog_version: str,
    filter: str,
    objects: list[int],
    wrapper: list[str],
    display_progress: bool = False,
):
    """Plot all models for a given FICL.

    Parameters
    ----------
    product_root : Path
        Path to root products directory.
    output_root : Path
        Path to root output directory.
    field : str
        Field of observation.
    image_version : str
        Image processing version of observation data.
    catalog_version : str
        Cataloguing version of objects in observation.
    filter : str
        Filter band of observation.
    objects : list[int]
        Objects to be plotted.
    wrapper : str
        Morphology fitting wrapper program name.
    display_progress : bool, optional
        Display progress via tqdm, by default False.
    """
    logger.info(
        f"Plotting models for FICL {'_'.join([field,image_version,catalog_version,filter])}."
    )

    # Iterate over each object in FICL
    for object in tqdm(objects) if display_progress else objects:
        # Get paths
        product_path_names = [
            "stamp",
            "sigma",
            "psf",
            "mask",
            wrapper + "_model",
            wrapper + "_plot",
        ]
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
            )
            for name in product_path_names
        }

        # Skip object if any products missing
        skip_object = False
        for path_name, path in object_paths.items():
            if ("plot" not in path_name) and (not path.exists()):
                if not display_progress:
                    logger.debug(f"Skipping object {object}, missing {path_name}.")
                skip_object = True
                break
        if skip_object:
            continue

        if not display_progress:
            logger.info(f"Plotting model for object {object}.")

        # Load in data
        stamp_file = fits.open(object_paths["stamp"])
        sigma_file = fits.open(object_paths["sigma"])
        psf_file = fits.open(object_paths["psf"])
        mask_file = fits.open(object_paths["mask"])
        model_file = fits.open(object_paths[wrapper + "_model"])
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
        plt.subplots(2, 3, figsize=(12, 8), facecolor="black")
        plt.subplots_adjust(hspace=0.0, wspace=0.0)
        plt.suptitle(
            f"{'_'.join([field,image_version,catalog_version,filter,str(object)])} "
            + wrapper
            + " model",
            fontsize=20,
            color="white",
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
        plt.imshow(mask, cmap=JHIVE_CMAP)
        plt.title("mask", y=0, fontsize=20, color="black")
        plt.axis("off")

        plt.subplot(2, 3, 4)
        plt.imshow(norm_model, cmap=JHIVE_CMAP)
        plt.title(wrapper + " model", y=0, fontsize=20, color="white")
        plt.axis("off")

        plt.subplot(2, 3, 5)
        plt.imshow(residual, cmap=JHIVE_CMAP)
        plt.title("residuals", y=0, fontsize=20, color="white")
        plt.axis("off")

        plt.subplot(2, 3, 6)
        plt.imshow(psf, cmap=JHIVE_CMAP)
        plt.title("psf", y=0, fontsize=20, color="white")
        plt.axis("off")

        # Save plot
        plt.savefig(
            object_paths[wrapper + "_plot"], bbox_inches="tight", pad_inches=0.0
        )
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
