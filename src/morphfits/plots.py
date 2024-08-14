"""Visualize output data from MorphFITS.
"""

# Imports


import gc
import logging
from pathlib import Path
from datetime import datetime as dt

import numpy as np
import pandas as pd
from astropy.io import fits
import matplotlib as mpl
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


FIGURE_COLOR = "black"
TEXT_COLOR = "white"
FIGURE_SIZE = (12, 8)
FIGURE_TITLE_SIZE = 20
FIGURE_LABEL_SIZE = 12
TICK_LABEL_SIZE = 8
mpl.rcParams["figure.facecolor"] = FIGURE_COLOR
mpl.rcParams["legend.facecolor"] = FIGURE_COLOR
mpl.rcParams["axes.facecolor"] = FIGURE_COLOR
mpl.rcParams["axes.edgecolor"] = TEXT_COLOR
mpl.rcParams["lines.color"] = TEXT_COLOR
mpl.rcParams["text.color"] = TEXT_COLOR
mpl.rcParams["figure.figsize"] = FIGURE_SIZE
mpl.rcParams["figure.titlesize"] = FIGURE_TITLE_SIZE
mpl.rcParams["figure.labelsize"] = FIGURE_LABEL_SIZE
mpl.rcParams["image.cmap"] = JHIVE_CMAP
mpl.rcParams["axes.spines.top"] = False
mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["xtick.color"] = TEXT_COLOR
mpl.rcParams["xtick.labelcolor"] = TEXT_COLOR
mpl.rcParams["xtick.labelsize"] = TICK_LABEL_SIZE
mpl.rcParams["ytick.color"] = TEXT_COLOR
mpl.rcParams["ytick.labelcolor"] = TEXT_COLOR
mpl.rcParams["ytick.labelsize"] = TICK_LABEL_SIZE
mpl.rcParams["savefig.bbox"] = "tight"
mpl.rcParams["savefig.pad_inches"] = 0.1
"""Set default matplotlib configurations.
"""


LINE_STYLES = [
    ("solid", "-"),
    ("dotted", (0, (1, 1))),
    ("dashed", (0, (5, 5))),
    ("dashdotted", (0, (3, 5, 1, 5))),
    ("dashdotdotted", (0, (3, 5, 1, 5, 1, 5))),
    ("loosely dotted", (0, (1, 10))),
    ("loosely dashed", (0, (5, 10))),
    ("loosely dashdotted", (0, (3, 10, 1, 10))),
    ("loosely dashdotdotted", (0, (3, 10, 1, 10, 1, 10))),
    ("densely dotted", (0, (1, 1))),
    ("densely dashed", (0, (5, 1))),
    ("densely dashdotted", (0, (3, 1, 1, 1))),
    ("densely dashdotdotted", (0, (3, 1, 1, 1, 1, 1))),
]
current_line_style = 0
"""Set default line styles for rotation.
"""


# Utility Functions


def next_line_style() -> tuple:
    """Get the next line style in a rotating list.

    Returns
    -------
    tuple
        Line style definition for `linestyle` style parameter.
    """
    global current_line_style
    line_style = LINE_STYLES[current_line_style][1]
    current_line_style = current_line_style + 1
    return line_style


def reset_line_style():
    """Reset the line style iterator."""
    global current_line_style
    current_line_style = 0


def get_y_ticks(max_count: int | float, num_ticks: int = 6) -> list[int]:
    """Get a list of y tick positions based on the maximum value.

    Parameters
    ----------
    max_count : int | float
        Maximum y tick value.
    num_ticks : int, optional
        Number of y tick values to generate, by default 6.

    Returns
    -------
    list[int]
        List of y tick positions.
    """
    max_count = int(max_count)
    intervals = [1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 25000, 50000]
    tick_interval = max(intervals)
    for interval in intervals:
        tick_interval = interval
        if max_count / interval < num_ticks:
            break
    return [0] + list(range(tick_interval, max_count, tick_interval)) + [max_count]


# Plotting Functions


def plot_model(
    output_root: Path,
    product_root: Path,
    field: str,
    image_version: str,
    catalog_version: str,
    filter: str,
    objects: list[int],
    wrapper: str,
    display_progress: bool = False,
):
    """Plot all models for a given FICL.

    Parameters
    ----------
    output_root : Path
        Path to root output directory.
    product_root : Path
        Path to root products directory.
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
    for object in (
        tqdm(objects, unit="object", leave=False) if display_progress else objects
    ):
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

        # Normalize model to stamp
        stamp_min, stamp_max = np.min(stamp), np.max(stamp)
        if len(model_file) > 2:
            residual = model_file[3].data
        else:
            norm_model = np.copy(model)
            norm_model -= np.min(model)
            norm_model /= np.max(model)
            norm_model *= stamp_max - stamp_min
            norm_model += stamp_min
            residual = norm_model - stamp
            del norm_model

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
        gc.collect()

        # Plot each product
        plt.subplots(2, 3)
        plt.subplots_adjust(hspace=0.0, wspace=0.0)
        plt.suptitle(
            f"{'_'.join([field,image_version,catalog_version,filter,str(object)])} "
            + wrapper
            + " model",
        )

        plt.subplot(2, 3, 1)
        plt.imshow(stamp, cmap=JHIVE_CMAP)
        plt.title("stamp", y=0)
        plt.axis("off")

        plt.subplot(2, 3, 2)
        plt.imshow(sigma, cmap=JHIVE_CMAP, vmin=stamp_min, vmax=stamp_max)
        plt.title("sigma", y=0)
        plt.axis("off")

        plt.subplot(2, 3, 3)
        plt.imshow(mask, cmap=JHIVE_CMAP)
        plt.title("mask", y=0)
        plt.axis("off")

        plt.subplot(2, 3, 4)
        plt.imshow(
            model,
            cmap=JHIVE_CMAP,
            vmin=stamp_min,
            vmax=stamp_max,
        )
        plt.title(wrapper + " model", y=0)
        plt.axis("off")

        plt.subplot(2, 3, 5)
        plt.imshow(residual, cmap=JHIVE_CMAP, vmin=stamp_min, vmax=stamp_max)
        plt.title("residuals", y=0)
        plt.axis("off")

        plt.subplot(2, 3, 6)
        plt.imshow(psf, cmap=JHIVE_CMAP)
        plt.title("psf", y=0)
        plt.axis("off")

        # Save plot
        plt.savefig(object_paths[wrapper + "_plot"])
        plt.close()

        # Clear memory
        del object_paths
        del stamp
        del sigma
        del psf
        del mask
        del model
        del stamp_min
        del stamp_max
        del residual
        gc.collect()


def plot_histogram(run_root: Path, datetime: dt, run_number: int):
    """Plot a histogram for each important fitting parameter, across filters,
    for a given run.

    Parameters
    ----------
    run_root : Path
        Path to root directory of runs.
    datetime : dt
        Datetime of run, in 'yyyymmddThhMMss' format.
    run_number : int
        Number of run if there are multiple of the same datetime.
    """

    # Get paths
    parameters_path = paths.get_path(
        "parameters", run_root=run_root, datetime=datetime, run_number=run_number
    )
    histogram_path = paths.get_path(
        "histogram", run_root=run_root, datetime=datetime, run_number=run_number
    )

    # Load catalog as data frame
    run_catalog = pd.read_csv(parameters_path)
    filters = sorted(list(set(run_catalog["filter"])))
    parameters = {
        "use": "use for analysis",
        "convergence": "failed to converge",
        "surface brightness": "surface brightness",
        "effective radius": "effective radius",
        "sersic": "sersic n",
        "axis ratio": "axis ratio",
    }

    # Setup plot
    subplot_separation = 0.2
    title_separation = -0.175
    histogram_type = "step"
    alpha = 0.5
    num_bins = min(int(np.sqrt(len(run_catalog))), 50)
    plt.subplots(2, 3)
    plt.subplots_adjust(hspace=subplot_separation, wspace=subplot_separation)
    plt.suptitle(
        f"catalog from {datetime.strftime('%Y-%m-%dT%H:%M:%S')}.{str(run_number).rjust(2,'0')}",
    )

    # Plot histograms on each parameter per filter
    ## Plot histogram for usability
    plt.subplot(2, 3, 1)
    labels_0 = ["no", "yes"]
    bins_0 = np.arange(3)
    max_count = 0
    for filter in filters:
        quantized_use = []
        for use in run_catalog[run_catalog["filter"] == filter]["use"]:
            quantized_use.append(1 if use else 0)
        count, bin_edges, patches = plt.hist(
            quantized_use,
            histtype=histogram_type,
            alpha=alpha,
            bins=bins_0,
            edgecolor=TEXT_COLOR,
            linestyle=next_line_style(),
            label=filter,
        )
        if np.max(count) > max_count:
            max_count = np.max(count)
    plt.title(parameters["use"], y=title_separation)
    plt.xticks(bins_0[:-1] + 0.5, labels_0)
    plt.yticks(get_y_ticks(max_count=max_count))
    reset_line_style()

    ## Plot histogram for parameter convergence
    plt.subplot(2, 3, 2)
    labels_1 = ["effective radius", "sersic", "axis ratio"]
    bins_1 = np.arange(4)
    max_count = 0
    for filter in filters:
        quantized_flags = []
        for flags in run_catalog[run_catalog["filter"] == filter]["convergence"]:
            for bin_1 in bins_1:
                if flags & 2**bin_1:
                    quantized_flags.append(bin_1)
        count, bin_edges, patches = plt.hist(
            quantized_flags,
            histtype=histogram_type,
            alpha=alpha,
            bins=bins_1,
            edgecolor=TEXT_COLOR,
            linestyle=next_line_style(),
            label=filter,
        )
        if np.max(count) > max_count:
            max_count = np.max(count)
    plt.title(parameters["convergence"], y=title_separation)
    plt.xticks(bins_1[:-1] + 0.5, labels_1)
    plt.yticks(get_y_ticks(max_count=max_count))
    reset_line_style()

    ## Plot histogram for parameter values
    for i in range(2, len(parameters)):
        plt.subplot(2, 3, i + 1)
        max_count = 0
        parameter_data = run_catalog[list(parameters.keys())[i]]
        bins = np.linspace(parameter_data.min(), parameter_data.max(), num_bins)
        for filter in filters:
            parameter_data = run_catalog[run_catalog["filter"] == filter][
                list(parameters.keys())[i]
            ]
            count, bin_edges, patches = plt.hist(
                parameter_data,
                histtype=histogram_type,
                alpha=alpha,
                bins=bins,
                edgecolor=TEXT_COLOR,
                linestyle=next_line_style(),
                label=filter,
            )
            if (len(count) > 0) and (np.max(count) > max_count):
                max_count = np.max(count)
        plt.title(parameters[list(parameters.keys())[i]], y=title_separation)
        plt.yticks(get_y_ticks(max_count=max_count))
        reset_line_style()
    reset_line_style()

    # Save plot
    plt.savefig(histogram_path)
    plt.close()

    # Clear memory
    del run_catalog
    gc.collect()


def plot_objects(
    output_root: Path,
    product_root: Path,
    field: str,
    image_version: str,
    catalog_version: str,
    filter: str,
    objects: list[int],
    columns: int = 8,
    display_progress: bool = False,
):
    """Plot all objects for a given FICL.

    Parameters
    ----------
    output_root : Path
        Path to root output directory.
    product_root : Path
        Path to root products directory.
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
        Number of columns in visualization, by default 8.
    display_progress : bool, optional
        Display progress via tqdm, by default False.
    """
    logger.info(
        f"Plotting all objects in FICL {'_'.join([field,image_version,catalog_version,filter])}."
    )

    # Get stamp paths for each object
    stamp_paths = {}
    logger.info(
        "Checking stamps for FICL "
        + f"{'_'.join([field, image_version, catalog_version, filter])}."
    )
    for object in (
        tqdm(objects, unit="object", leave=False) if display_progress else objects
    ):
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
    extra_height = 0.2
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(2 * columns, 2 * (rows + extra_height)),
    )
    plt.subplots_adjust(
        top=(1 - extra_height / (2 * (rows + extra_height))), hspace=0, wspace=0
    )

    # Remove extra spots
    for i in range(columns - num_stamps % columns):
        plt.delaxes(axes[-1, -1 - i])

    # Plot all objects
    for i in (
        tqdm(range(len(list(stamp_paths.keys()))), unit="object", leave=False)
        if display_progress
        else range(len(list(stamp_paths.keys())))
    ):
        object = list(stamp_paths.keys())[i]

        # Plot object stamp in current spot
        stamp_file = fits.open(stamp_paths[object])
        plt.subplot(rows, columns, i + 1)
        plt.imshow(stamp_file["PRIMARY"].data, cmap=JHIVE_CMAP)
        plt.title(object, y=0, color="white", fontsize=16)
        plt.axis("off")

        # Clear memory
        stamp_file.close()
        del stamp_file
        gc.collect()

    # Write title
    plt.suptitle(
        "_".join([field, image_version, catalog_version, filter]) + " objects",
        color="white",
        fontsize=20,
    )
    plt.tight_layout(pad=0)

    # Save plot
    objects_path = paths.get_path(
        "ficl_objects",
        output_root=output_root,
        field=field,
        image_version=image_version,
        catalog_version=catalog_version,
        filter=filter,
    )
    plt.savefig(objects_path, bbox_inches="tight", pad_inches=0.0)
    plt.close()
