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
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from tqdm import tqdm

from . import paths
from .utils import misc


# Constants


## Logging


logger = logging.getLogger("PLOTS")
"""Logger object for processes from this module.
"""


logging.getLogger("matplotlib").setLevel(100)
logging.getLogger("PIL").setLevel(100)
"""Ignore matplotlib and PIL logs."""


## Plotting


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


## Module


PARAMETER_LABELS = {
    "use": "use for analysis",
    "convergence": "failed to converge",
    "surface brightness": "surface brightness",
    "effective radius": "effective radius",
    "sersic": "sersic n",
    "axis ratio": "axis ratio",
}
"""Mapping for parameters from catalog headers to plot labels.
"""


# Functions


## Utility


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


## Sub-plotting


def setup_histogram(histogram_path: Path, type: str) -> tuple[Figure, np.ndarray[Axes]]:
    # Clean and create plot
    plt.clf()
    fig, axs = plt.subplots(2, 3)

    # Setup figure options
    subplot_separation = 0.2
    plt.subplots_adjust(hspace=subplot_separation, wspace=subplot_separation)

    # Setup figure title
    if type == "main":
        title = "MorphFITS Catalog Histogram"
    elif type == "ficl":
        name = "_".join(
            [
                histogram_path.parent.parent.parent.parent.name,
                histogram_path.parent.parent.parent.name,
                histogram_path.parent.parent.name,
                histogram_path.parent.name,
            ]
        )
        title = f"MorphFITS Catalog Histogram - {name}"
    else:
        name = histogram_path.parent.name
        title = f"Run Catalog Histogram - {name}"
    fig.suptitle(title)

    # Return figure and axes
    return fig, axs


def subplot_histogram(catalog: pd.DataFrame, ax: Axes, parameter: str):
    # Subplot settings, change these values to change how the plot looks
    title_separation = -0.175
    histogram_type = "step"
    alpha = 0.5
    num_bins = min(int(np.sqrt(len(catalog))), 50)

    # Get list of filters in catalog and set the ytick count to zero
    filters = misc.get_unique(catalog["filter"])
    max_count = 0

    # Set the bins for this histogram
    ## There are two bins for the 'use' histogram, 'yes' and 'no'
    if parameter == "use":
        bins = np.arange(3)
    ## There are three bins for the 'convergence' histogram, one for each important parameter
    elif parameter == "convergence":
        bins = np.arange(4)
    ## The range for each parameter histogram should be the range over all filters
    else:
        values = []
        for datum in catalog[parameter]:
            try:
                values.append(float(datum))
            except:
                continue
        min_value, max_value = np.min(values), np.max(values)
        if min_value == max_value:
            min_value -= 1
            max_value += 1
        bins = np.linspace(min_value, max_value, num_bins)

    # Plot the histogram
    ## Plot a histogram line for each filter for this parameter in the same subplot
    for filter in filters:
        ### Set the data for the first two histograms, which are word counts
        data = []
        for datum in catalog[catalog["filter"] == filter][parameter]:
            #### The first histogram range is boolean
            if parameter == "use":
                data.append(1 if datum else 0)
            #### The second histogram range is three important parameters
            elif parameter == "convergence":
                for bin in bins:
                    if datum & 2**bin:
                        data.append(bin)
            #### The other histograms are float distributions
            else:
                ##### Skip NaNs
                try:
                    data.append(float(datum))
                except:
                    continue

        ### Plot the histogram using the bins and data set above
        ### Note this is only the histogram for one filter
        count, bin_edges, patches = ax.hist(
            data,
            histtype=histogram_type,
            alpha=alpha,
            bins=bins,
            edgecolor=TEXT_COLOR,
            linestyle=next_line_style(),
            label=filter,
        )

        ### Increase the ytick count if the max frequency increased
        if (len(count) > 0) and (np.max(count) > max_count):
            max_count = np.max(count)

    # Set the subplot ticks and labels
    ## NOTE legend currently not implemented, seems more distracting than informative
    # if len(filters) > 1:
    #     ax.legend()
    if parameter == "use":
        ax.set_xticks(bins[:-1] + 0.5, ["no", "yes"])
    elif parameter == "convergence":
        ax.set_xticks(bins[:-1] + 0.5, ["effective radius", "sersic", "axis ratio"])
    ax.set_yticks(get_y_ticks(max_count=max_count))
    ax.set_title(PARAMETER_LABELS[parameter], y=title_separation)
    reset_line_style()


def save_histogram(histogram_path: Path, fig: Figure):
    fig.savefig(histogram_path)
    fig.clear()


## Plotting


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
            f"model_{wrapper}",
            f"plot_{wrapper}",
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
        model_file = fits.open(object_paths[f"model_{wrapper}"])
        stamp = stamp_file["PRIMARY"].data
        sigma = sigma_file["PRIMARY"].data
        psf = psf_file["PRIMARY"].data
        mask = mask_file["PRIMARY"].data
        model = model_file[2].data

        # Mask data
        mask_value = np.min(stamp)
        masked_stamp = np.where(1 - mask, stamp, np.mean(stamp))
        masked_sigma = np.where(1 - mask, sigma, np.mean(sigma))
        masked_model = np.where(1 - mask, model, np.mean(model))

        # Normalize model to stamp
        stamp_min, stamp_max = np.min(masked_stamp), np.max(masked_stamp)
        if len(model_file) > 2:
            masked_residual = np.where(
                1 - mask, model_file[3].data, np.mean(model_file[3].data)
            )
        else:
            norm_model = np.copy(masked_model)
            norm_model -= np.min(masked_model)
            norm_model /= np.max(masked_model)
            norm_model *= stamp_max - stamp_min
            norm_model += stamp_min
            masked_residual = np.where(
                1 - mask, norm_model - stamp, np.mean(norm_model - stamp)
            )
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
        plt.imshow(masked_stamp, cmap=JHIVE_CMAP)
        plt.title("masked stamp", y=0)
        plt.axis("off")

        plt.subplot(2, 3, 2)
        plt.imshow(sigma, cmap=JHIVE_CMAP, vmin=stamp_min, vmax=stamp_max)
        plt.title("sigma map", y=0)
        plt.axis("off")

        plt.subplot(2, 3, 3)
        plt.imshow(mask, cmap=JHIVE_CMAP)
        plt.title("mask", y=0)
        plt.axis("off")

        plt.subplot(2, 3, 4)
        plt.imshow(model, cmap=JHIVE_CMAP, vmin=stamp_min, vmax=stamp_max)
        plt.title(wrapper + " model", y=0)
        plt.axis("off")

        plt.subplot(2, 3, 5)
        plt.imshow(masked_residual, cmap=JHIVE_CMAP, vmin=stamp_min, vmax=stamp_max)
        plt.title("masked residuals", y=0)
        plt.axis("off")

        plt.subplot(2, 3, 6)
        plt.imshow(psf, cmap=JHIVE_CMAP)
        plt.title("cropped psf", y=0)
        plt.axis("off")

        # Save plot
        plt.savefig(object_paths[f"plot_{wrapper}"])
        plt.close()

        # Clear memory
        del object_paths
        del stamp
        del sigma
        del model
        del psf
        del mask
        del stamp_min
        del stamp_max
        gc.collect()


def plot_histogram(catalog: pd.DataFrame, histogram_path: Path, type: str):
    """Plot a MorphFITS histogram representing aggregate model fitting
    usability, convergence, and parameter values.

    Parameters
    ----------
    catalog : DataFrame
        DataFrame representing all catalog fitting data to be plotted, i.e.
        filtered catalog data.
    histogram_path : Path
        Path to which to write histogram PNG.
    type : str
        Source of histogram catalog, one of 'main', 'ficl', or 'run'.
    """
    # Setup plot
    fig, axs = setup_histogram(histogram_path=histogram_path, type=type)

    # Plot each subplot
    parameters = list(PARAMETER_LABELS.keys())
    for i in range(len(parameters)):
        row, col = int(i / 3), i % 3
        subplot_histogram(catalog=catalog, ax=axs[row][col], parameter=parameters[i])

    # Save and clear plot
    save_histogram(histogram_path=histogram_path, fig=fig)


def plot_histograms(
    output_root: Path,
    run_root: Path,
    field: str,
    datetime: dt,
    run_number: int,
):
    """Plot histograms representing fitting parameter distributions for the main
    MorphFITS catalog, each FICL, and the run catalog.

    Parameters
    ----------
    output_root : Path
        Path to root directory of output.
    run_root : Path
        Path to root directory of runs.
    field : str
        Field of run.
    datetime : dt
        Datetime of run, in 'yyyymmddThhMMss' format.
    run_number : int
        Number of run if there are multiple of the same datetime.
    """
    logger.info("Plotting fitted parameter distribution histograms.")

    # Plot run histogram if run catalog exists
    path_catalog_run = paths.get_path(
        "run_catalog",
        run_root=run_root,
        field=field,
        datetime=datetime,
        run_number=run_number,
    )
    path_histogram_run = paths.get_path(
        "run_histogram",
        run_root=run_root,
        field=field,
        datetime=datetime,
        run_number=run_number,
    )
    if path_catalog_run.exists():
        catalog_run = pd.read_csv(path_catalog_run)
        logger.info("Plotting parameter histogram for run.")
        plot_histogram(
            catalog=catalog_run, histogram_path=path_histogram_run, type="run"
        )

    # Plot main histogram
    path_catalog = paths.get_path("catalog", output_root=output_root)
    path_histogram = paths.get_path("histogram", output_root=output_root)
    catalog = pd.read_csv(path_catalog)
    logger.info("Updating main catalog parameter histogram.")
    plot_histogram(catalog=catalog, histogram_path=path_histogram, type="main")

    # Plot FICL histogram for each FICL found in catalog
    for cF in misc.get_unique(catalog["field"]):
        catalog_F = catalog[catalog["field"] == cF]
        for cI in misc.get_unique(catalog_F["image version"]):
            catalog_FI = catalog_F[catalog_F["image version"] == cI]
            for cC in misc.get_unique(catalog_FI["catalog version"]):
                catalog_FIC = catalog_FI[catalog_FI["catalog version"] == cC]
                for cL in misc.get_unique(catalog_FIC["filter"]):
                    catalog_FICL = catalog_FIC[catalog_FIC["filter"] == cL]
                    path_histogram_ficl = paths.get_path(
                        "ficl_histogram",
                        output_root=output_root,
                        field=cF,
                        image_version=cI,
                        catalog_version=cC,
                        filter=cL,
                    )
                    if path_histogram_ficl.parent.exists():
                        logger.info(
                            "Plotting parameter histogram for FICL "
                            + f"{'_'.join([cF,cI,cC,cL])}."
                        )
                        plot_histogram(
                            catalog=catalog_FICL,
                            histogram_path=path_histogram_ficl,
                            type="ficl",
                        )
                    else:
                        logger.warning(
                            f"Directory {path_histogram_ficl.parent} "
                            + "not found, skipping histogram."
                        )


## In Development


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
