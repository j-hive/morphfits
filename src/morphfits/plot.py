"""Visualize MorphFITS products and output.
"""

# Imports


import logging
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import colors as mplc
from matplotlib import pyplot as plt
from matplotlib import rcParams as rcp
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from tqdm import tqdm

from . import settings
from .settings import RuntimeSettings
from .utils import misc, science


# Constants


## Logging


logger = logging.getLogger("PLOT")
"""Logging object for this module.
"""


logging.getLogger("matplotlib").setLevel(100)
logging.getLogger("PIL").setLevel(100)
"""Ignore matplotlib and PIL logs."""


## Plotting


CMAP_COLORS = {
    "black": {"pos": 0.0, "clr": mplc.to_rgb("#000000")},
    "black_2": {"pos": 0.008, "clr": mplc.to_rgb("#000000")},
    "gray": {"pos": 0.3, "clr": mplc.to_rgb("#676f7a")},
    "blue": {"pos": 0.5, "clr": mplc.to_rgb("#849cba")},
    "yellow": {"pos": 0.7, "clr": mplc.to_rgb("#facb73")},
    "white": {"pos": 1.0, "clr": mplc.to_rgb("#ffffff")},
}
"""J-HIVE colors mapping their names to their position and RGB values (as a
fraction of 1) for creating a MPL colormap.
"""

JHIVE_CMAP = mplc.LinearSegmentedColormap(
    "jhive_cmap",
    {
        "red": [(c["pos"], c["clr"][0], c["clr"][0]) for c in CMAP_COLORS.values()],
        "green": [(c["pos"], c["clr"][1], c["clr"][1]) for c in CMAP_COLORS.values()],
        "blue": [(c["pos"], c["clr"][2], c["clr"][2]) for c in CMAP_COLORS.values()],
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
rcp["figure.facecolor"] = FIGURE_COLOR
rcp["legend.facecolor"] = FIGURE_COLOR
rcp["axes.facecolor"] = FIGURE_COLOR
rcp["axes.edgecolor"] = TEXT_COLOR
rcp["lines.color"] = TEXT_COLOR
rcp["text.color"] = TEXT_COLOR
rcp["figure.figsize"] = FIGURE_SIZE
rcp["figure.titlesize"] = FIGURE_TITLE_SIZE
rcp["figure.labelsize"] = FIGURE_LABEL_SIZE
rcp["image.cmap"] = JHIVE_CMAP
rcp["axes.spines.top"] = False
rcp["axes.spines.right"] = False
rcp["xtick.color"] = TEXT_COLOR
rcp["xtick.labelcolor"] = TEXT_COLOR
rcp["xtick.labelsize"] = TICK_LABEL_SIZE
rcp["ytick.color"] = TEXT_COLOR
rcp["ytick.labelcolor"] = TEXT_COLOR
rcp["ytick.labelsize"] = TICK_LABEL_SIZE
rcp["savefig.bbox"] = "tight"
rcp["savefig.pad_inches"] = 0.1
"""Set default matplotlib configurations.
"""


LINE_STYLES = {
    "solid": "-",
    "dotted": (0, (1, 1)),
    "dashed": (0, (5, 5)),
    "dashdotted": (0, (3, 5, 1, 5)),
    "dashdotdotted": (0, (3, 5, 1, 5, 1, 5)),
    "loosely dotted": (0, (1, 10)),
    "loosely dashed": (0, (5, 10)),
    "loosely dashdotted": (0, (3, 10, 1, 10)),
    "loosely dashdotdotted": (0, (3, 10, 1, 10, 1, 10)),
    "densely dotted": (0, (1, 1)),
    "densely dashed": (0, (5, 1)),
    "densely dashdotted": (0, (3, 1, 1, 1)),
    "densely dashdotdotted": (0, (3, 1, 1, 1, 1, 1)),
}
"""Mapping from MPL line style names to their 'formulas'.

Used to rotate line styles for each filter in a histogram.
"""


HISTOGRAM_SUBPLOT_SEPARATION = 0.2
HISTOGRAM_TITLE_SEPARATION = -0.175
HISTOGRAM_TYPE = "step"
HISTOGRAM_ALPHA = 0.5
MINIMUM_BINS = 50
"""Histogram plotting setting constants.
"""


MODEL_SUBPLOT_SEPARATION = 0.0
"""Model plotting setting constants.
"""


# Functions


## Tertiary


def get_use_data(
    catalog: pd.DataFrame, filters: list[str] | None = None
) -> dict[str, list[int]]:
    #
    if filters is None:
        filters = misc.get_unique(catalog["filter"])

    #
    data = {}

    #
    for filter in filters:
        data[filter] = []

        #
        for usable in catalog[catalog["filter"] == filter]["use"]:
            #
            try:
                data[filter].append(1 if bool(usable) else 0)
            except:
                continue

    #
    return data


def get_convergence_data(
    catalog: pd.DataFrame, filters: list[str] | None = None
) -> dict[str, list[int]]:
    #
    if filters is None:
        filters = misc.get_unique(catalog["filter"])

    #
    bins = np.arange(4)

    #
    data = {}

    #
    for filter in filters:
        data[filter] = []

        #
        for datum in catalog[catalog["filter"] == filter]["convergence"]:
            #
            for bin in bins:
                try:
                    if int(datum) & 2**bin:
                        data[filter].append(bin)
                except:
                    continue

    #
    return data


def get_parameter_data(
    catalog: pd.DataFrame, parameter: str, filters: list[str] | None = None
) -> tuple[np.ndarray, dict[str, list[int]]]:

    #
    num_bins = np.min([int(np.sqrt(len(catalog))), MINIMUM_BINS])
    min_value, max_value = np.nanmin(catalog[parameter]), np.nanmax(catalog[parameter])

    #
    if min_value == max_value:
        min_value -= 1
        max_value += 1

    #
    bins = np.linspace(min_value, max_value, num_bins)

    #
    if filters is None:
        filters = misc.get_unique(catalog["filter"])

    #
    data = {}

    #
    for filter in filters:
        data[filter] = []

        #
        for datum in catalog[catalog["filter"] == filter][parameter]:
            #
            try:
                assert not np.isnan(datum)
                data[filter].append(float(datum))
            except:
                continue

    #
    return bins, data


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
    #
    if not isinstance(max_count, int):
        max_count = int(max_count)

    #
    intervals = [1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 25000, 50000]

    #
    tick_interval = max(intervals)
    for interval in intervals:
        tick_interval = interval

        #
        if max_count / interval < num_ticks:
            break

    #
    return [0] + list(range(tick_interval, max_count, tick_interval)) + [max_count]


## Secondary


def setup_six_subplots(title: str, spacing: float) -> tuple[Figure, np.ndarray[Axes]]:
    # Clean and create plot
    plt.clf()
    fig, axs = plt.subplots(2, 3)

    # Setup figure options
    plt.subplots_adjust(hspace=spacing, wspace=spacing)

    # Setup figure title
    fig.suptitle(title)

    # Return figure and axes
    return fig, axs


def sub_histogram(
    ax: Axes,
    filters: list[str],
    data: dict[str, list[int | float]],
    bins: np.ndarray,
    title: str,
    labels: str | None = None,
):
    #
    max_count = 0

    #
    for i in range(len(filters)):
        filter = filters[i]
        line_style = list(LINE_STYLES.keys())[i]
        filter_data = data[filter]

        #
        count, bin_edges, patches = ax.hist(
            x=filter_data,
            histtype=HISTOGRAM_TYPE,
            alpha=HISTOGRAM_ALPHA,
            bins=bins,
            edgecolor=TEXT_COLOR,
            linestyle=line_style,
            label=filter,
        )

        #
        if (len(count) > 0) and (np.max(count) > max_count):
            max_count = np.max(count)

    # Set ticks and labels
    ax.set_title(title, y=HISTOGRAM_TITLE_SEPARATION)
    if labels is not None:
        ax.set_xticks(bins[:-1] + 0.5, labels)
    # ax.set_yticks(get_y_ticks(max_count=max_count))
    ax.set_yscale("log")


def sub_model(
    ax: Axes,
    image: np.ndarray,
    title: str,
    vmin: float | None = None,
    vmax: float | None = None,
):
    #
    ax.imshow(image, cmap=JHIVE_CMAP, vmin=vmin, vmax=vmax)

    #
    ax.set_title(title, y=0)
    ax.set_axis_off()


def save(path: Path, fig: Figure):
    fig.savefig(path)
    fig.clear()


## Primary


def histogram(path: Path, title: str, catalog: pd.DataFrame):
    #
    filters = misc.get_unique(catalog["filter"])

    # Setup plot
    fig, axs = setup_six_subplots(title=title, spacing=HISTOGRAM_SUBPLOT_SEPARATION)

    #
    try:
        use_data = get_use_data(catalog)
        sub_histogram(
            ax=axs[0][0],
            filters=filters,
            data=use_data,
            bins=np.arange(3),
            title="use for analysis",
            labels=["no", "yes"],
        )
    except Exception as e:
        logger.debug(f"Skipping making usability sub-histogram - {e}.")

    #
    try:
        convergence_data = get_convergence_data(catalog)
        sub_histogram(
            ax=axs[0][1],
            filters=filters,
            data=convergence_data,
            bins=np.arange(4),
            title="failed to converge",
            labels=["effective radius", "sersic", "axis ratio"],
        )
    except Exception as e:
        logger.debug(f"Skipping making convergence sub-histogram - {e}.")

    #
    histogram_parameters = {
        "surface brightness": axs[0][2],
        "effective radius": axs[1][0],
        "sersic": axs[1][1],
        "axis ratio": axs[1][2],
    }
    for parameter, ax in histogram_parameters.items():
        try:
            bins, data = get_parameter_data(catalog, parameter)
            sub_histogram(
                ax=ax,
                filters=filters,
                data=data,
                bins=bins,
                title=parameter,
            )
        except Exception as e:
            logger.debug(f"Skipping making {parameter} sub-histogram - {e}.")

    # Save and clear plot
    save(path=path, fig=fig)


def model(
    path: Path,
    title: str,
    stamp_image: np.ndarray,
    sigma_image: np.ndarray,
    psf_image: np.ndarray,
    mask_image: np.ndarray,
    model_image: np.ndarray,
    residuals_image: np.ndarray,
    vmin: float,
    vmax: float,
):
    # Setup plot
    fig, axs = setup_six_subplots(title=title, spacing=MODEL_SUBPLOT_SEPARATION)

    #
    try:
        sub_model(ax=axs[0][0], image=stamp_image, title="masked stamp")
    except Exception as e:
        logger.debug(f"Skipping making stamp sub-histogram - {e}.")

    #
    try:
        sub_model(
            ax=axs[0][1], image=sigma_image, title="sigma map", vmin=vmin, vmax=vmax
        )
    except Exception as e:
        logger.debug(f"Skipping making sigma sub-histogram - {e}.")

    #
    try:
        sub_model(ax=axs[0][2], image=mask_image, title="mask")
    except Exception as e:
        logger.debug(f"Skipping making mask sub-histogram - {e}.")

    #
    try:
        sub_model(ax=axs[1][0], image=model_image, title="model", vmin=vmin, vmax=vmax)
    except Exception as e:
        logger.debug(f"Skipping making model sub-histogram - {e}.")

    #
    try:
        sub_model(
            ax=axs[1][1],
            image=residuals_image,
            title="masked residuals",
            vmin=vmin,
            vmax=vmax,
        )
    except Exception as e:
        logger.debug(f"Skipping making residuals sub-histogram - {e}.")

    #
    try:
        sub_model(ax=axs[1][2], image=psf_image, title="PSF crop")
    except Exception as e:
        logger.debug(f"Skipping making psf sub-histogram - {e}.")

    # Save and clear plot
    save(path=path, fig=fig)


# plot run histogram
# open run catalog
# plot histogram for each filter
# plot merge histogram
# open most recent merge catalog
# plot histogram for each filter
def all_histograms(runtime_settings: RuntimeSettings):
    #
    try:
        logger.info("Making histogram for run.")

        #
        run_catalog_path = settings.get_path(
            name="run_catalog",
            runtime_settings=runtime_settings,
            field=runtime_settings.ficls[0].field,
        )
        run_histogram_path = settings.get_path(
            name="run_histogram",
            runtime_settings=runtime_settings,
            field=runtime_settings.ficls[0].field,
        )

        #
        if not run_catalog_path.exists():
            raise FileNotFoundError("missing run catalog")

        #
        run_catalog = pd.read_csv(run_catalog_path)

        #
        histogram(
            path=run_histogram_path,
            title="MorphFITS Histogram - Run on "
            + misc.get_str_from_datetime(runtime_settings.date_time)
            + "."
            + misc.get_str_from_run_number(runtime_settings.run_number),
            catalog=run_catalog,
        )

    #
    except Exception as e:
        logger.debug(f"Skipping making run histogram - {e}.")

    #
    try:
        logger.info("Making histogram for all runs.")

        #
        catalog_path = settings.get_path(
            name="merge_catalog", runtime_settings=runtime_settings
        )
        histogram_path = settings.get_path(
            name="histogram", runtime_settings=runtime_settings
        )

        #
        if not catalog_path.exists():
            raise FileNotFoundError("missing merge catalog")

        #
        catalog = pd.read_csv(catalog_path)

        #
        histogram(
            path=histogram_path,
            title="MorphFITS Histogram as of "
            + misc.get_str_from_datetime(runtime_settings.date_time)
            + "."
            + misc.get_str_from_run_number(runtime_settings.run_number),
            catalog=catalog,
        )

    #
    except Exception as e:
        logger.debug(f"Skipping making merge histogram - {e}.")


# for each ficl
# for each object
# plot model
def all_models(runtime_settings: RuntimeSettings):
    # Iterate over each FICL in this run
    for ficl in runtime_settings.ficls:
        # Try to get objects from FICL
        try:
            logger.info(f"FICL {ficl}: Plotting models.")
            logger.info(
                f"Objects: {min(ficl.objects)} to {max(ficl.objects)} "
                + f"({len(ficl.objects)} objects)."
            )

            # Get iterable object list, displaying progress bar if flagged
            if runtime_settings.progress_bar:
                objects = tqdm(iterable=ficl.objects, unit="obj", leave=False)
            else:
                objects = ficl.objects

        # Catch any error opening FICL
        except Exception as e:
            logger.error(f"FICL {ficl}: Skipping plotting - {e}.")
            continue

        # Iterate over each object
        skipped = 0
        for object in objects:
            # Try
            try:
                # Get path to plot
                plot_path = settings.get_path(
                    name="plot_" + runtime_settings.morphology._name(),
                    path_settings=runtime_settings.roots,
                    ficl=ficl,
                    object=object,
                )

                # Skip previously fitted objects unless requested
                if plot_path.exists() and not runtime_settings.remake.plots:
                    if not runtime_settings.progress_bar:
                        logger.debug(f"Object {object}: Skipping plotting - exists.")
                    skipped += 1
                    continue

                # Get paths to output and product files
                stamp_path = settings.get_path(
                    name="stamp",
                    path_settings=runtime_settings.roots,
                    ficl=ficl,
                    object=object,
                )
                sigma_path = settings.get_path(
                    name="sigma",
                    path_settings=runtime_settings.roots,
                    ficl=ficl,
                    object=object,
                )
                psf_path = settings.get_path(
                    name="psf",
                    path_settings=runtime_settings.roots,
                    ficl=ficl,
                    object=object,
                )
                mask_path = settings.get_path(
                    name="mask",
                    path_settings=runtime_settings.roots,
                    ficl=ficl,
                    object=object,
                )
                model_path = settings.get_path(
                    name="model_" + runtime_settings.morphology._name(),
                    path_settings=runtime_settings.roots,
                    ficl=ficl,
                    object=object,
                )

                #
                if (
                    not stamp_path.exists()
                    or not sigma_path.exists()
                    or not psf_path.exists()
                    or not mask_path.exists()
                    or not model_path.exists()
                ):
                    if not runtime_settings.progress_bar:
                        logger.debug(
                            f"Object {object}: Skipping plotting - missing product or model."
                        )
                    skipped += 1
                    continue

                #
                stamp_image, stamp_headers = science.get_fits_data(stamp_path)
                sigma_image, sigma_headers = science.get_fits_data(sigma_path)
                psf_image, psf_headers = science.get_fits_data(psf_path)
                mask_image, mask_headers = science.get_fits_data(mask_path)
                model_image, model_headers = science.get_fits_data(model_path, hdu=2)

                #
                masked_stamp = np.where(
                    1 - mask_image, stamp_image, np.mean(stamp_image)
                )
                masked_model = np.where(
                    1 - mask_image, model_image, np.mean(model_image)
                )

                #
                stamp_min, stamp_max = np.nanmin(masked_stamp), np.nanmax(masked_stamp)

                #
                try:
                    residuals_image, residuals_headers = science.get_fits_data(
                        model_path, hdu=3
                    )
                    masked_residuals = np.where(
                        1 - mask_image, residuals_image, np.mean(residuals_image)
                    )

                #
                except:
                    normalized_model = np.copy(masked_model)
                    normalized_model -= np.min(masked_model)
                    normalized_model /= np.max(masked_model)
                    normalized_model *= stamp_max - stamp_min
                    normalized_model += stamp_min
                    masked_residuals = np.where(
                        1 - mask_image,
                        normalized_model - stamp_image,
                        np.mean(normalized_model - stamp_image),
                    )

                # Run GALFIT for object
                if not runtime_settings.progress_bar:
                    logger.debug(f"Object {object}: Plotting model.")
                model(
                    path=plot_path,
                    title=f"{ficl}_{object} "
                    + runtime_settings.morphology._upper_name()
                    + "Model",
                    stamp_image=masked_stamp,
                    sigma_image=sigma_image,
                    psf_image=psf_image,
                    mask_image=mask_image,
                    model_image=model_image,
                    residuals_image=masked_residuals,
                    vmin=stamp_min,
                    vmax=stamp_max,
                )

            # Catch any errors and skip to next object
            except Exception as e:
                if not runtime_settings.progress_bar:
                    logger.debug(f"Object {object}: Skipping plotting - {e}.")
                skipped += 1
                continue

        # Log number of skipped or failed objects
        logger.info(f"FICL {ficl}: Plotted models - skipped {skipped} objects.")
