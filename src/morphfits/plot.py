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
    """Get usability data for a histogram as a list of 1s and 0s corresponding
    to an object fitting's usability, per filter.

    Parameters
    ----------
    catalog : pd.DataFrame
        Data frame containing morphology fitting information.
    filters : list[str] | None, optional
        List of filters in this catalog, by default None (find in this
        function).

    Returns
    -------
    dict[str, list[int]]
        Dict containing integer boolean data for each filter.
    """
    # Get list of all filters in catalog if not passed
    if filters is None:
        filters = misc.get_unique(catalog["filter"])

    # Initialize data as empty dict
    data = {}

    # Iterate over each filter in catalog
    for filter in filters:
        data[filter] = []

        # Iterate over each object's usability
        for usable in catalog[catalog["filter"] == filter]["use"]:
            # Try to add this object's usability in data
            try:
                data[filter].append(1 if bool(usable) else 0)
            except:
                continue

    # Return usability data
    return data


def get_convergence_data(
    catalog: pd.DataFrame, filters: list[str] | None = None
) -> dict[str, list[int]]:
    """Get convergence bitmask data for a histogram as a list of integers
    indexed by filter.

    Searches a catalog for each available filter, and for each filter, the
    convergence bitmask for each object.

    Parameters
    ----------
    catalog : pd.DataFrame
        Data frame containing morphology fitting information.
    filters : list[str] | None, optional
        List of filters in this catalog, by default None (find in this
        function).

    Returns
    -------
    dict[str, list[int]]
        Dict containing integer bitmask data for each filter.
    """
    # Get list of all filters in catalog if not passed
    if filters is None:
        filters = misc.get_unique(catalog["filter"])

    # Get list of binary positions for convergence bitmask
    bins = np.arange(4)

    # Initialize data as empty dict
    data = {}

    # Iterate over each filter in catalog
    for filter in filters:
        data[filter] = []

        # Iterate over each object's convergence
        for datum in catalog[catalog["filter"] == filter]["convergence"]:
            # Try to add this object's convergence flags
            # Adds 1 if second parameter failed, etc.
            for bin in bins:
                try:
                    if int(datum) & 2**bin:
                        data[filter].append(bin)
                except:
                    continue

    # Return convergence data
    return data


def get_parameter_data(
    catalog: pd.DataFrame, parameter: str, filters: list[str] | None = None
) -> tuple[np.ndarray, dict[str, list[float]]]:
    """Get parameter fitting data for a histogram as a list of floats indexed by
    filter, along with the appropriate bins for this parameter set.

    Parameters
    ----------
    catalog : pd.DataFrame
        Data frame containing morphology fitting information.
    parameter : str
        Name of parameter in catalog, e.g. 'sersic'.
    filters : list[str] | None, optional
        List of filters in this catalog, by default None (find in this
        function).

    Returns
    -------
    tuple[np.ndarray, dict[str, list[float]]]
        Histogram bins for this parameter, and dict containing float fitted
        values for each filter.
    """
    # Get number of bins, and minimum and maximum for this parameter
    num_bins = np.min([int(np.sqrt(len(catalog))), MINIMUM_BINS])
    min_value, max_value = np.nanmin(catalog[parameter]), np.nanmax(catalog[parameter])

    # Add two bins if parameter only has one value
    if min_value == max_value:
        min_value -= 1
        max_value += 1

    # Get bins for this parameter
    bins = np.linspace(min_value, max_value, num_bins)

    # Get list of all filters in catalog if not passed
    if filters is None:
        filters = misc.get_unique(catalog["filter"])

    # Initialize data as empty dict
    data = {}

    # Iterate over each filter in catalog
    for filter in filters:
        data[filter] = []

        # Iterate over object's fitted parameter value
        for datum in catalog[catalog["filter"] == filter][parameter]:
            # Try to add this object's parameter value
            try:
                assert not np.isnan(datum)
                data[filter].append(float(datum))
            except:
                continue

    # Return bins and parameter data
    return bins, data


def get_y_ticks(max_count: int | float, num_ticks: int = 6) -> list[int]:
    """Get a list of y tick positions for a plot based on the maximum y-value.

    Parameters
    ----------
    max_count : int | float
        Maximum y-value.
    num_ticks : int, optional
        Number of y tick values to generate, by default 6.

    Returns
    -------
    list[int]
        List of y tick positions.
    """
    # Get maximum y-value as integer
    if not isinstance(max_count, int):
        max_count = int(max_count)

    # Set list of easy-to-decipher y-axis increments
    intervals = [1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 25000, 50000]

    # Get increment appropriate to maximum y-value and number of y-ticks
    tick_interval = max(intervals)
    for interval in intervals:
        tick_interval = interval
        if max_count / interval < num_ticks:
            break

    # Return list of y-ticks from 0 to maximum y-value, separated by calculated
    # intervals
    return [0] + list(range(tick_interval, max_count, tick_interval)) + [max_count]


## Secondary


def setup_six_subplots(title: str, spacing: float) -> tuple[Figure, np.ndarray[Axes]]:
    """Create a MPL figure with six subplots, in two rows by three columns.

    Parameters
    ----------
    title : str
        Title of plot.
    spacing : float
        Spacing between subplots, in pixels.

    Returns
    -------
    tuple[Figure, np.ndarray[Axes]]
        Figure object for plot and Axes object per subplot.
    """
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
    """Plot a subplot for a MorphFITS histogram.

    Parameters
    ----------
    ax : Axes
        Axes object of the subplot to plot on.
    filters : list[str]
        List of filters over which to plot data.
    data : dict[str, list[int | float]]
        Dict mapping filters to data to be represented in this histogram.
    bins : np.ndarray
        Bins for this sub-histogram.
    title : str
        Title for this sub-histogram.
    labels : str | None, optional
        Labels for this sub-histogram's bins, by default None (N/A).
    """
    # Track maximum y-value for setting y-ticks
    max_count = 0

    # Iterate over each filter
    for i in range(len(filters)):
        filter = filters[i]
        line_style = list(LINE_STYLES.keys())[i]
        filter_data = data[filter]

        # Plot histogram for this parameter in subplot
        count, bin_edges, patches = ax.hist(
            x=filter_data,
            histtype=HISTOGRAM_TYPE,
            alpha=HISTOGRAM_ALPHA,
            bins=bins,
            edgecolor=TEXT_COLOR,
            linestyle=line_style,
            label=filter,
        )

        # Increase maximum y-value if applicable
        if (len(count) > 0) and (np.max(count) > max_count):
            max_count = np.max(count)

    # Set ticks and labels
    ax.set_title(title, y=HISTOGRAM_TITLE_SEPARATION)
    if labels is not None:
        ax.set_xticks(bins[:-1] + 0.5, labels)
    ax.set_yticks(get_y_ticks(max_count=max_count))
    ax.set_yscale("log")


def sub_model(
    ax: Axes,
    image: np.ndarray,
    title: str,
    vmin: float | None = None,
    vmax: float | None = None,
):
    """Plot an imshow subplot for a MorphFITS model and product comparison plot.

    Parameters
    ----------
    ax : Axes
        Axes object of the subplot to plot on.
    image : np.ndarray
        Image data for this subplot.
    title : str
        Title for this subplot.
    vmin : float | None, optional
        Minimum for color map scale, by default None (auto scale).
    vmax : float | None, optional
        Maximum for color map scale, by default None (auto scale).
    """
    # Plot image in subplot
    ax.imshow(image, cmap=JHIVE_CMAP, vmin=vmin, vmax=vmax)

    # Set title and turn off axes
    ax.set_title(title, y=0)
    ax.set_axis_off()


def save(path: Path, fig: Figure):
    """Save a figure to file.

    Parameters
    ----------
    path : Path
        Path to which to save figure.
    fig : Figure
        MPL plot to save to file.
    """
    fig.savefig(path)
    fig.clear()


## Primary


def histogram(path: Path, title: str, catalog: pd.DataFrame):
    """Plot the distribution of morphology fits.

    A MorphFITS histogram has six sub-histograms:
        1. usability
        2. convergence bitmask
        3. surface brightness
        4. effective radius
        5. sersic index
        6. axis ratio

    Parameters
    ----------
    path : Path
        Path to which to write histogram plot.
    title : str
        Title of histogram plot.
    catalog : pd.DataFrame
        Morphology fitting data from which to generate histograms.
    """
    # Get list of filters from catalog
    filters = misc.get_unique(catalog["filter"])

    # Setup plot
    fig, axs = setup_six_subplots(title=title, spacing=HISTOGRAM_SUBPLOT_SEPARATION)

    # Try plotting usability sub-histogram
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

    # Try plotting convergence sub-histogram
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

    # Try plotting parameter sub-histograms for each parameter of interest
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
    """Plot the image data of various output and product files for a FICLO.

    This plot contains six subplots:
        1. Stamp
        2. Sigma map
        3. Mask
        4. Model
        5. Residuals (model - stamp)
        6. PSF crop

    Parameters
    ----------
    path : Path
        Path to which to write plot.
    title : str
        Title of plot.
    stamp_image : np.ndarray
        Image data of stamp of object.
    sigma_image : np.ndarray
        Image data of sigma map of object.
    psf_image : np.ndarray
        Image data of PSF crop of object.
    mask_image : np.ndarray
        Image data of mask of object.
    model_image : np.ndarray
        Image data of model of object.
    residuals_image : np.ndarray
        Image data of residuals of object.
    vmin : float
        Minimum for color map scale.
    vmax : float
        Maximum for color map scale.
    """
    # Setup plot
    fig, axs = setup_six_subplots(title=title, spacing=MODEL_SUBPLOT_SEPARATION)

    # Try plotting stamp subplot
    try:
        sub_model(ax=axs[0][0], image=stamp_image, title="masked stamp")
    except Exception as e:
        logger.debug(f"Skipping making stamp sub-histogram - {e}.")

    # Try plotting sigma map subplot
    try:
        sub_model(
            ax=axs[0][1], image=sigma_image, title="sigma map", vmin=vmin, vmax=vmax
        )
    except Exception as e:
        logger.debug(f"Skipping making sigma sub-histogram - {e}.")

    # Try plotting mask subplot
    try:
        sub_model(ax=axs[0][2], image=mask_image, title="mask")
    except Exception as e:
        logger.debug(f"Skipping making mask sub-histogram - {e}.")

    # Try plotting model subplot
    try:
        sub_model(ax=axs[1][0], image=model_image, title="model", vmin=vmin, vmax=vmax)
    except Exception as e:
        logger.debug(f"Skipping making model sub-histogram - {e}.")

    # Try plotting residuals subplot
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

    # Try plotting PSF crop subplot
    try:
        sub_model(ax=axs[1][2], image=psf_image, title="PSF crop")
    except Exception as e:
        logger.debug(f"Skipping making psf sub-histogram - {e}.")

    # Save and clear plot
    save(path=path, fig=fig)


def all_histograms(runtime_settings: RuntimeSettings):
    """Plot all histograms for a MorphFITS program run.

    Plots in this order:
        1. run histogram
        2. merge histogram

    Parameters
    ----------
    runtime_settings : RuntimeSettings
        Settings for this program run.

    Raises
    ------
    FileNotFoundError
        Missing run or merge catalog.
    """
    # Try making run histogram
    try:
        logger.info("Making histogram for run.")

        # Get paths to run catalog and run histogram
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

        # Skip if missing run catalog
        if not run_catalog_path.exists():
            raise FileNotFoundError("missing run catalog")

        # Get catalog as pandas data frame
        run_catalog = pd.read_csv(run_catalog_path)

        # Plot histogram and write to file
        histogram(
            path=run_histogram_path,
            title="MorphFITS Histogram - Run on "
            + misc.get_str_from_datetime(runtime_settings.date_time)
            + "."
            + misc.get_str_from_run_number(runtime_settings.run_number),
            catalog=run_catalog,
        )

    # Catch errors and skip to merge histogram
    except Exception as e:
        logger.debug(f"Skipping making run histogram - {e}.")

    # Try making merge histogram
    try:
        logger.info("Making histogram for all runs.")

        # Get paths to merge catalog and histogram
        catalog_path = settings.get_path(
            name="merge_catalog", runtime_settings=runtime_settings
        )
        histogram_path = settings.get_path(
            name="histogram", runtime_settings=runtime_settings
        )

        # Skip if missing merge catalog
        if not catalog_path.exists():
            raise FileNotFoundError("missing merge catalog")

        # Get catalog as pandas data frame
        catalog = pd.read_csv(catalog_path)

        # Plot histogram and write to file
        histogram(
            path=histogram_path,
            title="MorphFITS Histogram as of "
            + misc.get_str_from_datetime(runtime_settings.date_time)
            + "."
            + misc.get_str_from_run_number(runtime_settings.run_number),
            catalog=catalog,
        )

    # Catch errors and skip
    except Exception as e:
        logger.debug(f"Skipping making merge histogram - {e}.")


def all_models(runtime_settings: RuntimeSettings):
    """Plot all model product comparison plots for this Morphfits program run.

    Plots for each object, for each FICL in this program run.

    Parameters
    ----------
    runtime_settings : RuntimeSettings
        Settings for this program run.
    """
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
            # Try making comparison plot
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

                # Skip object if missing any product or model
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

                # Get image data of each subplot image from FITS files
                stamp_image, stamp_headers = science.get_fits_data(stamp_path)
                sigma_image, sigma_headers = science.get_fits_data(sigma_path)
                psf_image, psf_headers = science.get_fits_data(psf_path)
                mask_image, mask_headers = science.get_fits_data(mask_path)
                model_image, model_headers = science.get_fits_data(model_path, hdu=2)

                # Mask stamp and model
                masked_stamp = np.where(
                    1 - mask_image, stamp_image, np.mean(stamp_image)
                )
                masked_model = np.where(
                    1 - mask_image, model_image, np.mean(model_image)
                )

                # Get minimum and maximum of stamp for plotting scale
                stamp_min, stamp_max = np.nanmin(masked_stamp), np.nanmax(masked_stamp)

                # Get residuals image from model if found in file
                try:
                    residuals_image, residuals_headers = science.get_fits_data(
                        model_path, hdu=3
                    )
                    masked_residuals = np.where(
                        1 - mask_image, residuals_image, np.mean(residuals_image)
                    )

                # Calculate residuals from model and stamp images if not found
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

                # Plot model comparison for this object
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
