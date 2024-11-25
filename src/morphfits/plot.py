"""Visualize MorphFITS products and output.
"""

# Imports


import logging
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import colors as mplc
from matplotlib import pyplot as plt
from matplotlib import ticker as mplt
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


HISTOGRAM_SUBPLOT_SEPARATION = 0.3
HISTOGRAM_TITLE_SEPARATION = -0.175
HISTOGRAM_TYPE = "step"
HISTOGRAM_ALPHA = 0.5
MINIMUM_BINS = 4
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


def get_chi_data(
    catalog: pd.DataFrame, filters: list[str] | None = None
) -> tuple[np.ndarray, dict[str, list[float]]]:
    """Get reduced chi squared data for a histogram as a list of floats
    indexed by filter.

    Parameters
    ----------
    catalog : pd.DataFrame
        Data frame containing morphology fitting information.
    filters : list[str] | None, optional
        List of filters in this catalog, by default None (find in this
        function).

    Returns
    -------
    tuple[np.ndarray, dict[str, list[float]]]
        Dict containing float reduced chi squared data for each filter.
    """
    # Get list of all filters in catalog if not passed
    if filters is None:
        filters = misc.get_unique(catalog["filter"])

    # Get list of bins between 0 and 1 for detail, and above 1 for "bad fits"
    # But set them as 0 to 8 for equal spacing, and label correctly
    # i.e. "good fits" = [0, 0.2, 0.4, 0.6, 0.8, 1]
    # "bad fits" = (1, 100, >100]
    n_good = 5
    bins = np.arange(n_good + 3)

    # Initialize data as empty dict
    data = {}

    # Iterate over each filter in catalog
    for filter in filters:
        data[filter] = []

        # Iterate over each object's reduced chi squared
        for datum in catalog[catalog["filter"] == filter]["chi^2/nu"]:
            # Try to add this object's reduced chi squared value
            try:
                assert not np.isnan(datum)
                chi = float(datum)

                # Map "good" chi values from float[0, 1] to int[0, 4]
                if int(chi * n_good) <= n_good:
                    if int(chi * n_good) == n_good:
                        data[filter].append(n_good - 1)
                    else:
                        data[filter].append(int(chi * n_good))

                # Map "bad" chi values from float[0, inf) to int[5, 7]
                elif (chi > 1) and (chi <= 100):
                    data[filter].append(n_good)
                elif chi > 100:
                    data[filter].append(n_good + 1)
            except:
                continue

    # Return bins and chi data
    return bins, data


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
    num_bins = np.max([int(np.sqrt(len(catalog))), MINIMUM_BINS + 1])
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


def get_parameter_comparison_data(
    catalog: pd.DataFrame,
    numerator: str,
    denominator: str,
    filters: list[str] | None = None,
) -> tuple[np.ndarray, dict[str, list[float]]]:
    """Get comparison data for a histogram as a list of floats indexed by
    filter, along with the appropriate bins for this parameter set.

    Comparison data is data comparing two parameters from the merge catalog, as
    a quotient.

    Parameters
    ----------
    catalog : pd.DataFrame
        Data frame containing merge catalog fitting information.
    numerator : str
        Name of parameter in catalog to use for numerator, e.g. 'sersic'.
    denominator : str
        Name of parameter in catalog to use for denominator.
    filters : list[str] | None, optional
        List of filters in this catalog, by default None (find in this
        function).

    Returns
    -------
    tuple[np.ndarray, dict[str, list[float]]]
        Histogram bins for this parameter, and dict containing float fitted
        values for each filter.
    """
    # Get list of all filters in catalog if not passed
    if filters is None:
        filters = misc.get_unique(catalog["filter"])

    # Get catalog subset for only filters of interest
    sub_catalog = catalog[catalog["filter"].isin(filters)]

    # Get comparisons of parameters
    comparison = np.ma.masked_invalid(sub_catalog[numerator] / sub_catalog[denominator])

    # Get minimum and maximum values from comparison quotient
    min_value, max_value = np.nanmin(comparison), np.nanmax(comparison)

    # Add two bins if parameter only has one value
    if min_value == max_value:
        min_value -= 1
        max_value += 1

    # Get number of bins from length of sub-catalog
    num_bins = np.max([int(np.sqrt(len(sub_catalog))), MINIMUM_BINS + 1])

    # Get bins for this parameter
    bins = np.linspace(min_value, max_value, num_bins)

    # Initialize data as empty dict
    data = {}

    # Iterate over each filter of interest
    for filter in filters:
        filtered_catalog = sub_catalog[sub_catalog["filter"] == filter]
        data[filter] = []

        # Iterate over object in filter
        for i_loc in range(len(filtered_catalog)):
            row = filtered_catalog.iloc[i_loc]

            # Skip rows without information for parameters of interest
            if (numerator not in row) or (denominator not in row):
                continue

            # Try to add parameter comparison for this object
            try:
                comparison = row[numerator] / row[denominator]
                assert not np.isnan(comparison)
                data[filter].append(float(comparison))
            except:
                continue

    # Return bins and parameter data
    return bins, data


def get_y_ticks(
    max_count: int | float,
    num_ticks: int = 10,
    log: bool = True,
) -> list[int]:
    """Get a list of y tick positions for a plot based on the maximum y-value.

    Parameters
    ----------
    max_count : int | float
        Maximum y-value.
    num_ticks : int, optional
        Number of y tick values to generate, by default 6.
    log : bool, optional
        Return intervals for log scaled plots, by default True.

    Returns
    -------
    list[int]
        List of y tick positions.
    """
    # Get maximum y-value as integer
    if not isinstance(max_count, int):
        max_count = int(max_count)

    # Set maximum as 1 if max count is 0
    if max_count == 0:
        max_count = 1

    # Return log-increasing list if in log mode
    if log:
        exponent = int(np.log10(max_count))
        exponent_range = range(exponent)
        power_range = np.power(10, exponent_range)
        return [0] + list(power_range) + [max_count]

    # Return linear-increasing list if not in log mode
    else:
        # Set list of easy-to-decipher y-axis increments
        interval_base = np.power(10, np.arange(10))
        intervals = interval_base.copy()
        intervals = np.append(intervals, 2.5 * interval_base)
        intervals = np.append(intervals, 5 * interval_base)
        intervals.sort()

        # Get increment appropriate to maximum y-value and number of y-ticks
        tick_interval = max(intervals)
        for interval in intervals:
            tick_interval = interval
            if max_count / interval < num_ticks:
                break

        # Return list of y-ticks from 0 to maximum y-value, separated by
        # calculated intervals
        return [0] + list(range(tick_interval, max_count, tick_interval)) + [max_count]


## Secondary


def setup_subplots(
    rows: int,
    columns: int,
    title: str,
    spacing: float,
    figsize: tuple[int, int] = FIGURE_SIZE,
) -> tuple[Figure, np.ndarray[Axes]]:
    """Create a MPL figure with six subplots, in two rows by three columns.

    Parameters
    ----------
    rows : int
        Number of subplot rows.
    columns : int
        Number of subplot columns.
    title : str
        Title of plot.
    spacing : float
        Spacing between subplots, in pixels.
    figsize : tuple[int, int], optional
        Dimensions of figure, in inches, by default (12, 8).

    Returns
    -------
    tuple[Figure, np.ndarray[Axes]]
        Figure object for plot and Axes object per subplot.
    """
    # Clean and create plot
    plt.clf()
    fig, axs = plt.subplots(rows, columns, figsize=figsize)

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
    labels: str | dict[float, str] | None = None,
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
    labels : str | dict[float, str] | None, optional
        Labels for this sub-histogram's bins, as a list of strings or dict of
        strings indexed by their positions, by default None (N/A).
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
            linestyle=LINE_STYLES[line_style],
            label=filter,
        )

        # Increase maximum y-value if applicable
        if (len(count) > 0) and (np.max(count) > max_count):
            max_count = np.max(count)

    # Set ticks and labels
    ax.set_title(title, y=HISTOGRAM_TITLE_SEPARATION)
    if labels is not None:
        if isinstance(labels, list):
            ax.set_xticks(bins[:-1] + 0.5, labels)
        else:
            sorted_positions = sorted(labels)
            sorted_labels = [labels[pos] for pos in sorted_positions]
            ax.set_xticks(sorted_positions, sorted_labels)
    ax.set_yscale("symlog")
    y_ticks = get_y_ticks(max_count=max_count, log=True)
    ax.set_ylim(y_ticks[0], y_ticks[-1])
    ax.set_yticks(y_ticks)
    ax.get_yaxis().set_major_formatter(mplt.ScalarFormatter())


def sub_model(
    ax: Axes,
    image: np.ndarray,
    title: str,
    vmin: float | None = None,
    vmax: float | None = None,
    log: bool = False,
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
    log : bool, optional
        Plot with log normalization, by default False.
    """
    # Plot image in subplot
    ax.imshow(
        image,
        cmap=JHIVE_CMAP,
        vmin=vmin,
        vmax=vmax,
        norm=mplc.LogNorm(vmin=vmin, vmax=vmax) if log else None,
    )

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
    fig, axs = setup_subplots(
        rows=3,
        columns=3,
        title=title,
        spacing=HISTOGRAM_SUBPLOT_SEPARATION,
        figsize=(12, 12),
    )

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

    # Try plotting reduced chi squared sub-histogram
    try:
        chi_bins, chi_data = get_chi_data(catalog)
        sub_histogram(
            ax=axs[0][2],
            filters=filters,
            data=chi_data,
            bins=chi_bins,
            title="reduced chi squared",
            labels={
                0: "0",
                1: "0.2",
                2: "0.4",
                3: "0.6",
                4: "0.8",
                5: "1",
                6.5: ">100",
            },
        )
    except Exception as e:
        logger.debug(f"Skipping making chi sub-histogram - {e}.")

    # Try plotting parameter sub-histograms for each parameter of interest
    histogram_parameters = {
        "sersic": axs[1][0],
        "effective radius": axs[1][1],
        "surface brightness": axs[1][2],
        "axis ratio": axs[2][0],
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

    # Try plotting parameter comparison sub-histograms
    histogram_comparisons = {
        "effective radius fit to error ratio": (
            axs[2][1],
            "effective radius",
            "effective radius error",
        ),
        "surface brightness fit to estimate ratio": (
            axs[2][2],
            "surface brightness",
            "surface brightness guess",
        ),
    }
    for comparison, (ax, numerator, denominator) in histogram_comparisons.items():
        try:
            bins, data = get_parameter_comparison_data(catalog, numerator, denominator)
            sub_histogram(
                ax=ax,
                filters=filters,
                data=data,
                bins=bins,
                title=comparison,
            )
        except Exception as e:
            logger.debug(f"Skipping making {comparison} sub-histogram - {e}.")

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
    fig, axs = setup_subplots(
        rows=2, columns=3, title=title, spacing=MODEL_SUBPLOT_SEPARATION
    )

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
        sub_model(ax=axs[1][2], image=psf_image, title="PSF crop", log=True)
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
    total_objects = 0
    for ficl in runtime_settings.ficls:
        # Try to get objects from FICL
        try:
            logger.info(f"FICL {ficl}: Plotting models.")
            logger.info(
                f"Objects: {min(ficl.objects)} to {max(ficl.objects)} "
                + f"({len(ficl.objects)} objects)."
            )
            total_objects += len(ficl.objects)

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
        logger.info(
            f"FICL {ficl}: Plotted models - skipped {skipped}/{total_objects} objects."
        )
