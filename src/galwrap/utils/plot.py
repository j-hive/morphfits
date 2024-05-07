"""Visualize output data from GalWrap.
"""

# Imports


from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

# montagepy doesn't have well-typed init
# See montage.ipac.caltech.edu/MontageNotebooks/mSubimage.html for more info
from MontagePy.main import mSubimage

from .. import config


## Typing


from numpy import ndarray
from matplotlib.pyplot import Figure
from matplotlib.axes import Axes

from ..config import GalWrapConfig


# Functions


def make_cutouts(
    image_path: str | Path,
    stamp_path: str | Path,
    ra: float,
    dec: float,
    size: float = 6.0,
    remove_edge: bool = False,
):
    pass


def plot_filters(
    galwrap_config: GalWrapConfig,
    galaxy_id: int | float | str,
    figsize: tuple[int | float, int | float] = (30, 25),
    cmap: str = "icefire",
    normalize: bool = False,
    stretch: str = "power",
    power: float = 1.5,
):
    """Show effects of filters on galaxy.

    Parameters
    ----------
    galwrap_config : GalWrapConfig
        Configuration settings for this program execution.
    galaxy_id : int | float | str
        ID of galaxy.
    figsize : tuple[int  |  float, int  |  float], optional
        Dimensions of plot, by default (30, 25).
    cmap : str, optional
        Colourmap of plots, by default "icefire".
    normalize : bool, optional
        Flag to normalize data, by default False.
    stretch : str, optional
        Normalization algorithm, by default "power".
    power : float, optional
        Normalization power factor, by default 1.5.
    """
    # Create 5 rows of 6 subplots and flatten axes object
    subplots: tuple[Figure, ndarray] = plt.subplots(5, 6, figsize=figsize)
    fig, axes = subplots[0], subplots[1].ravel()

    # Plot filtered images in subplots
    for i in range(len(galwrap_config.filters)):
        image_ax: Axes = axes[i]
        filter = galwrap_config.filters[i]

        # Plot image via imshow
        vmax = np.max(sci_image) / 5.0
        image_ax.imshow(
            sci_image,
            cmap=cmap,
            origin="lower",
            norm=(
                simple_norm(sci_image, stretch=stretch, power=power, max_cut=vmax)
                if normalize
                else "linear"
            ),
        )

        # Show filter name in top left of subplot
        image_text = image_ax.text(
            x=0.01,
            y=1,
            s=filter.upper(),
            horizontalalignment="left",
            verticalalignment="top",
            transform=image_ax.transAxes,
            fontsize=20,
            color="k",
        )
        image_text.set_bbox(dict(facecolor="white", alpha=1, edgecolor="w"))

    # Plot mask in lower right subplot and show name in top left of subplot
    mask_ax: Axes = axes[-1]
    mask_ax.imshow(mask_image, origin="lower")
    mask_text = mask_ax.text(
        x=0.01,
        y=1,
        s="MASK",
        horizontalalignment="left",
        verticalalignment="top",
        transform=mask_ax.transAxes,
        fontsize=20,
        color="k",
    )
    mask_text.set_bbox(dict(facecolor="white", alpha=1, edgecolor="w"))

    # Hide all axes decorations
    for ax in axes:
        ax.axis("off")

    # Title and save plot
    fig.suptitle(f"{OBJ}{FIELD}_{IMVER}: {galaxy_id}", fontsize=50)
    fig.subplots_adjust(wspace=0.01, hspace=0.01)
    fig.savefig(
        HOME / "MORPHDIR" / "INSPECTION" / f"{OBJ}{FIELD}_{IMVER}.{CATVER}.png",
        bbox_inches="tight",
        dpi=300,
        transparent=False,
    )
