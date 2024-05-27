"""Visualize output data from MorphFITS.
"""

# Imports


import logging
from pathlib import Path

import numpy as np
from astropy.io import fits
from matplotlib import pyplot as plt


# Constants


logger = logging.getLogger("PLOTS")
"""Logger object for processes from this module.
"""


logging.getLogger("matplotlib").setLevel(100)
logging.getLogger("PIL").setLevel(100)
"""Ignore matplotlib and PIL logs."""


# Functions


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
