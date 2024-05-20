"""Main program execution for morphology fitting.
"""

# Imports


import logging
from pathlib import Path

import typer

from .galwrap import paths

from .galwrap.objects import config
from .utils import logs


# Execution


# Getting config object from module
galwrap_config = config.galwrap_config

# Creating logger for program and module
main_logger = logs.create_logger()
logger = logging.getLogger("MAIN")

# TODO TEMP
# Set variables
import numpy as np
from astropy.io import fits
from astropy.table import Table

tab = fits.open(paths.get_path("file_photometry_catalog"))[1]
CAT = Table(tab.data)
TOT_FLAG = (
    (CAT["USE_PHOT"] == True)
    * (CAT["FLAG_STAR"] == False)
    * (CAT["DIST_FOV_NIRCAM"] > 10)
)
ID_all = CAT["NUMBER"][TOT_FLAG]
RA_all = CAT["RA"][TOT_FLAG]
Dec_all = CAT["DEC"][TOT_FLAG]
IMSIZE = CAT["KRON_RADIUS"][TOT_FLAG]
GALSIZE = CAT["A"][TOT_FLAG]
FIELDID = np.round(np.min(CAT["NUMBER"]), -5)
flux = CAT["FLUX_KRON_F200W"][TOT_FLAG]
cut = flux < 2e2


def setup_psf(galwrap_config: config.GalWrapConfig):
    for filter in galwrap_config.filters:
        pass


## Main

app = typer.Typer()


@app.command()
def main(name: str):
    print(name)


if __name__ == "__main__":
    app()
