"""Utility functions for GalWrap.
"""

# Imports


from pathlib import Path
from datetime import datetime

from jinja2 import Template

from .. import DATA_ROOT


# Functions


def generate_feedfile(
    stamp_in: str | Path,
    stamp_out: str | Path,
    feedfile: str | Path,
    psfname: str | Path,
    bpmim: str | Path,
    noisim: str | Path,
    imsize: int,
    mag: float,
    ba: float,
    re: float,
):
    # Set configuration parameters from input
    feedfile_dict = {
        "stamp_in": stamp_in + "\t\t\t",
        "stamp_out": stamp_out + "\t",
        "noisim": noisim + "",
        "psfname": psfname,
        "bpmim": bpmim,
        "imsize": imsize,
        "mp": imsize / 2.0,
        "mag_round": round(mag, 2),
        "re_2": round(re, 2),
        "ba": ba,
    }

    # Open and write parameters to template
    with open(DATA_ROOT / "galfit_feedfile_template.jinja", "r") as feedfile_template:
        template = Template(feedfile_template.read())
    lines = template.render(feedfile_dict)
    with open(
        DATA_ROOT / "galfit_"
        + datetime.now().isoformat(timespec="seconds")
        + ".feedfile",
        "w",
    ) as feedfile:
        feedfile.write(lines)


if __name__ == "__main__":
    OBJ = "a370"
    FIELD = "ncf"
    IMVER = "v2p0"
    CATVER = 1
    FEEDDIR = (
        f"/arc/projects/canucs/MORPHDIR/FITDIR/{OBJ}{FIELD}/{IMVER}.{CATVER}"
        + "FEEDDIR"
    )
    ID = 2200540
    FILT = "test_filt"
    PIXNAME = "test_pixname"

    SCINAME = "../STAMPDIR/{}/{}_{}{}-{}_sci.fits".format(FILT, ID, OBJ, FIELD, FILT)
    STAMPOUT = "../GALFITOUTDIR/{}/{}_{}{}-{}_model.fits".format(
        FILT, ID, OBJ, FIELD, FILT
    )
    FEEDFILE = "{}{}/{}_{}{}-{}.feedfile".format(FEEDDIR, FILT, ID, OBJ, FIELD, FILT)
    PSFNAME = "../PSF/{}{}_{}_{}_psf_{}.{}_crop.fits".format(
        OBJ, FIELD, FILT, PIXNAME, IMVER, CATVER
    )
    MASKNAME = "../MASKDIR/{}_{}{}_mask.fits".format(ID, OBJ, FIELD)
    RMSNAME = "../RMSDIR/{}/{}_{}{}-{}_rms.fits".format(FILT, ID, OBJ, FIELD, FILT)
    IMSIZE = 2000
    MAG = 20.0
    A, B = 50.2, 120.0
    BA = B / A

    generate_feedfile(
        stamp_in=SCINAME,
        stamp_out=STAMPOUT,
        feedfile=FEEDFILE,
        psfname=PSFNAME,
        bpmim=MASKNAME,
        noisim=RMSNAME,
        imsize=IMSIZE,
        mag=MAG,
        ba=BA,
        re=A,
    )
