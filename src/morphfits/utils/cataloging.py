"""Utility functions for writing MorphFITS catalogs.
"""

# Imports


from pathlib import Path
import re

import numpy as np
from astropy.io import fits

from .. import paths


# Constants


HEADERS = [
    "use",
    "field",
    "image version",
    "catalog version",
    "filter",
    "object",
    "return code",
    "flags",
    "convergence",
    "center x",
    "center y",
    "surface brightness",
    "effective radius",
    "sersic",
    "axis ratio",
    "position angle",
    "center x error",
    "center y error",
    "surface brightness error",
    "effective radius error",
    "sersic error",
    "axis ratio error",
    "position angle error",
]
"""Headers for the MorphFITS catalog CSV.
"""


FLAGS = {
    "1": 0,
    "2": 1,
    "A-1": 2,
    "A-2": 3,
    "A-3": 4,
    "A-4": 5,
    "A-5": 6,
    "A-6": 7,
    "C-1": 8,
    "C-2": 9,
    "H-1": 10,
    "H-2": 11,
    "H-3": 12,
    "H-4": 13,
    "I-1": 14,
    "I-2": 15,
    "I-3": 16,
    "I-4": 17,
    "I-5": 18,
}
"""GALFIT flags as written in the model headers, and their corresponding
bit-mask exponent, where 2 is the base.
"""


GALFIT_LOG_REGEX = "[\*|\[]?\d{1,10}[\.]\d{1,2}[\*|\[]?"
GALFIT_LOG_FLOAT_REGEX = "\d{1,10}[\.]\d{1,2}"
"""Regex for seven .2f numbers found in GALFIT logs, which may or may not be
enveloped by * or [] characters.

See Also
--------
`record_parameters`
    Function using this expression to record the fitting parameters from logs.
"""


FAIL = 495
"""GALFIT flags which indicate a failed run.

See Also
--------
README.md 
    Breakdown on binary flag values, and which flags result in failed runs. 
"""


# Functions


def get_galfit_flags(
    output_root: Path,
    field: str,
    image_version: str,
    catalog_version: str,
    filter: str,
    object: int,
) -> int:
    """Calculate and return the GALFIT flags bitmask from the raised flags
    during a GALFIT fitting for a given FICLO with a successful fit.

    Parameters
    ----------
    output_root : Path
        Path to root MorphFITS output directory.
    field : str
        Field of observation.
    image_version : str
        Version of image processing used for observation.
    catalog_version : str
        Version of cataloging used for observation.
    filter : str
        Filter used for observation.
    object : int
        Integer ID of object in catalog for observation.

    Returns
    -------
    int
        Flag bitmask for raised GALFIT flags.
    """
    # Get headers from model FITS file
    model_path = paths.get_path(
        "model_galfit",
        output_root=output_root,
        field=field,
        image_version=image_version,
        catalog_version=catalog_version,
        filter=filter,
        object=object,
    )
    model = fits.open(model_path)
    headers = model[2].header
    model.close()

    # Add to flags bitmask from 'FLAGS' header
    flags = 0
    for flag in headers["FLAGS"].split():
        flags += 2 ** FLAGS[flag]

    return flags


def get_galfit_parameters(
    output_root: Path,
    field: str,
    image_version: str,
    catalog_version: str,
    filter: str,
    object: int,
) -> list[str]:
    """Find and return the fitted parameters from a GALFIT log for a given FICLO with a
    successful fit.

    Parameters
    ----------
    output_root : Path
        Path to root MorphFITS output directory.
    field : str
        Field of observation.
    image_version : str
        Version of image processing used for observation.
    catalog_version : str
        Version of cataloging used for observation.
    filter : str
        Filter used for observation.
    object : int
        Integer ID of object in catalog for observation.

    Returns
    -------
    tuple[int, list[str], list[str]]
        Parameter convergence bitmask, list of GALFIT-fitted float
        morphology parameters with .2f precision, as strings, and their
        associated errors, also as strings.
    """
    # Open log as text file
    log_path = paths.get_path(
        "log_galfit",
        output_root=output_root,
        field=field,
        image_version=image_version,
        catalog_version=catalog_version,
        filter=filter,
        object=object,
    )
    with open(log_path, mode="r") as log_file:
        lines = log_file.readlines()

        # Find parameters and errors from specific line in log via regex
        i = 0
        while i < len(lines) - 8:
            if (
                ("---" in lines[i])
                and (lines[i][0] != "#")
                and ("Input image" in lines[i + 2])
            ):
                raw_parameters = re.findall(GALFIT_LOG_REGEX, lines[i + 7])
                errors = re.findall(GALFIT_LOG_FLOAT_REGEX, lines[i + 8])
                break
            else:
                i += 1

        # Strip parameters of non-digit characters
        convergence, parameters = 0, []
        for i in range(len(raw_parameters)):
            parameter = raw_parameters[i]

            #### Only care about convergence of size, sersic, and ratio
            if i in [3, 4, 5]:
                for fail_indicator in ["[", "]", "*"]:
                    if fail_indicator in parameter:
                        convergence += 2 ** (i - 3)
                        parameter = parameter.replace(fail_indicator, "")
            parameters.append(parameter)

    # Return convergence bitmask, cleaned parameters, and their errors
    return convergence, parameters, errors


def get_usability(return_code: int, flags: int, convergence: int) -> bool:
    """Get the usability of a GALFIT fitting from its return code, raised flags
    bitmask, and parameter convergence bitmask.

    Parameters
    ----------
    return_code : int
        Integer return code of GALFIT when run in a subprocess.
    flags : int
        Integer bitmask of raised GALFIT flags as found in the model headers.
        The mapping between flags and bits can be found in the README.
    convergence : int
        Integer bitmask of primary fitting parameters which failed to converge.
        The mapping between parameters and bits can be found in the README.

    Returns
    -------
    bool
        Whether or not the model is recommended for use.
    """
    return (return_code == 0) and ((flags & FAIL) == 0) and (convergence == 0)


def get_csv_row(
    field: str,
    image_version: str,
    catalog_version: str,
    filter: str,
    object: int,
    return_code: int,
    use: bool | None = None,
    flags: int | None = None,
    convergence: int | None = None,
    parameters: list[str] | None = None,
    errors: list[str] | None = None,
) -> list[str]:
    """Get a row of CSV data as a list of strings, with which to append to or
    update a MorphFITS catalog.

    Parameters
    ----------
    field : str
        Field of observation.
    image_version : str
        Version of image processing used for observation.
    catalog_version : str
        Version of cataloging used for observation.
    filter : str
        Filter used for observation.
    object : int
        Integer ID of object in catalog for observation.
    return_code : int
        Integer return code of GALFIT when run in a subprocess.
    use : bool | None, optional
        Whether the model is recommended for use, by default None (failed fit).
    flags : int | None, optional
        Integer bitmask of raised GALFIT flags as found in the model headers, by
        default None (failed fit).
    convergence : int | None, optional
        Integer bitmask of primary fitting parameters which failed to converge,
        by default None (failed fit).
    parameters : list[str] | None, optional
        List of GALFIT-fitted float morphology parameters with .2f precision, as
        strings, by default None (failed fit).
    errors : list[str] | None, optional
        List of .2f precision float errors associated with above morphology
        parameters, also as strings, by default None (failed fit).

    Returns
    -------
    list[str]
        _description_
    """
    # If use flag has not been passed, the fit failed
    # Thus, return a row of empty data
    if use is None:
        csv_row = [
            False,
            field,
            image_version,
            catalog_version,
            filter,
            object,
            return_code,
            0,
            0,
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
        ]
        return csv_row

    # If the use flag has been passed, the fit was successful
    else:
        csv_row = [
            use,
            field,
            image_version,
            catalog_version,
            filter,
            object,
            return_code,
            flags,
            convergence,
        ]
        for parameter in parameters:
            csv_row.append(parameter)
        for error in errors:
            csv_row.append(error)
        return csv_row
