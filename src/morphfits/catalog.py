"""Write a morphology fitting catalog for the current MorphFITS program
execution.
"""

# Imports


import logging
import warnings
import re
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from . import settings
from .settings import (
    FICL,
    RuntimeSettings,
    GALFITSettings,
    ImcascadeSettings,
    PysersicSettings,
)
from .utils import science
from .wrappers.galfit import GALWRAP_OUTPUT_END


# Constants


logger = logging.getLogger("CATALOG")
"""Logging object for this module.
"""


warnings.filterwarnings("ignore")
"""Ignore warnings from pandas.
"""


CATALOG_COLUMN_NAMES = [
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
"""Column names for a MorphFITS catalog.
"""

CATALOG_ROW_TYPE = tuple[
    bool,
    str,
    str,
    str,
    str,
    int,
    int,
    int,
    int,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
]
PARAMETER_LIST_TYPE = tuple[float, float, float, float, float, float, float]
"""Column types for a MorphFITS catalog, and parameter types, as types.
"""


GALFIT_LOG_REGEX = "[\*|\[]?\d{1,10}[\.]\d{1,2}[\*|\[]?"
GALFIT_LOG_FLOAT_REGEX = "\d{1,10}[\.]\d{1,2}"
"""Regex for seven .2f numbers found in GALFIT logs, which may or may not be
enveloped by * or [] characters.
"""


GALFIT_PARAMETER_FAIL_INDICATORS = ["[", "]", "*"]
"""Characters encasing a parameter in the GALFIT fit log file if its convergence
failed.
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


FAIL = 495
"""GALFIT flags which indicate a failed run.

See Also
--------
README.md 
    Breakdown on binary flag values, and which flags result in failed runs. 
"""


# Functions


## Tertiary


def get_galfit_flags(model_path: Path) -> int:
    """Calculate and return the GALFIT flags bitmask from the raised flags
    during a GALFIT fitting for a given FICLO with a successful fit.

    Parameters
    ----------
    model_path : Path
        Path to model FITS file.

    Returns
    -------
    int
        Flag bitmask for raised GALFIT flags.
    """
    # Get headers from model FITS file
    image, headers = science.get_fits_data(path=model_path, hdu=2)

    # Add to flags bitmask from 'FLAGS' header
    flags = 0
    for flag in headers["FLAGS"].split():
        flags += 2 ** FLAGS[flag]

    return flags


def get_parameters(
    fit_log_path: Path,
) -> tuple[int, int, PARAMETER_LIST_TYPE, PARAMETER_LIST_TYPE]:
    """Find and return the fitted parameters from a GALFIT log for a given FICLO with a
    successful fit.

    Parameters
    ----------
    fit_log_path : Path
        Path to GalWrap output fitting log file.

    Returns
    -------
    tuple[int, int, PARAMETER_LIST_TYPE, PARAMETER_LIST_TYPE]
        Fitting return code, parameter convergence bitmask, .2f precision
        morphology fitting parameters as strings, and their associated errors
        as strings.
    """
    # Open log as text file
    with open(fit_log_path, mode="r") as fit_log_file:
        lines = fit_log_file.readlines()

        # Find parameters and errors from specific line in log via regex
        i = 0
        while i < len(lines) - 8:
            if (
                ("---" in lines[i])
                and (lines[i][0] != "#")
                and ("Input image" in lines[i + 2])
            ):
                raw_parameters = re.findall(GALFIT_LOG_REGEX, lines[i + 7])
                raw_errors = re.findall(GALFIT_LOG_FLOAT_REGEX, lines[i + 8])
                break
            else:
                i += 1

        # Find return code from last line
        return_code = int(lines[-1].split(GALWRAP_OUTPUT_END)[-1].strip())

    # Strip parameters of non-digit characters
    convergence, parameters, errors = 0, [], []
    for i in range(len(raw_parameters)):
        parameter = raw_parameters[i]

        # Only care about convergence of size, sersic, and ratio
        if i in [3, 4, 5]:
            for fail_indicator in GALFIT_PARAMETER_FAIL_INDICATORS:
                if fail_indicator in parameter:
                    convergence += 2 ** (i - 3)
                    parameter = parameter.replace(fail_indicator, "")

        # Try casting to float and add to lists
        try:
            parameters.append(float(parameter))
        except:
            parameters.append(None)
        try:
            errors.append(float(raw_errors[i]))
        except:
            errors.append(None)

    # Return convergence bitmask, cleaned parameters, and their errors
    return return_code, convergence, parameters, errors


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


## Secondary


def get_catalog_row(
    model_path: Path,
    fit_log_path: Path,
    ficl: FICL,
    object: int,
    morphology: GALFITSettings | ImcascadeSettings | PysersicSettings,
) -> CATALOG_ROW_TYPE:
    # Fitting log should always exist
    if fit_log_path.exists():
        # Get catalog row from model file and fit log file if successful fitting
        if model_path.exists():
            # Get catalog row for each morphology
            if isinstance(morphology, GALFITSettings):
                # Get GALFIT flags from model file
                flags = get_galfit_flags(model_path)

                # Get fitting parameters from fit log file
                return_code, convergence, parameters, errors = get_parameters(
                    fit_log_path
                )

                # Get fitting usability from return code, flags, and convergence
                use = get_usability(return_code, flags, convergence)

                # Make catalog row from data
                catalog_row = [
                    use,
                    ficl.field,
                    ficl.image_version,
                    ficl.catalog_version,
                    ficl.filter,
                    object,
                    return_code,
                    flags,
                    convergence,
                ]

                # Add each parameter and error to row
                for parameter in parameters:
                    catalog_row.append(parameter)
                for error in errors:
                    catalog_row.append(error)

            elif isinstance(morphology, ImcascadeSettings):
                raise NotImplementedError("unimplemented morphology method")
            elif isinstance(morphology, PysersicSettings):
                raise NotImplementedError("unimplemented morphology method")
            else:
                raise NotImplementedError("unknown morphology method")

        # Get catalog row with empty parameters if failed fitting
        else:
            # Get return code from fit log file
            with open(fit_log_path, mode="r") as fit_log_file:
                lines = fit_log_file.readlines()

                # Return code should be last line of file
                return_code = None
                for line in lines:
                    if GALWRAP_OUTPUT_END in line:
                        return_code = int(line.split(GALWRAP_OUTPUT_END)[-1].strip())

                # Raise error if return code not found from file
                if return_code is None:
                    raise FileNotFoundError("return code missing")

            # Set catalog row as failed row with no parameters
            catalog_row = [
                False,
                ficl.field,
                ficl.image_version,
                ficl.catalog_version,
                ficl.filter,
                object,
                return_code,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            ]

        # Return catalog row
        return tuple(catalog_row)

    # Raise error if fitting log does not exist
    else:
        raise FileNotFoundError(f"fit log missing")


## Primary


def get_data(runtime_settings: RuntimeSettings) -> pd.DataFrame:
    # Initialize catalog as dict with empty columns
    catalog_data = {name: [] for name in CATALOG_COLUMN_NAMES}

    # Iterate over each FICL in this run
    for ficl in runtime_settings.ficls:
        # Try to get parameters for FICL
        try:
            logger.info(f"FICL {ficl}: Reading fitting parameters.")

            # Get iterable object list, displaying progress bar if flagged
            if runtime_settings.progress_bar:
                objects = tqdm(iterable=ficl.objects, unit="obj", leave=False)
            else:
                objects = ficl.objects

        # Catch any errors reading parameters for FICL
        except Exception as e:
            logger.error(f"FICL {ficl}: Skipping reading parameters - {e}.")

        # Iterate over each object
        for object in objects:
            # Try running GALFIT for object
            try:
                # Get path to model and log
                model_path = settings.get_path(
                    name="model_" + runtime_settings.morphology._name(),
                    path_settings=runtime_settings.roots,
                    ficl=ficl,
                    object=object,
                )
                fit_log_path = settings.get_path(
                    name="log_" + runtime_settings.morphology._name(),
                    path_settings=runtime_settings.roots,
                    ficl=ficl,
                    object=object,
                )

                # Get row of catalog data from model file and fit log file
                catalog_row = get_catalog_row(
                    model_path=model_path,
                    fit_log_path=fit_log_path,
                    ficl=ficl,
                    object=object,
                    morphology=runtime_settings.morphology,
                )

                # Iterate over each datum in row of catalog data
                for i in range(len(CATALOG_COLUMN_NAMES)):
                    # Add datum to corresponding list in catalog data dict
                    catalog_data[CATALOG_COLUMN_NAMES[i]].append(catalog_row[i])

            # Catch any errors and skip to next object
            except Exception as e:
                if not runtime_settings.progress_bar:
                    logger.debug(f"Object {object}: Skipping reading parameters - {e}.")
                continue

    # Return catalog as data frame
    return pd.DataFrame(catalog_data)


def update_temporary(runtime_settings: RuntimeSettings, ficl: FICL, objects: list[int]):
    #
    temp_catalog_path = settings.get_path(
        name="run_catalog",
        runtime_settings=runtime_settings,
        field=runtime_settings.ficls[0].field,
    )

    #
    temp_runtime_settings = runtime_settings.model_copy(deep=True)
    temp_runtime_settings.ficls = [ficl.model_copy(deep=True)]
    temp_runtime_settings.ficls[0].objects = objects

    #
    catalog_data = get_data(runtime_settings=temp_runtime_settings)

    #
    if temp_catalog_path.exists():
        temp_catalog = pd.read_csv(temp_catalog_path)
        catalog_data = pd.concat([temp_catalog, catalog_data], join="inner")

    # Write run catalog CSV file
    catalog_data.to_csv(temp_catalog_path, index=False)


def make_run(runtime_settings: RuntimeSettings, catalog_data: pd.DataFrame):
    # Get path to run catalog file
    run_catalog_path = settings.get_path(
        name="run_catalog",
        runtime_settings=runtime_settings,
        field=runtime_settings.ficls[0].field,
    )

    # Skip writing if missing data
    if len(catalog_data.index) == 0:
        logger.debug(f"Skipping making run catalog - missing data.")
        return

    # Write run catalog CSV file
    catalog_data.to_csv(run_catalog_path, index=False)


def make_merge(runtime_settings: RuntimeSettings, catalog_data: pd.DataFrame):
    # Get path to merge catalog file and its parent directory
    catalog_path = settings.get_path(name="catalog", runtime_settings=runtime_settings)
    catalog_dir_path = settings.get_path(
        name="output_catalogs", runtime_settings=runtime_settings
    )

    # Get paths to previous merge catalog files, sorted by filename
    previous_catalog_paths = sorted(list(catalog_dir_path.iterdir()))

    # Merge all previous catalogs if any exist
    # Remove any failed fit rows and refit rows, except the most recent fit
    if len(previous_catalog_paths) > 0:
        # Get first catalog sorted by date time and run number, as data frame
        merge_catalog = pd.read_csv(previous_catalog_paths[0]).dropna()

        # Merge every previous merge catalog file sorted by date time and run
        previous_catalog_paths.pop(0)
        while len(previous_catalog_paths) > 0:
            previous_catalog = pd.read_csv(previous_catalog_paths[0]).dropna()
            merge_catalog = pd.concat([merge_catalog, previous_catalog], join="inner")
            previous_catalog_paths.pop(0)

        # Merge current run's catalog data and remove empty rows and refit rows
        merge_catalog = pd.concat([merge_catalog, catalog_data], join="inner")
        merge_catalog = merge_catalog.dropna()
        merge_catalog = merge_catalog.drop_duplicates(
            subset=["field", "image version", "catalog version", "filter", "object"],
            keep="last",
        )

    # Write current run's catalog data if no previous catalogs exist
    else:
        merge_catalog = catalog_data

    # Skip writing if missing data
    if len(merge_catalog.index) == 0:
        logger.debug(f"Skipping making merge catalog - missing data.")
        return

    # Write merge catalog to CSV file
    merge_catalog.to_csv(catalog_path, index=False)


def make_all(runtime_settings: RuntimeSettings):
    # Get all fit parameters from output files for this run as a dict of lists
    try:
        logger.info("Reading fitting output files for catalog.")
        catalog_data = get_data(runtime_settings)

    # Catch any errors and return if un-skip-able error getting data
    except Exception as e:
        logger.error(f"Skipping making catalogs - {e}.")
        return

    # Try writing catalog for run
    try:
        logger.info("Making catalog for run.")
        make_run(runtime_settings, catalog_data)
    except Exception as e:
        logger.error(f"Skipping making run catalog - {e}.")

    # Try writing new aggregate catalog based on previous catalogs
    try:
        logger.info("Making new merge catalog.")
        make_merge(runtime_settings, catalog_data)
    except Exception as e:
        logger.error(f"Skipping making merge catalog - {e}.")
