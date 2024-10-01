"""Configure and setup a program execution of the MorphFITS program.
"""

# Imports


import gc
import logging
import itertools
import shutil
from pathlib import Path
from typing import Generator, Annotated
from datetime import datetime as dt

from astropy.table import Table
import yaml
from pydantic import BaseModel, StringConstraints
from tqdm import tqdm

from . import paths, ROOT
from .utils import logs, misc, science


# Constants


logger = logging.getLogger("CONFIG")
"""Logger object for this module.
"""


LowerStr = Annotated[str, StringConstraints(to_lower=True)]
"""Lowercase string type for pydantic model.
"""


DEFAULT_CATVER = "dja-v7.2"
"""Default catalog version, v7.2 of the DJA catalogue.
"""


REQUIRED_FIC_INPUTS = ["input_segmap", "input_catalog"]
REQUIRED_FICL_INPUTS = ["input_psf", "exposure", "science", "weights"]
"""Required input files to fit a FICL observation, in the format of their path names.
"""


# Classes


class FICL(BaseModel):
    """Configuration model for a single FICL.

    FICL is an abbreviation for the field, image version, catalog version, and
    filter of a JWST science observation. Each FICL corresponds to a single
    observation.

    Parameters
    ----------
    BaseModel : class
        Base pydantic model class for type validation.

    Attributes
    ----------
    field : str
        Field of observation, e.g. "abell2744clu".
    image_version : str
        Version string of JWST image processing, e.g. "grizli-v7.2".
    catalog_version : str
        Version string of JWST cataloging, e.g. "dja-v7.2".
    filter : str
        Observational filter band, e.g. "f140w".
    objects : list[int]
        Integer IDs of galaxies or cluster targets in catalog.
    pixscale : tuple[float, float]
        Pixel scale along x and y axes, in arcseconds per pixel.

    Notes
    -----
    All strings are converted to lowercase upon validation.
    """

    field: Annotated[str, StringConstraints(to_lower=True)]
    image_version: Annotated[str, StringConstraints(to_lower=True)]
    catalog_version: Annotated[str, StringConstraints(to_lower=True)]
    filter: Annotated[str, StringConstraints(to_lower=True)]
    objects: list[int]
    pixscale: tuple[float, float]

    def __str__(self) -> str:
        return "_".join(
            [self.field, self.image_version, self.catalog_version, self.filter]
        )


class MorphFITSConfig(BaseModel):
    """Configuration model for a program execution of MorphFITS.

    Parameters
    ----------
    BaseModel : class
        Base pydantic model class to enforce type validation upon creation.

    Attributes
    ----------
    input_root : Path
        Path to root input directory.
    output_root : Path
        Path to root output directory.
    product_root : Path
        Path to root products directory.
    run_root : Path
        Path to root runs directory.
    datetime : datetime
        Datetime at start of program run.
    run_number : int
        Number of run ordering runs with the same datetime.
    fields : list[str]
        List of fields over which to fit.
    image_versions : list[str]
        List of image versions over which to fit.
    filters : list[str]
        List of filters over which to fit.
    wrappers : list[str], optional
        List of morphology fitting algorithms to run, by default only GALFIT.
    morphfits_root : Path, optional
        Path to root directory containing all of input, product, and output
        directories, by default the repository root.
    galfit_path : Path
        Path to GALFIT binary, by default None.
    catalog_versions : list[str], optional
        List of catalog versions over which to fit, by default only the v7.2 DJA
        catalog.
    objects : list[int], optional
        List of object IDs within the catalog over which to fit, by default
        empty (all items in catalog).
    ficls : list[FICL], optional
        List of FICLs with the requisite input files, generated from this
        program run's FICLs, by default empty (not yet set).
    """

    input_root: Path
    output_root: Path
    product_root: Path
    run_root: Path
    datetime: dt
    run_number: int
    fields: list[str]
    image_versions: list[str]
    filters: list[str]
    wrappers: list[str] = ["galfit"]
    morphfits_root: Path = ROOT
    galfit_path: Path = None
    catalog_versions: list[str] = [DEFAULT_CATVER]
    objects: list[int] = []
    ficls: list[FICL] = []

    def setup_paths(
        self,
        pre_logger: logging.Logger,
        display_progress: bool = False,
        download_mode: bool = False,
    ):
        """Create product and output directories for each FICLO of this
        configuration object.

        Parameters
        ----------
        pre_logger : Logger
            Logging object for this module, prior to the creation of the run
            directory.
        display_progress : bool, optional
            Display setup progress via tqdm, by default False.
        download_mode : bool, optional
            Whether this program run is in download mode, by default False.
        """
        pre_logger.info("Creating product and output directories where missing.")

        # Create run directory for both download and fitting runs
        paths.get_path(
            name="run",
            run_root=self.run_root,
            field=self.fields[0],
            datetime=self.datetime,
            run_number=self.run_number,
        ).mkdir(parents=True, exist_ok=True)

        # Only create input and run directories for download run
        if download_mode:
            # Iterate over each possible FICL from configurations
            for ficl in self.ficls:
                # Iterate over each required input directory
                for path_name in ["input_data", "input_images"]:
                    # Create directory if it does not exist
                    paths.get_path(
                        name=path_name,
                        input_root=self.input_root,
                        field=ficl.field,
                        image_version=ficl.image_version,
                        filter=ficl.filter,
                    ).mkdir(parents=True, exist_ok=True)

            # Make PSFs directory
            paths.get_path(
                name="input_psfs",
                input_root=self.input_root,
            ).mkdir(parents=True, exist_ok=True)

        # Only create product and output directories for fitting run
        else:
            # Iterate over each possible FICLO from configurations
            for ficl in self.ficls:
                # Iterate over each object in FICL
                for object in (
                    tqdm(ficl.objects, unit="dir", leave=False)
                    if display_progress
                    else ficl.objects
                ):
                    # Make leaf FICLO directories
                    for path_name in ["product_ficlo", "output_ficlo"]:
                        # Create directory if it does not exist
                        paths.get_path(
                            name=path_name,
                            output_root=self.output_root,
                            product_root=self.product_root,
                            field=ficl.field,
                            image_version=ficl.image_version,
                            catalog_version=ficl.catalog_version,
                            filter=ficl.filter,
                            object=object,
                        ).mkdir(parents=True, exist_ok=True)

    def clean_paths(self, display_progress: bool = False):
        """Remove product and output directories for skipped FICLOs of this
        configuration object.

        Parameters
        ----------
        display_progress : bool, optional
            Display cleaning progress via tqdm, by default False.
        """
        logger.info("Cleaning product and output directories where skipped.")

        # Iterate over each possible FICLO from configurations
        for ficl in self.ficls:
            # Iterate over each object in FICL
            for object in (
                tqdm(ficl.objects, unit="dir", leave=False)
                if display_progress
                else ficl.objects
            ):
                # Clean product paths
                for path_name in ["stamp", "sigma", "psf", "mask"]:
                    # Remove FICLO products directory if any missing
                    if not paths.get_path(
                        name=path_name,
                        product_root=self.product_root,
                        field=ficl.field,
                        image_version=ficl.image_version,
                        catalog_version=ficl.catalog_version,
                        filter=ficl.filter,
                        object=object,
                    ).exists():
                        shutil.rmtree(
                            paths.get_path(
                                name="product_ficlo",
                                product_root=self.product_root,
                                field=ficl.field,
                                image_version=ficl.image_version,
                                catalog_version=ficl.catalog_version,
                                filter=ficl.filter,
                                object=object,
                            )
                        )
                        break

                # Clean output paths
                for wrapper in self.wrappers:
                    # Remove FICLO outputs directory if any models missing
                    if not paths.get_path(
                        name=f"model_{wrapper}",
                        output_root=self.output_root,
                        field=ficl.field,
                        image_version=ficl.image_version,
                        catalog_version=ficl.catalog_version,
                        filter=ficl.filter,
                        object=object,
                    ).exists():
                        shutil.rmtree(
                            paths.get_path(
                                name="output_ficlo",
                                output_root=self.output_root,
                                field=ficl.field,
                                image_version=ficl.image_version,
                                catalog_version=ficl.catalog_version,
                                filter=ficl.filter,
                                object=object,
                            )
                        )
                        break

    def write(self):
        """Write configurations settings for a program run to a YAML file in the
        corresponding run directory.
        """
        logger.info("Recording configuration settings for this run.")

        # Convert Path objects to strings
        write_config = self.__dict__
        for key in write_config:
            if isinstance(write_config[key], Path):
                write_config[key] = str(write_config[key])

        # Convert FICL objects to dicts
        write_config["ficls"] = []
        for ficl in self.ficls:
            write_config["ficls"].append(ficl.__dict__)

        # Write config to file
        yaml.safe_dump(
            write_config,
            open(
                paths.get_path(
                    "run_config",
                    run_root=self.run_root,
                    field=self.fields[0],
                    datetime=self.datetime,
                    run_number=self.run_number,
                ),
                mode="w",
            ),
        )


# Functions


## Utility


def ficl_is_unset(config_dict: dict, key: str) -> bool:
    """Evaluate whether a FICL setting is not yet configured in a configuration
    dictionary.

    A setting is validly configured if its key name is in the dictionary, and
    its value is a non-empty list.

    Parameters
    ----------
    key : str
        Name of the FICL setting key.
    config_dict : dict
        Dictionary representing configuration settings.

    Returns
    -------
    bool
        Whether the passed FICL attribute is incorrectly configured in the
        dictionary.
    """
    return (
        (key not in config_dict)
        or (not isinstance(config_dict[key], list))
        or (len(config_dict[key]) < 1)
    )


def ficl_is_missing_input(
    config_dict: dict, field: str, image_version: str, filter: str, fic: bool = False
) -> bool:
    required_inputs = REQUIRED_FIC_INPUTS if fic else REQUIRED_FICL_INPUTS

    # Iterate over each required input path name
    for required_input in required_inputs:
        ## Get path to required input
        path_input = paths.get_path(
            required_input,
            input_root=config_dict["input_root"],
            field=field,
            image_version=image_version,
            filter=filter,
        )

        ## If path doesn't point to an input file, FICL is missing an input file
        if not path_input.exists():
            return True

    # If all paths point to files, FICL has all input files
    return False


def get_all_objects(
    input_root: Path, field: str, image_version: str, catalog_version: str
) -> list[int]:
    # Read input catalog
    path_input_catalog = paths.get_path(
        "input_catalog",
        input_root=input_root,
        field=field,
        image_version=image_version,
        catalog_version=catalog_version,
    )
    input_catalog = Table.read(path_input_catalog)

    # Return list of IDs as integers
    return [int(id_object) - 1 for id_object in input_catalog["id"]]


def get_objects(
    config_dict: dict,
    field: str,
    image_version: str,
    catalog_version: str,
) -> list[int]:
    # Set objects prior to any batch mode alterations
    ## Set object list as entire catalog range if not yet set or in batch mode
    if (
        (ficl_is_unset(config_dict=config_dict, key="objects"))
        or (config_dict["object_first"] is not None)
        or (config_dict["object_last"] is not None)
        or (config_dict["batch_n_process"] > 1)
    ):
        objects = get_all_objects(
            input_root=config_dict["input_root"],
            field=field,
            image_version=image_version,
            catalog_version=catalog_version,
        )
        first_possible_object = objects[0]
        last_possible_object = objects[-1]

    ## Set object list as set via CLI or YAML if set
    else:
        objects = config_dict["objects"]

    # Set objects as range if first or last object are provided
    ## Remove all objects before first object
    if config_dict["object_first"] is not None:
        while objects[0] < config_dict["object_first"]:
            objects.pop(0)

    ## Remove all objects after last object
    if config_dict["object_last"] is not None:
        while objects[-1] > config_dict["object_last"]:
            objects.pop(-1)

    # Set objects as sub-range of current range if in batch mode
    if config_dict["batch_n_process"] > 1:
        ## Get sub-range indices and values from batch settings
        start_index, stop_index = misc.get_unique_batch_limits(
            process_id=config_dict["batch_process_id"],
            n_process=config_dict["batch_n_process"],
            n_items=len(objects),
        )
        objects = objects[start_index:stop_index]

        ## Remove objects out of range
        while (len(objects) > 0) and (objects[0] < first_possible_object):
            objects.pop(0)
        while (len(objects) > 0) and (objects[-1] > last_possible_object):
            objects.pop(-1)

    # Return object list
    return objects


def get_loggers(
    morphfits_config: MorphFITSConfig, pre_logger: logging.Logger, pre_logger_path: Path
) -> tuple[logging.Logger, logging.Logger]:
    # Remove pre-program logger
    pre_logger_path.unlink()
    del pre_logger
    gc.collect()

    # Create program loggers
    logs.create_logger(
        filename=paths.get_path(
            "run_log",
            run_root=morphfits_config.run_root,
            field=morphfits_config.fields[0],
            datetime=morphfits_config.datetime,
            run_number=morphfits_config.run_number,
        )
    )
    global logger
    logger = logging.getLogger("CONFIG")
    main_logger = logging.getLogger("MORPHFITS")

    # Return program loggers
    return logger, main_logger


## Validate and Configure Settings


def set_paths(
    config_dict: dict,
    cli_settings: dict,
    download_mode: bool,
    pre_logger: logging.Logger,
    pre_logger_path: Path,
) -> dict:
    pre_logger.info("Configuring paths.")

    # Cast any path settings to Path objects
    for path_key in [
        "morphfits_root",
        "input_root",
        "output_root",
        "product_root",
        "run_root",
        "galfit_path",
    ]:
        if path_key in config_dict:
            config_dict[path_key] = paths.get_path_obj(config_dict[path_key])
        if cli_settings[path_key] is not None:
            config_dict[path_key] = paths.get_path_obj(cli_settings[path_key])

    # Set input root directory
    ## Terminate if input root not set
    if "input_root" not in config_dict:
        pre_logger_path.unlink()
        raise ValueError("Input root not configured, terminating.")

    ## Terminate if input root not found, AND not in download mode
    elif (not config_dict["input_root"].exists()) and (not download_mode):
        pre_logger_path.unlink()
        raise FileNotFoundError(f"Input root {config_dict['input_root']} not found.")

    ## Create input root if not found, and in download mode
    elif (not config_dict["input_root"].exists()) and (download_mode):
        config_dict["input_root"].mkdir(parents=True, exist_ok=True)

    # Set root directory, if not found, as parent of input root
    if "morphfits_root" not in config_dict:
        config_dict["morphfits_root"] = config_dict["input_root"].parent

    # Set product, output, and run directories from root directory
    for root_key in ["output_root", "product_root", "run_root"]:
        if root_key not in config_dict:
            config_dict[root_key] = paths.get_path(
                root_key, morphfits_root=config_dict["morphfits_root"]
            )

    # Return config dict with set paths
    return config_dict


def set_ficl_settings(
    config_dict: dict, cli_settings: dict, pre_logger: logging.Logger
) -> dict:
    pre_logger.info("Configuring FICLs.")

    # Set FICLO settings from CLI if passed, i.e. override config file with CLI
    for ficlo_key in [
        "fields",
        "image_versions",
        "catalog_versions",
        "filters",
        "objects",
        "wrappers",
    ]:
        if cli_settings[ficlo_key] is not None:
            config_dict[ficlo_key] = cli_settings[ficlo_key]

    # Set FICL settings from input root directories if still unset at this point
    input_root: Path = config_dict["input_root"]

    # Set fields to input root subdirectories, if unset
    if ficl_is_unset(config_dict=config_dict, key="fields"):
        config_dict["fields"] = paths.get_directories(path=input_root)

        ## Remove PSFs directory, as it is also an input root subdirectory
        for field in config_dict["fields"]:
            if field.name == "psfs":
                config_dict["fields"].remove(field)
                break

    # Set image versions to field subdirectories, if unset
    if ficl_is_unset(config_dict=config_dict, key="image_versions"):
        config_dict["image_versions"] = []
        for field in config_dict["fields"]:
            for image_version in paths.get_directories(path=(input_root / field)):
                config_dict["image_versions"].append(image_version)

    # Set catalog versions to default catalogue versions, if unset
    if ficl_is_unset(config_dict=config_dict, key="catalog_versions"):
        config_dict["catalog_versions"] = [DEFAULT_CATVER]

    # Set filters to image version subdirectories, if unset
    if ficl_is_unset(config_dict=config_dict, key="filters"):
        config_dict["filters"] = []
        for field in config_dict["fields"]:
            for image_version in config_dict["image_versions"]:
                path_input_data = input_root / field / image_version
                for filter in paths.get_directories(path=path_input_data):
                    config_dict["filters"].append(filter)

    # Return config dict with configured FICL settings
    return config_dict


def clean_filters(config_dict: dict, pre_logger: logging.Logger) -> dict:
    pre_logger.info("Cleaning filters.")

    # Iterate over each FICL in config dict
    for field in config_dict["fields"]:
        for image_version in config_dict["image_versions"]:
            for filter in config_dict["filters"]:
                ## Get path to input science frame from FIL
                science_path = paths.get_path(
                    name="science",
                    input_root=config_dict["input_root"],
                    field=field,
                    image_version=image_version,
                    filter=filter,
                )

                ## Skip filters with valid science paths
                if science_path.exists():
                    continue

                ## Retry getting science frame path using other formats, if failed
                ### All known clear filter arrangements
                possible_filters = [
                    f"{filter}-clear",
                    f"clear-{filter}",
                    f"{filter}-clearp",
                    f"clearp-{filter}",
                ]

                ### Iterate over each clear filter arrangement
                filter_found = False
                for possible_filter in possible_filters:
                    #### Get path to input science frame using new filter name
                    possible_science_path = paths.get_path(
                        name="science",
                        input_root=config_dict["input_root"],
                        field=field,
                        image_version=image_version,
                        filter=possible_filter,
                    )
                    if possible_science_path.exists():
                        pre_logger.warning(
                            f"Input data for filter '{filter}' "
                            + f"not found, replacing with '{possible_filter}'."
                        )
                        config_dict["filters"].remove(filter)
                        config_dict["filters"].append(possible_filter)
                        filter_found = True
                        break

                ### If valid filter name hasn't been found, remove filter
                if not filter_found:
                    pre_logger.warning(f"Filter '{filter}' not found, removing.")
                    config_dict["filters"].remove(filter)

    # Return config dict with valid filter names
    return config_dict


def set_batch_settings(
    config_dict: dict,
    object_first: int | None,
    object_last: int | None,
    batch_n_process: int,
    batch_process_id: int,
    pre_logger: logging.Logger,
) -> dict:
    # Set first object
    if object_first is None:
        ## Set to YAML value if CLI value not provided
        if "first_object" in config_dict:
            ### Set to None if YAML value is not an integer
            try:
                config_dict["object_first"] = int(config_dict["first_object"])
            except:
                pre_logger.warning(
                    f"first_object '{config_dict['first_object']}' "
                    + "set in configuration file invalid, ignoring."
                )
                config_dict["object_first"] = None
        ## Set to None if neither CLI nor YAML provided
        else:
            config_dict["object_first"] = None
    ## Set to CLI value if provided
    else:
        config_dict["object_first"] = object_first

    # Set last object
    if object_last is None:
        ## Set to YAML value if CLI value not provided
        if "last_object" in config_dict:
            ### Set to None if YAML value is not an integer
            try:
                config_dict["object_last"] = int(config_dict["last_object"])
            except:
                pre_logger.warning(
                    f"last_object '{config_dict['last_object']}' "
                    + "set in configuration file invalid, ignoring."
                )
                config_dict["object_last"] = None
        ## Set to None if neither CLI nor YAML provided
        else:
            config_dict["object_last"] = None
    ## Set to CLI value if provided
    else:
        config_dict["object_last"] = object_last

    # Set first and last object to None if less than zero or invalid range
    if (config_dict["object_first"] is not None) and (config_dict["object_first"] < 0):
        pre_logger.warning(f"First object {config_dict['object_first']} invalid.")
        config_dict["object_first"] = None
    if (config_dict["object_last"] is not None) and (config_dict["object_last"] < 0):
        pre_logger.warning(f"Last object {config_dict['object_last']} invalid.")
        config_dict["object_last"] = None
    if (
        (config_dict["object_first"] is not None)
        and (config_dict["object_last"] is not None)
        and (config_dict["object_first"] > config_dict["object_last"])
    ):
        pre_logger.warning(
            f"First and last object range ({config_dict['object_first']}, "
            + f"{config_dict['object_last']}) invalid, ignoring."
        )
        config_dict["object_first"] = None
        config_dict["object_last"] = None

    # Set number of processes in batch
    config_dict["batch_n_process"] = batch_n_process

    # Set ID of process in batch
    config_dict["batch_process_id"] = batch_process_id

    # Return config dict with batch mode settings configured
    return config_dict


def set_ficl_objects_download_mode(
    config_dict: dict, pre_logger: logging.Logger
) -> dict:
    pre_logger.info("Setting FICLs for download.")

    # Iterate over each FICL
    config_dict["objects"] = []
    config_dict["ficls"] = []
    for field in config_dict["fields"]:
        for catalog_version in config_dict["catalog_versions"]:
            for image_version in config_dict["image_versions"]:
                for filter in config_dict["filters"]:
                    ## Create FICL object and add to config dict FICL list
                    ficl = FICL(
                        field=field,
                        image_version=image_version,
                        catalog_version=catalog_version,
                        filter=filter,
                        objects=[-1],
                        pixscale=[-1, -1],
                    )
                    config_dict["ficls"].append(ficl)

    # Return config dict with FICL objects set
    return config_dict


def set_ficl_objects(
    config_dict: dict,
    pre_logger: logging.Logger,
    object_first: int | None = None,
    object_last: int | None = None,
    batch_n_process: int = 1,
    batch_process_id: int = 0,
) -> dict:
    pre_logger.info("Setting FICLs for run.")

    # Set batch mode settings - first and last objects, n, id
    config_dict = set_batch_settings(
        config_dict=config_dict,
        object_first=object_first,
        object_last=object_last,
        batch_n_process=batch_n_process,
        batch_process_id=batch_process_id,
        pre_logger=pre_logger,
    )

    # Iterate over each FICL permutation
    config_dict["ficls"] = []
    for field in config_dict["fields"]:
        for image_version in config_dict["image_versions"]:
            for catalog_version in config_dict["catalog_versions"]:
                ## Skip FIC if input catalog or input segmap missing
                if ficl_is_missing_input(
                    config_dict=config_dict,
                    field=field,
                    image_version=image_version,
                    filter="",
                    fic=True,
                ):
                    pre_logger.warning(
                        f"Field {field} and image version "
                        + f"{image_version} missing input files, skipping."
                    )
                    continue

                ## Get list of object IDs for FIC
                objects = get_objects(
                    config_dict=config_dict,
                    field=field,
                    image_version=image_version,
                    catalog_version=catalog_version,
                    batch_n_process=batch_n_process,
                    batch_process_id=batch_process_id,
                )

                ## Iterate over each filter
                for filter in config_dict["filters"]:
                    ### Skip FICL if any input files missing
                    if ficl_is_missing_input(
                        config_dict=config_dict,
                        field=field,
                        image_version=image_version,
                        filter=filter,
                    ):
                        pre_logger.warning(
                            "FICL "
                            + "_".join([field, image_version, catalog_version, filter])
                            + "missing input files, skipping."
                        )
                        continue

                    ### Get pixel scale from science file
                    path_science = paths.get_path(
                        "science",
                        input_root=config_dict["input_root"],
                        field=field,
                        image_version=image_version,
                        filter=filter,
                    )
                    pixscale = science.get_pixscale(path_science)

                    ### Create FICL object and add to config dict FICL list
                    ficl = FICL(
                        field=field,
                        image_version=image_version,
                        catalog_version=catalog_version,
                        filter=filter,
                        objects=objects,
                        pixscale=pixscale,
                    )
                    config_dict["ficls"].append(ficl)

    # Return config dict with FICL objects set
    return config_dict


def set_run_settings(config_dict: dict) -> dict:
    # Set datetime to current local time
    config_dict["datetime"] = dt.now()

    # Set run number to 1 by default
    run_number = 1

    ## Increase run number if other processes in same batch are found
    while paths.get_path(
        "run",
        run_root=config_dict["run_root"],
        field=config_dict["fields"][0],
        datetime=config_dict["datetime"],
        run_number=run_number,
    ).exists():
        run_number += 1

    # Set run number
    config_dict["run_number"] = run_number

    # Return config dict with run settings set
    return config_dict


## Main


def create_config(
    config_path: str | Path | None = None,
    morphfits_root: str | Path | None = None,
    input_root: str | Path | None = None,
    output_root: str | Path | None = None,
    product_root: str | Path | None = None,
    run_root: str | Path | None = None,
    fields: list[str] | None = None,
    image_versions: list[str] | None = None,
    catalog_versions: list[str] | None = None,
    filters: list[str] | None = None,
    objects: list[int] | None = None,
    object_first: int | None = None,
    object_last: int | None = None,
    batch_n_process: int = 1,
    batch_process_id: int = 0,
    wrappers: list[str] | None = None,
    galfit_path: str | Path | None = None,
    display_progress: bool = False,
    download: bool = False,
) -> MorphFITSConfig:
    """Create a MorphFITS configuration object from hierarchically preferred
    variables, in order of values from
        1. CLI call from terminal
        2. Specified config file,
        3. Filesystem discovery

    Parameters
    ----------
    config_path : str | Path | None, optional
        Path to user config yaml file, by default None (no user config file
        provided).
    morphfits_root : str | Path | None, optional
        Path to root directory of MorphFITS filesystem, by default None (not
        passed through CLI).
    input_root : str | Path | None, optional
        Path to root directory of input products, e.g. catalogs, science images,
        and PSFs, by default None (not passed through CLI).
    output_root : str | Path | None, optional
        Path to root directory of GALFIT output products, e.g. morphology model
        and plots, by default None (not passed through CLI).
    product_root : str | Path | None, optional
        Path to root directory of products generated by this program to execute
        GALFIT, e.g. cutouts/stamps, masks, and feedfiles, by default None (not
        passed through CLI).
    run_root : str | Path | None, optional
        Path to root directory of records generated by this program for each
        run, by default None (not passed through CLI).
    fields : list[str] | None, optional
        List of fields over which to execute GALFIT, by default None (not passed
        through CLI).
    image_versions : list[str] | None, optional
        List of image versions over which to execute GALFIT, by default None
        (not passed through CLI).
    catalog_versions : list[str] | None, optional
        List of catalog versions over which to execute GALFIT, by default None
        (not passed through CLI).
    filters : list[str] | None, optional
        List of filter bands over which to execute GALFIT, by default None (not
        passed through CLI).
    objects : list[int] | None, optional
        List of target IDs over which to execute GALFIT, for each catalog, by
        default None (not passed through CLI).
    object_first : int | None, optional
        ID of first object in range of objects to run a batch over, by default
        None (not passed through CLI, or not set).
        Overrides object parameter.
    object_last : int | None, optional
        ID of last object in range of objects to run a batch over, by default
        None (not passed through CLI, or not set).
        Overrides object parameter.
    batch_n_process : int, optional
        Number of cores over which to divide this program run, in terms of
        objects, by default 1.
    batch_process_id : int, optional
        Process number in batch run, i.e. which sub-range of object range over
        which to run, by default 0.
    wrappers : list[str], optional
        List of morphology fitting algorithms to run, by default only GALFIT.
    galfit_path : str | Path | None, optional
        Path to GALFIT binary file, by default None (not passed through CLI).
    display_progress : bool, optional
        Display progress via tqdm, by default False.
    download : bool, optional
        Create configuration in download mode, i.e. create directory structure
        if missing, by default False.

    Returns
    -------
    MorphFITSConfig
        A configuration object for this program execution.
    """
    # Create a temporary logger
    pre_logger_path = Path("tmp.log").resolve()
    logs.create_logger(filename=pre_logger_path)
    pre_logger = logging.getLogger("CONFIG")
    pre_logger.info("Configuring settings for run.")

    # Set all configurations from YAML file if passed
    if config_path is None:
        config_dict = {}
    else:
        pre_logger.info(f"Loading configuration settings from {config_path}.")
        config_dict = yaml.safe_load(open(config_path, mode="r"))

    # Set all paths
    config_dict = set_paths(
        config_dict=config_dict,
        cli_settings=locals(),
        download_mode=download,
        pre_logger=pre_logger,
        pre_logger_path=pre_logger_path,
    )

    # Set all FICL settings (fields, image versions, etc.)
    config_dict = set_ficl_settings(config_dict=config_dict, cli_settings=locals())

    # Set all filter names to valid filters corresponding with science frame locations
    config_dict = clean_filters(config_dict=config_dict, pre_logger=pre_logger)

    # Set all FICL objects (list[FICL])
    if download:
        config_dict = set_ficl_objects_download_mode(
            config_dict=config_dict, pre_logger=pre_logger
        )
    else:
        config_dict = set_ficl_objects(
            config_dict=config_dict,
            pre_logger=pre_logger,
            object_first=object_first,
            object_last=object_last,
            batch_n_process=batch_n_process,
            batch_process_id=batch_process_id,
        )

    # Set start datetime and run number
    config_dict = set_run_settings(config_dict=config_dict)

    # Create configuration object from config dict
    morphfits_config = MorphFITSConfig(**config_dict)

    # Terminate if fitter is GALFIT and binary file is not linked
    if ("galfit" in morphfits_config.wrappers) and (
        (morphfits_config.galfit_path is None)
        or (not morphfits_config.galfit_path.exists())
    ):
        raise FileNotFoundError("GALFIT binary file not found or linked.")

    # Setup directories where missing
    morphfits_config.setup_paths(
        pre_logger=pre_logger, display_progress=True, download_mode=download
    )

    # Create program logger and remove pre-program logger
    logger, main_logger = get_loggers(
        morphfits_config=morphfits_config,
        pre_logger=pre_logger,
        pre_logger_path=pre_logger_path,
    )
    main_logger.info("Starting MorphFITS.")

    # Display if batch mode
    if batch_n_process > 1:
        logger.info(f"Running in batch mode.")
        if len(morphfits_config.objects) > 0:
            logger.info(
                f"Batch object ID range: {morphfits_config.objects[0]} "
                + f"to {morphfits_config.objects[-1]}."
            )
        logger.info(f"Batch process: {batch_process_id} / {batch_n_process-1}")

    # Return configuration object
    return morphfits_config
