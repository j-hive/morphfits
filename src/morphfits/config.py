"""Configure and setup a program execution of the MorphFITS program.
"""

# Imports


import gc
import logging
import tempfile
import shutil
from pathlib import Path
from typing import Annotated
from datetime import datetime as dt

from astropy.table import Table
import yaml
from pydantic import BaseModel, StringConstraints
from tqdm import tqdm

from . import paths
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

    Attributes
    ----------
    morphfits_root : Path
        Path to root directory containing all of input, product, and output
        directories.
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
    ficls : list[FICL]
        List of FICLs for this program run, i.e. FICLs with the required input
        files.
    wrappers : list[str], optional
        List of morphology fitting algorithms to run, by default only GALFIT.
    galfit_path : Path
        Path to GALFIT binary, by default None.
    """

    morphfits_root: Path
    input_root: Path
    output_root: Path
    product_root: Path
    run_root: Path
    datetime: dt
    run_number: int
    ficls: list[FICL]
    wrappers: list[str] = ["galfit"]
    galfit_path: Path = None

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
            field=self.ficls[0].field,
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
        write_config = self.__dict__.copy()
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
                    field=self.ficls[0].field,
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
    config_dict: dict,
    field: str,
    image_version: str,
    filter: str,
    fic: bool = False,
) -> bool:
    """Evaluate whether a FICL (or FIC) is missing input files required to run.

    Parameters
    ----------
    config_dict : dict
        Dictionary representing configuration settings.
    field : str
        Field of the target observation.
    image_version : str
        Image processing version of the target observation.
    filter : str
        Filter of the target observation.
    fic : bool, optional
        Consider a FIC and not a FICL, i.e. without the filter, by default
        False.

    Returns
    -------
    bool
        FICL is missing at least one input file.
    """
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
    input_root: Path,
    field: str,
    image_version: str,
    catalog_version: str,
) -> list[int]:
    """Obtain the entire object ID range in a catalog for a given FIC, in the
    form of a list of integers.

    Parameters
    ----------
    input_root : Path
        Path to the root of the input directory.
    field : str
        Field of the target observation.
    image_version : str
        Image processing version of the target observation.
    catalog_version : str
        Cataloging version of the target observation.

    Returns
    -------
    list[int]
        Object IDs in catalog, as a list of integers.
    """
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
    """Obtain the object ID list for a given FICL, considering the batch mode
    settings.

    Parameters
    ----------
    config_dict : dict
        Dictionary representing configuration settings.
    field : str
        Field of the target observation.
    image_version : str
        Image processing version of the target observation.
    catalog_version : str
        Cataloging version of the target observation.

    Returns
    -------
    list[int]
        Object IDs for a FICL, as a list of integers.
    """
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


def clean_filter(
    input_root: str,
    field: str,
    image_version: str,
    filter: str,
    pre_logger: logging.Logger,
) -> str | None:
    """Resolve a filter name in the case of an unnecessary 'clear' filter in the
    name.

    Parameters
    ----------
    input_root : str
        Path to root of input directory.
    field : str
        Field of target observation.
    image_version : str
        Image processing version of target observation.
    filter : str
        Filter of target observation.
    pre_logger : logging.Logger
        Logging object prior to the creation of the logging file.

    Returns
    -------
    str | None
        Valid matching filter name, or None, if not found.
    """
    # Get path to input science frame from FIL
    science_path = paths.get_path(
        name="science",
        input_root=input_root,
        field=field,
        image_version=image_version,
        filter=filter,
    )

    # Skip filters with valid science paths
    if science_path.exists():
        return filter

    # Retry getting science frame path using other formats, if failed
    ## All known clear filter arrangements
    possible_filters = [
        f"{filter}-clear",
        f"clear-{filter}",
        f"{filter}-clearp",
        f"clearp-{filter}",
        filter.replace("clear", "").replace("-", ""),
        filter.replace("clearp", "").replace("-", ""),
    ]

    ## Iterate over each clear filter arrangement
    for possible_filter in possible_filters:
        ### Get path to input science frame using new filter name
        possible_science_path = paths.get_path(
            name="science",
            input_root=input_root,
            field=field,
            image_version=image_version,
            filter=possible_filter,
        )

        ### Return new filter name if matching file found
        if possible_science_path.exists():
            pre_logger.warning(
                f"Input data for filter '{filter}' "
                + f"not found, replacing with '{possible_filter}'."
            )
            return possible_filter

    ## Return nothing if filter not found
    pre_logger.warning(f"Filter '{filter}' not found, skipping.")


def get_loggers(
    morphfits_config: MorphFITSConfig,
) -> tuple[logging.Logger, logging.Logger]:
    """Create and obtain the module and program loggers for this run.

    Parameters
    ----------
    morphfits_config : MorphFITSConfig
        Configuration object for this program run.

    Returns
    -------
    tuple[logging.Logger, logging.Logger]
        The config module logger, and the MorphFITS program logger.
    """
    # Create program loggers
    logs.create_logger(
        filename=paths.get_path(
            "run_log",
            run_root=morphfits_config.run_root,
            field=morphfits_config.ficls[0].field,
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
) -> dict:
    """Resolve and configure the path settings in the configuration dictionary.

    Parameters
    ----------
    config_dict : dict
        Dictionary representing configuration settings.
    cli_settings : dict
        Settings passed from the CLI call.
    download_mode : bool
        In download program mode, i.e. the input root can be created and not
        found.
    pre_logger : logging.Logger
        Logging object prior to the creation of the logging file.

    Returns
    -------
    dict
        The configuration dictionary with the path settings configured.

    Raises
    ------
    ValueError
        Input root not passed.
    FileNotFoundError
        Input root not found.
    """
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
        raise ValueError("Input root not configured, terminating.")

    ## Terminate if input root not found, AND not in download mode
    elif (not config_dict["input_root"].exists()) and (not download_mode):
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


def set_batch_settings(
    config_dict: dict,
    object_first: int | None,
    object_last: int | None,
    batch_n_process: int,
    batch_process_id: int,
    pre_logger: logging.Logger,
) -> dict:
    """Set the batch mode settings in the configuration dictionary.

    Note the settings aren't ingested in the configuration object to be created
    later, but are used in determining the object ranges of each FICL object.

    Parameters
    ----------
    config_dict : dict
        Dictionary representing configuration settings.
    object_first : int | None
        First object ID in a range to be run over.
    object_last : int | None
        Last object ID in a range to be run over.
    batch_n_process : int
        Number of processes in this batch.
    batch_process_id : int
        ID of this process in this batch, starting from 0.
    pre_logger : logging.Logger
        Logging object prior to the creation of the logging file.

    Returns
    -------
    dict
        The configuration dictionary with the batch mode settings configured.
    """
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
    if ("batch_n_process" not in config_dict) or (batch_n_process > 1):
        config_dict["batch_n_process"] = batch_n_process

    # Set ID of process in batch
    if ("batch_process_id" not in config_dict) or (batch_process_id > 0):
        config_dict["batch_process_id"] = batch_process_id

    # Return config dict with batch mode settings configured
    return config_dict


def set_ficls_download_mode(
    config_dict: dict, cli_settings: dict, pre_logger: logging.Logger
) -> dict:
    """Set the list of FICL objects for the configuration object, for the
    download program, i.e. with irrelevant objects and pixscales.

    Parameters
    ----------
    config_dict : dict
        Dictionary representing configuration settings.
    pre_logger : logging.Logger
        Logging object prior to the creation of the logging file.

    Returns
    -------
    dict
        The configuration dictionary with the list of FICL objects set.
    """
    pre_logger.info("Configuring FICLs for download.")

    # Set FICLO settings from CLI if passed, i.e. override config file with CLI
    for ficlo_key in [
        "fields",
        "image_versions",
        "filters",
    ]:
        if cli_settings[ficlo_key] is not None:
            config_dict[ficlo_key] = cli_settings[ficlo_key]

    # Iterate over each FICL
    config_dict["objects"] = []
    config_dict["ficls"] = []
    for field in config_dict["fields"]:
        for image_version in config_dict["image_versions"]:
            for filter in config_dict["filters"]:
                ## Create FICL object and add to config dict FICL list
                ficl = FICL(
                    field=field,
                    image_version=image_version,
                    catalog_version=DEFAULT_CATVER,
                    filter=filter,
                    objects=[-1],
                    pixscale=[-1, -1],
                )
                pre_logger.info(f"Adding FICL {ficl}.")
                config_dict["ficls"].append(ficl)

    # Return config dict with FICL objects set
    return config_dict


def set_ficls(
    config_dict: dict,
    cli_settings: dict,
    pre_logger: logging.Logger,
    object_first: int | None,
    object_last: int | None,
    batch_n_process: int,
    batch_process_id: int,
) -> dict:
    """Set the list of FICL objects for the configuration object.

    Note these are the valid FICLs for the entire run.

    Parameters
    ----------
    config_dict : dict
        Dictionary representing configuration settings.
    cli_settings : dict
        Settings passed from the CLI call.
    pre_logger : logging.Logger
        Logging object prior to the creation of the logging file.
    object_first : int | None
        First object ID in a range to be run over.
    object_last : int | None
        Last object ID in a range to be run over.
    batch_n_process : int
        Number of processes in this batch.
    batch_process_id : int
        ID of this process in this batch, starting from 0.

    Returns
    -------
    dict
        The configuration dictionary with the list of FICL objects set.
    """
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

    # Set batch mode settings - first and last objects, n, id
    config_dict = set_batch_settings(
        config_dict=config_dict,
        object_first=object_first,
        object_last=object_last,
        batch_n_process=batch_n_process,
        batch_process_id=batch_process_id,
        pre_logger=pre_logger,
    )

    # Initialize list of FICLs as empty list
    config_dict["ficls"] = []

    # Iterate over fields
    ## Iterate over all input root subdirectories if user does not specify fields
    if ficl_is_unset(config_dict=config_dict, key="fields"):
        fields = paths.get_directories(path=config_dict["input_root"])

        ## Remove PSFs directory, as it is also an input root subdirectory
        for field in fields:
            if field.name == "psfs":
                fields.remove(field)
                break
    ## Iterate over all specified fields if user specifies
    else:
        fields = [config_dict["input_root"] / field for field in config_dict["fields"]]

    for field in fields:
        pre_logger.info(f"Checking input for field {field.name}.")

        # Iterate over image versions
        ## Iterate over all field subdirectories if user does not specify
        ## image versions
        if ficl_is_unset(config_dict=config_dict, key="image_versions"):
            image_versions = paths.get_directories(path=field)
        ## Iterate over all specified image versions if user specifies
        else:
            image_versions = [
                field / image_version for image_version in config_dict["image_versions"]
            ]

        for image_version in image_versions:
            pre_logger.info(f"Checking input for image version {image_version.name}.")

            # Iterate over catalog versions
            ## Iterate over all field subdirectories if user does not specify
            ## catalog versions
            if ficl_is_unset(config_dict=config_dict, key="catalog_versions"):
                catalog_versions = [DEFAULT_CATVER]
            ## Iterate over all specified catalog versions if user specifies
            else:
                catalog_versions = config_dict["catalog_versions"]

            for catalog_version in catalog_versions:
                pre_logger.info(
                    f"Checking input for catalog version {catalog_version}."
                )
                fic_names = [field.name, image_version.name, catalog_version]

                ## Skip FIC if input catalog or input segmap missing
                if ficl_is_missing_input(
                    config_dict=config_dict,
                    field=field.name,
                    image_version=image_version.name,
                    filter="",
                    fic=True,
                ):
                    pre_logger.warning(
                        f"FIC {'_'.join(fic_names)} missing input files, skipping."
                    )
                    continue

                ## Get list of object IDs for FIC
                pre_logger.info(f"Resolving object IDs for FIC {'_'.join(fic_names)}.")
                objects = get_objects(
                    config_dict=config_dict,
                    field=field.name,
                    image_version=image_version.name,
                    catalog_version=catalog_version,
                )

                # Iterate over filters
                ## Iterate over all image version subdirectories if user does
                ## not specify filters
                if ficl_is_unset(config_dict=config_dict, key="filters"):
                    filters = paths.get_directories(path=image_version)
                ## Iterate over all specified filters if user specifies
                else:
                    filters = [
                        image_version / filter for filter in config_dict["filters"]
                    ]

                for filter in filters:
                    ## Resolve filter name
                    cleaned_filter = clean_filter(
                        input_root=config_dict["input_root"],
                        field=field.name,
                        image_version=image_version.name,
                        filter=filter.name,
                        pre_logger=pre_logger,
                    )

                    ## Skip filter if valid filter name not found
                    if cleaned_filter is None:
                        continue

                    ## Skip FICL if any input files missing
                    if ficl_is_missing_input(
                        config_dict=config_dict,
                        field=field.name,
                        image_version=image_version.name,
                        filter=cleaned_filter,
                    ):
                        ficl_names = [
                            field.name,
                            image_version.name,
                            catalog_version,
                            cleaned_filter,
                        ]
                        pre_logger.warning(
                            f"FICL {'_'.join(ficl_names)}"
                            + "missing input files, skipping."
                        )
                        continue

                    ## Get pixel scale from science file
                    path_science = paths.get_path(
                        "science",
                        input_root=config_dict["input_root"],
                        field=field.name,
                        image_version=image_version.name,
                        filter=cleaned_filter,
                    )
                    pixscale = science.get_pixscale(path_science)

                    ## Create FICL object and add to config dict FICL list
                    ficl = FICL(
                        field=field.name,
                        image_version=image_version.name,
                        catalog_version=catalog_version,
                        filter=cleaned_filter,
                        objects=objects,
                        pixscale=pixscale,
                    )
                    pre_logger.info(f"Adding FICL {ficl}.")
                    config_dict["ficls"].append(ficl)

    # Return config dict with FICL objects set
    return config_dict


def set_run_settings(config_dict: dict) -> dict:
    """Set the run settings in the configuration dictionary.

    Parameters
    ----------
    config_dict : dict
        Dictionary representing configuration settings.

    Returns
    -------
    dict
        The configuration dictionary with the run settings configured.
    """
    # Set datetime to current local time
    config_dict["datetime"] = dt.now()

    # Set run number to 1 by default
    run_number = 1

    ## Increase run number if other processes in same batch are found
    while paths.get_path(
        "run",
        run_root=config_dict["run_root"],
        field=config_dict["ficls"][0].field,
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
    pre_log = tempfile.NamedTemporaryFile()
    base_logger = logs.create_logger(filename=pre_log.name)
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
    )

    # Set all FICL objects (list[FICL])
    if download:
        config_dict = set_ficls_download_mode(
            config_dict=config_dict, cli_settings=locals(), pre_logger=pre_logger
        )
    else:
        config_dict = set_ficls(
            config_dict=config_dict,
            cli_settings=locals(),
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

    # Remove pre-program loggers
    base_logger.handlers.clear()
    pre_logger.handlers.clear()
    pre_log.close()

    # Create program logger and remove pre-program logger
    logger, main_logger = get_loggers(morphfits_config=morphfits_config)
    main_logger.info("Starting MorphFITS.")

    # Display if batch mode
    if batch_n_process > 1:
        main_logger.info("Running in batch mode.")
        main_logger.info(f"Batch process: {batch_process_id} / {batch_n_process-1}")

    # Return configuration object
    return morphfits_config
