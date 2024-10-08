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
        Integer IDs of galaxies or cluster targets in catalog, e.g. `[1003,
        6371]`.
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
    catalog_versions : list[str], optional
        List of catalog versions over which to fit, by default only the v7.2 DJA
        catalog.
    objects : list[int], optional
        List of object IDs within the catalog over which to fit, by default
        empty (all items in catalog).
    morphfits_root : Path, optional
        Path to root directory containing all of input, product, and output
        directories, by default the repository root.
    wrappers : list[str], optional
        List of morphology fitting algorithms to run, by default only GALFIT.
    galfit_path : Path
        Path to GALFIT binary, by default None.
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
    catalog_versions: list[str] = ["dja-v7.2"]
    objects: list[int] = []
    morphfits_root: Path = ROOT
    wrappers: list[str] = ["galfit"]
    galfit_path: Path = None

    def get_FICLs(
        self,
        pre_input: bool = False,
    ) -> Generator[FICL, None, None]:
        """Generate all FICL permutations for this configuration object, and
        return those with the necessary input files.

        Parameters
        ----------
        pre_input : bool, optional
            Skip file checks if this function is called prior to download.

        Yields
        ------
        FICL
            FICL permutation with existing input files.
        """
        # Iterate over each FICL in config object
        for field, image_version, catalog_version, filter in itertools.product(
            self.fields, self.image_versions, self.catalog_versions, self.filters
        ):
            # Create FICL for iteration, with every object, and set pixscales
            if pre_input:
                pixscale = [0.04, 0.04]
            else:
                science_path = paths.get_path(
                    "science",
                    input_root=self.input_root,
                    field=field,
                    image_version=image_version,
                    filter=filter,
                )
                pixscale = science.get_pixscale(science_path)
            ficl = FICL(
                field=field,
                image_version=image_version,
                catalog_version=catalog_version,
                filter=filter,
                objects=self.objects,
                pixscale=pixscale,
            )

            # Skip FICL if any input missing
            if not pre_input:
                missing_input = False
                for required_input in [
                    "original_psf",
                    "segmap",
                    "catalog",
                    "exposure",
                    "science",
                    "weights",
                ]:
                    if not paths.get_path(
                        name=required_input,
                        input_root=self.input_root,
                        field=field,
                        image_version=image_version,
                        catalog_version=catalog_version,
                        filter=filter,
                    ).exists():
                        logger.warning(
                            f"FICL {ficl} missing {required_input} file, skipping."
                        )
                        missing_input = True
                        break
                if missing_input:
                    continue

            yield ficl

    def setup_paths(self, display_progress: bool = False, pre_input: bool = False):
        """Create product and output directories for each FICLO of this
        configuration object.

        Parameters
        ----------
        display_progress : bool, optional
            Display setup progress via tqdm, by default False.
        pre_input : bool, optional
            Skip file checks if this function is called prior to download.
        """
        print("Creating product and output directories where missing.")

        # Create run directory for both download and fitting runs
        paths.get_path(
            name="run",
            run_root=self.run_root,
            datetime=self.datetime,
            run_number=self.run_number,
        ).mkdir(parents=True, exist_ok=True)

        # Only create input and run directories for download run
        if pre_input:
            # Iterate over each possible FICL from configurations
            for ficl in self.get_FICLs(pre_input=pre_input):
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

        else:
            # Iterate over each possible FICLO from configurations
            for ficl in self.get_FICLs(pre_input=pre_input):
                # Iterate over each object in FICL
                for object in (
                    tqdm(ficl.objects, unit="dir", leave=False)
                    if display_progress
                    else ficl.objects
                ):
                    # Make leaf FICLO directories
                    for path_name in [
                        "ficlo_products",
                        "logs",
                        "models",
                        "plots",
                    ]:
                        # Create directory if it does not exist
                        paths.get_path(
                            name=path_name,
                            morphfits_root=self.morphfits_root,
                            output_root=self.output_root,
                            product_root=self.product_root,
                            run_root=self.run_root,
                            field=ficl.field,
                            image_version=ficl.image_version,
                            catalog_version=ficl.catalog_version,
                            filter=ficl.filter,
                            object=object,
                            datetime=self.datetime,
                            run_number=self.run_number,
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
        for ficl in self.get_FICLs():
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
                                name="ficlo_products",
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
                        name=f"{wrapper}_model",
                        output_root=self.output_root,
                        field=ficl.field,
                        image_version=ficl.image_version,
                        catalog_version=ficl.catalog_version,
                        filter=ficl.filter,
                        object=object,
                    ).exists():
                        shutil.rmtree(
                            paths.get_path(
                                name="ficlo_output",
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

        # Write config to file
        yaml.safe_dump(
            write_config,
            open(
                paths.get_path(
                    "config",
                    run_root=self.run_root,
                    datetime=self.datetime,
                    run_number=self.run_number,
                ),
                mode="w",
            ),
        )


# Functions


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
    print("Loading configuration.")

    # Load config file values
    config_dict = {} if config_path is None else yaml.safe_load(open(config_path))

    ## Cast and resolve paths
    for root_key in [
        "morphfits_root",
        "input_root",
        "output_root",
        "product_root",
        "run_root",
        "galfit_path",
    ]:
        if root_key in config_dict:
            config_dict[root_key] = paths.get_path_obj(config_dict[root_key])
        if locals()[root_key] is not None:
            config_dict[root_key] = paths.get_path_obj(locals()[root_key])

    # Set any parameters passed through CLI call
    ## Paths
    ### Terminate if input root not found
    if "input_root" not in config_dict:
        raise FileNotFoundError("Input root not passed, terminating.")
    else:
        input_root_dir = Path(config_dict["input_root"]).resolve()
        if not input_root_dir.exists():
            #### Create input root and other directories if in download mode
            if download:
                input_root_dir.mkdir(parents=True, exist_ok=True)
            else:
                raise FileNotFoundError(
                    f"Input root {config_dict['input_root']} not found."
                )

    ### Set root directory as parent of input root if not found
    if "morphfits_root" not in config_dict:
        config_dict["morphfits_root"] = config_dict["input_root"].parent

    ### Set product, output, and run directories from root directory
    for root in ["output_root", "product_root", "run_root"]:
        if root not in config_dict:
            config_dict[root] = paths.get_path(
                root, morphfits_root=config_dict["morphfits_root"]
            )

    ## FICLOs - overwrites configuration file
    if fields is not None:
        config_dict["fields"] = fields
    if image_versions is not None:
        config_dict["image_versions"] = image_versions
    if catalog_versions is not None:
        config_dict["catalog_versions"] = catalog_versions
    if filters is not None:
        config_dict["filters"] = filters
    if objects is not None:
        config_dict["objects"] = objects
    if wrappers is not None:
        config_dict["wrappers"] = wrappers

    # If parameters are still unset, assume program execution over all
    # discovered values in input directory
    if download:
        config_dict["objects"] = []
    for parameter in [
        "field",
        "image_version",
        # "catalog_version",
        "filter",
        "object",
    ]:
        if (parameter + "s" not in config_dict) or (
            config_dict[parameter + "s"] is None
        ):
            if download:
                raise ValueError(f"Parameter '{parameter}' must be provided.")
            else:
                config_dict[parameter + "s"] = paths.find_parameter_from_input(
                    parameter_name=parameter, input_root=config_dict["input_root"]
                )

    # Set batch mode parameters
    if "first_object" in config_dict:
        config_dict["object_first"] = config_dict["first_object"]
    else:
        config_dict["object_first"] = object_first
    if "last_object" in config_dict:
        config_dict["object_last"] = config_dict["last_object"]
    else:
        config_dict["object_last"] = object_last
    config_dict["batch_n_process"] = batch_n_process
    config_dict["batch_process_id"] = batch_process_id

    # If in batch mode or object range defined, reset objects accordingly
    if (
        (batch_n_process > 1)
        or (config_dict["object_first"] is not None)
        or (config_dict["object_last"] is not None)
    ):
        ## Terminate if more than one catalog specified for batch mode
        if (batch_n_process) and (
            (len(config_dict["fields"]) > 1) or (len(config_dict["image_versions"]) > 1)
        ):
            raise ValueError(
                "Cannot set ranges for multiple catalog versions, "
                + "as their ID ranges differ."
            )

        ## Get total object ID range from catalog
        catalog_range = []
        catalog_path = paths.get_path(
            "catalog",
            input_root=config_dict["input_root"],
            field=config_dict["fields"][0],
            image_version=config_dict["image_versions"][0],
        )
        catalog = Table.read(catalog_path)
        for id in catalog["id"]:
            catalog_range.append(int(id))
        del catalog
        gc.collect()

        ## Set objects to all if range not specified
        if (config_dict["object_first"] is None) and (
            config_dict["object_last"] is None
        ):
            user_range = catalog_range
        ## Get specified object ID range from user
        elif config_dict["object_first"] is None:
            user_range = list(range(catalog_range[0], config_dict["object_last"]))
        elif config_dict["object_last"] is None:
            user_range = list(range(config_dict["object_first"], catalog_range[-1]))
        else:
            user_range = list(
                range(config_dict["object_first"], config_dict["object_last"])
            )

        ## Get batch object ID range from user parameters
        start_index, stop_index = misc.get_unique_batch_limits(
            process_id=batch_process_id,
            n_process=batch_n_process,
            n_items=len(user_range),
        )
        batch_range = user_range[start_index:stop_index]

        ## Remove objects out of range
        while (len(batch_range) > 0) and (batch_range[0] < catalog_range[0]):
            batch_range.pop(0)
        while (len(batch_range) > 0) and (batch_range[-1] > catalog_range[-1]):
            batch_range.pop(-1)

        ## Set object ID range for this batch run
        config_dict["objects"] = batch_range

    # Set start datetime and run number
    config_dict["datetime"] = dt.now()
    run_number = 1
    while paths.get_path(
        "run",
        run_root=config_dict["run_root"],
        datetime=config_dict["datetime"],
        run_number=run_number,
    ).exists():
        run_number += 1
    config_dict["run_number"] = run_number

    # Create configuration object from dict
    morphfits_config = MorphFITSConfig(**config_dict)

    # Terminate if fitter is GALFIT and binary file is not linked
    if ("galfit" in morphfits_config.wrappers) and (
        (morphfits_config.galfit_path is None)
        or (not morphfits_config.galfit_path.exists())
    ):
        raise FileNotFoundError("GALFIT binary file not found or linked.")

    # Setup directories where missing
    morphfits_config.setup_paths(display_progress=True, pre_input=download)

    # Create logger
    logs.create_logger(
        filename=paths.get_path(
            "morphfits_log",
            run_root=morphfits_config.run_root,
            datetime=morphfits_config.datetime,
            run_number=morphfits_config.run_number,
        )
    )
    global logger
    logger = logging.getLogger("CONFIG")
    main_logger = logging.getLogger("MORPHFITS")
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
