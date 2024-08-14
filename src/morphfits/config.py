"""Configure and setup a program execution of the MorphFITS program.
"""

# Imports


import logging
import itertools
import shutil
from pathlib import Path
from typing import Generator, Annotated
from datetime import datetime as dt

from numpy import sqrt
from astropy.io import fits
from astropy.table import Table
import yaml
from pydantic import BaseModel, StringConstraints
from tqdm import tqdm

from . import paths, ROOT
from .utils import science


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
            science_path = paths.get_path(
                "science",
                input_root=self.input_root,
                field=field,
                image_version=image_version,
                filter=filter,
            )
            if pre_input:
                pixscale = [0.04, 0.04]
            else:
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
        logger.info("Creating product and output directories where missing.")
        print("Creating product and output directories where missing.")

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
                    "run",
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
    logger.info(f"Loading configuration.")
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
        logger.error("Input root not passed, terminating.")
        raise FileNotFoundError("Input root not passed, terminating.")
    else:
        input_root_dir = Path(config_dict["input_root"]).resolve()
        if not input_root_dir.exists():
            #### Create input root and other directories if in download mode
            if download:
                input_root_dir.mkdir(parents=True, exist_ok=True)
            else:
                logger.error(f"Input root {config_dict['input_root']} not found.")
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
            config_dict[parameter + "s"] = paths.find_parameter_from_input(
                parameter_name=parameter, input_root=config_dict["input_root"]
            )

    # Get object count from catalog
    catalog_ranges = {}
    for field in config_dict["fields"]:
        catalog_ranges[field] = {}
        for image_version in config_dict["image_versions"]:
            catalog_ranges[field][image_version] = []
            catalog_path = paths.get_path(
                "catalog",
                input_root=config_dict["input_root"],
                field=field,
                image_version=image_version,
            )
            catalog = Table.read(catalog_path)
            for id in catalog["id"]:
                catalog_ranges[field][image_version].append(int(id))

    # If any batch mode parameters are set, rewrite objects accordingly
    if (object_first is not None) or (object_last is not None):
        pass

    # Remove objects out of range

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
        logger.error(
            "GALFIT chosen as fitter but binary file "
            + "not found or linked, terminating."
        )
        raise FileNotFoundError("GALFIT binary file not found or linked.")

    # Setup directories where missing
    morphfits_config.setup_paths(display_progress=display_progress, pre_input=download)
    if not download:
        paths.get_path(
            name="run",
            run_root=morphfits_config.run_root,
            datetime=morphfits_config.datetime,
            run_number=morphfits_config.run_number,
        ).mkdir(parents=True, exist_ok=True)

    # Return configuration object
    return morphfits_config
