"""Configure and setup a program execution of the MorphFITS program.
"""

# Imports


import logging
import itertools
import re
from pathlib import Path
from typing import Generator, Annotated
from datetime import datetime as dt

import yaml
from pydantic import BaseModel, StringConstraints
from tqdm import tqdm

from . import paths, ROOT
from .utils import logs


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
    filter of a JWST science observation.

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
    pixscale : str, optional
        Pixel scale, in arcseconds per pixel, by default `0.04`, corresponding
        to "40mas".

    Notes
    -----
    All strings are converted to lowercase upon validation.
    """

    field: Annotated[str, StringConstraints(to_lower=True)]
    image_version: Annotated[str, StringConstraints(to_lower=True)]
    catalog_version: Annotated[str, StringConstraints(to_lower=True)]
    filter: Annotated[str, StringConstraints(to_lower=True)]
    objects: list[int]
    pixscale: float = 0.04

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
    datetime : datetime
        Datetime at start of program run.
    input_root : Path
        Path to root input directory.
    product_root : Path
        Path to root products directory.
    output_root : Path
        Path to root output directory.
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
    pixscales : list[float], optional
        List of pixel scales over which to execute GALFIT, by default only `0.04`,
        corresponding to "40mas".
    wrappers : list[str], optional
        List of morphology fitting algorithms to run, by default only GALFIT.
    """

    datetime: dt
    input_root: Path
    product_root: Path
    output_root: Path
    fields: list[str]
    image_versions: list[str]
    filters: list[str]
    catalog_versions: list[str] = ["dja-v7.2"]
    objects: list[int] = []
    morphfits_root: Path = ROOT
    pixscales: list[float] = [0.04]
    wrappers: list[str] = ["galfit"]

    def get_FICLs(self) -> Generator[FICL, None, None]:
        """Generate all FICL permutations for this configuration object, and
        return those with the necessary input files.

        Yields
        ------
        FICL
            FICL permutation with existing input files.
        """
        # Iterate over each FICL in config object
        for field, image_version, catalog_version, filter in itertools.product(
            self.fields, self.image_versions, self.catalog_versions, self.filters
        ):
            # Create FICL for iteration, with every object
            ficl = FICL(
                field=field,
                image_version=image_version,
                catalog_version=catalog_version,
                filter=filter,
                objects=self.objects,
            )

            # Skip FICL if any input missing
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
                    logger.debug(
                        f"FICL {ficl} missing {required_input} file, skipping."
                    )
                    missing_input = True
                    break
            if missing_input:
                continue

            yield ficl

    def setup_paths(self, display_progress: bool = False):
        """Create product and output directories for each FICLO of this
        configuration object.

        Parameters
        ----------
        display_progress : bool, optional
            Display setup progress via tqdm, by default False.
        """
        logger.info("Creating product and output directories where missing.")

        # Iterate over each possible FICLO from configurations
        for ficl in self.get_FICLs():
            # Iterate over each object in FICL
            for object in (
                tqdm(ficl.objects, unit="dir", leave=False)
                if display_progress
                else ficl.objects
            ):
                # Make leaf FICLO directories
                for path_name in [
                    "ficlo_products",
                    "run",
                    "logs",
                    "models",
                    "visualizations",
                ]:
                    # Create directory if it does not exist
                    paths.get_path(
                        name=path_name,
                        morphfits_root=self.morphfits_root,
                        product_root=self.product_root,
                        output_root=self.output_root,
                        field=ficl.field,
                        image_version=ficl.image_version,
                        catalog_version=ficl.catalog_version,
                        filter=ficl.filter,
                        object=object,
                        datetime=self.datetime,
                    ).mkdir(parents=True, exist_ok=True)

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
                    "config", output_root=self.output_root, datetime=self.datetime
                ),
                mode="w",
            ),
        )


# Functions


def create_config(
    config_path: str | Path | None = None,
    morphfits_root: str | Path | None = None,
    input_root: str | Path | None = None,
    product_root: str | Path | None = None,
    output_root: str | Path | None = None,
    fields: list[str] | None = None,
    image_versions: list[str] | None = None,
    catalog_versions: list[str] | None = None,
    filters: list[str] | None = None,
    objects: list[int] | None = None,
    wrappers: list[str] | None = None,
    display_progress: bool = False,
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
    product_root : str | Path | None, optional
        Path to root directory of products generated by this program to execute
        GALFIT, e.g. cutouts/stamps, masks, and feedfiles, by default None (not
        passed through CLI).
    output_root : str | Path | None, optional
        Path to root directory of GALFIT output products, e.g. morphology model
        and plots, by default None (not passed through CLI).
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
    display_progress : bool, optional
        Display progress via tqdm, by default False.
    wrappers : list[str], optional
        List of morphology fitting algorithms to run, by default only GALFIT.

    Returns
    -------
    MorphFITSConfig
        A configuration object for this program execution.
    """
    logger.info(f"Loading configuration.")

    # Load config file values
    config_dict = {} if config_path is None else yaml.safe_load(open(config_path))
    ## Cast and resolve paths
    for config_key in ["morphfits_root", "input_root", "product_root", "output_root"]:
        if config_key in config_dict:
            config_dict[config_key] = paths.get_path_obj(config_dict[config_key])

    # Set any parameters passed through CLI call
    ## Paths
    if morphfits_root is not None:
        config_dict["morphfits_root"] = paths.get_path_obj(morphfits_root)
    if input_root is not None:
        config_dict["input_root"] = paths.get_path_obj(input_root)
    if product_root is not None:
        config_dict["product_root"] = paths.get_path_obj(product_root)
    if output_root is not None:
        config_dict["output_root"] = paths.get_path_obj(output_root)

    ### Terminate if input root not found
    if "input_root" not in config_dict:
        logger.error("Input root not passed, terminating.")
        raise FileNotFoundError("Input root not passed, terminating.")
    else:
        if not Path(config_dict["input_root"]).resolve().exists():
            logger.error(f"Input root {config_dict['input_root']} not found.")
            raise FileNotFoundError(
                f"Input root {config_dict['input_root']} not found."
            )
    if "morphfits_root" not in config_dict:
        config_dict["morphfits_root"] = config_dict["input_root"].parent
    if "product_root" not in config_dict:
        config_dict["product_root"] = config_dict["morphfits_root"] / "products"
    if "output_root" not in config_dict:
        config_dict["output_root"] = config_dict["morphfits_root"] / "output"

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
        # "pixscale",
    ]:
        if parameter + "s" not in config_dict:
            config_dict[parameter + "s"] = paths.find_parameter_from_input(
                parameter_name=parameter, input_root=config_dict["input_root"]
            )

    # Remove any suffices from filters, such as 'clear'
    cleaned_filters = []
    for field in config_dict["filters"]:
        cleaned_filters.append(field.split("-")[0])
    config_dict["filters"] = cleaned_filters

    # Set start datetime
    config_dict["datetime"] = dt.now()

    # Create configuration object from dict
    morphfits_config = MorphFITSConfig(**config_dict)

    # Setup directories where missing
    morphfits_config.setup_paths(display_progress=display_progress)

    # Return configuration object
    return morphfits_config
