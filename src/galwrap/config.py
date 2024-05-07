"""Configure and setup a program execution of the GalWrap package.
"""

# Imports


from typing import Optional, Generator
from pathlib import Path
import itertools

from pydantic import BaseModel
import yaml

from . import CONFIG_ROOT


# Classes


class OFIC(BaseModel):
    """Configuration model for a program execution of GalFit.

    Parameters
    ----------
    BaseModel : class
        Base pydantic model class to enforce type validation upon creation.

    Attributes
    ----------
    object : str
        Designation of target.
    field : str
        Field to analyze.
    image_version : str
        Version of imaging to use.
    catalog_version : int
        Version of cataloguing to use.
    morphology_version : str | None, optional
        Version of morphology to use, by default None.
    filter : str | None, optional
        Filter to use, by default None.
    pixscale : float | None, optional
        Cutout pixel scale with which to determine image size, by default None.
    """

    object: str
    field: str
    image_version: str
    catalog_version: int
    morphology_version: Optional[str]
    filter: Optional[str]
    pixscale: Optional[float]


class GalWrapConfig(BaseModel):
    """Configuration model for a program execution of GalWrap.

    Parameters
    ----------
    BaseModel : class
        Base pydantic model class to enforce type validation upon creation.

    Attributes
    ----------
    photometry_root : Path
        Path to root of input photometry products.
    imaging_root : Path
        Path to root of input imaging products.
    output_root : Path
        Path to root of directory to which to write output products.
    objects : list[str] | None, optional
        List of objects to execute this program over, by default None.
    fields : list[str] | None, optional
        List of fields to execute this program over.
    image_versions : list[str] | None, optional
        List of image versions to execute this program over, by default None.
    catalog_versions : list[int] | None, optional
        List of catalog versions to execute this program over, by default None.
    morphology_versions : list[str] | None, optional
        List of morphology versions to execute this program over, by default
        None.
    filters : list[str] | None, optional
        List of filters to execute this program over, by default None.
    pixscales : list[float] | None, optional
        List of pixscales to execute this program over, by default None.
    """

    photometry_root: Path
    imaging_root: Path
    output_root: Path
    objects: Optional[list[str]] = None
    fields: Optional[list[str]] = None
    image_versions: Optional[list[str]] = None
    catalog_versions: Optional[list[int]] = None
    morphology_versions: Optional[list[str]] = None
    filters: Optional[list[str]] = None
    pixscales: Optional[list[float]] = None

    def get_ofics(
        self,
        objects: list[str] | None = None,
        fields: list[str] | None = None,
        image_versions: list[str] | None = None,
        catalog_versions: list[int] | None = None,
        morphology_versions: list[str] | None = None,
        filters: list[str] | None = None,
        pixscales: list[float] | None = None,
    ) -> Generator[OFIC, None, None]:
        """Generate OFIC permutations from input configurations.

        Parameters
        ----------
        objects : list[str] | None, optional
            List of objects over which to execute this program, by default None
            (all of them).
        fields : list[str] | None, optional
            List of fields over which to execute this program, by default None
            (all of them).
        image_versions : list[str] | None, optional
            List of image versions over which to execute this program, by
            default None (all of them).
        catalog_versions : list[str] | None, optional
            List of catalog versions over which to execute this program, by
            default None (all of them).
        morphology_versions : list[str] | None, optional
            List of morphology versions over which to execute this program, by
            default None (all of them).
        filters : list[str] | None, optional
            List of filters over which to execute this program, by default None
            (all of them).
        pixscales : list[float] | None, optional
            List of pixel scales over which to execute this program, by default
            None (all of them).

        Yields
        ------
        OFIC
            OFIC permutation from specified objects, fields, image, catalog, and
            morphology versions, filters, and pixel scales.
        """
        # Terminate if necessary input not specified
        if (
            ((objects is None) and (self.objects is None))
            or ((fields is None) and (self.fields is None))
            or ((image_versions is None) and (self.image_versions is None))
            or ((catalog_versions is None) and (self.catalog_versions is None))
            or ((morphology_versions is None) and (self.morphology_versions is None))
            or ((filters is None) and (self.filters is None))
            or ((pixscales is None) and (self.pixscales is None))
        ):
            raise ValueError("Necessary input for OFIC configuration not specified.")

        # Generate OFIC permutations from all specified configurations
        for (
            object,
            field,
            image_version,
            catalog_version,
            morphology_version,
            filter,
            pixscale,
        ) in itertools.product(
            self.objects if objects is None else objects,
            self.fields if fields is None else fields,
            self.image_versions if image_versions is None else image_versions,
            self.catalog_versions if catalog_versions is None else catalog_versions,
            (
                self.morphology_versions
                if morphology_versions is None
                else morphology_versions
            ),
            self.filters if filters is None else filters,
            self.pixscales if pixscales is None else pixscales,
        ):
            yield OFIC(
                object=object,
                field=field,
                image_version=image_version,
                catalog_version=catalog_version,
                morphology_version=morphology_version,
                filter=filter,
                pixscale=pixscale,
            )


# Functions


def create_config(
    config_path: str | Path = CONFIG_ROOT / "config.yaml",
) -> GalWrapConfig:
    """Create a configuration object from a user-created configuration file,
    using default values where unspecified.

    Parameters
    ----------
    config_path : str | Path, optional
        Path to user config yaml file, by default CONFIG_ROOT / "config.yaml".

    Returns
    -------
    GalWrapConfig
        A configuration object for this program execution.
    """
    # Load default and config dicts from config files
    default_config_dict = yaml.safe_load(open(CONFIG_ROOT / "default.yaml"))
    # TODO determine how this is read from user
    config_dict = yaml.safe_load(open(config_path))

    # Set any required parameters not set by user to default
    ## Iterate over keys in default config and add to config if not set
    for config_key in default_config_dict:
        if config_key not in config_dict:
            config_dict[config_key] = default_config_dict[config_key]

    # Create GalWrapConfig from dict
    galwrap_config = GalWrapConfig(**config_dict)

    # Set filter info lists
    # TODO might need to be grouped with other variables
    # filter_info = ascii.read(get_path("file_filter_info", ))
    # galwrap_config.filters = filter_info["FILTNAME"]
    # galwrap_config.pixscales = filter_info["PIXSCALES"]

    return galwrap_config
