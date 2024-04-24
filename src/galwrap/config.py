"""Configure and setup a program execution of the GalWrap package.
"""

# Imports


from typing import Any
from pathlib import Path

from pydantic import BaseModel
import yaml

from . import CONFIG_ROOT, PATH_NAMES


# Classes


class GalWrapConfig(BaseModel):
    """Configuration model for a program execution of GalWrap.

    Parameters
    ----------
    BaseModel : class
        Base pydantic model class to enforce type validation upon creation.

    Attributes
    ----------
    input_root : Path
        Path to root of input directories.
    output_root : Path
        Path to root of output directories.
    target : str
        Designation of target.
    field : str
        Field to analyze.
    image_version : str
        Version of imaging to use.
    catalog_version : int
        Version of cataloguing to use.
    morphology_version : str
        Version of morphology to use.
    filter : str
        Filter to use.
    pixname : str
        TODO describe.
    pixscale : float
        TODO resolve.
    """

    input_root: Path
    output_root: Path
    target: str
    field: str
    image_version: str
    catalog_version: int
    morphology_version: str
    filter: str
    pixname: str
    pixscale: float


## WIP


class FilterConfig(BaseModel):
    filters: list[str]
    pixnames: list[str]  # TODO resolve name
    pixel_scales: list[float]


class FileConfig(BaseModel):
    # Many just used once
    filtinfo: Path  # only used by filts pixname pixscale
    depth: Path  # only used by magdepth filt totflag
    segmap: Path  # make_all_cutouts
    photcat: Path  # cat_table
    maskname: Path
    sciname: Path


class VarConfig(BaseModel):
    pass


# Functions


def create_config() -> GalWrapConfig:
    """Create a configuration object from a user-created configuration file,
    using default values where unspecified.

    Returns
    -------
    GalWrapConfig
        A configuration object for this program execution.
    """
    # Load default and config dicts from config files
    default_config_dict = yaml.safe_load(open(CONFIG_ROOT / "default.yaml"))
    # TODO determine how this is read from user
    config_dict = yaml.safe_load(open(CONFIG_ROOT / "config.yaml"))

    # Set any required parameters not set by user to default
    ## Iterate over keys in default config and add to config if not set
    for config_key in default_config_dict:
        if config_key not in config_dict:
            config_dict[config_key] = config_dict[config_key]

    # Create and return GalWrapConfig
    return GalWrapConfig(**config_dict)


def resolve_path_name(name: str) -> str:
    """Get internally-standardized name of path corresponding to specified name.

    Parameters
    ----------
    name: str
        Name of directory or file, e.g. `STAMPDIR` or `stamps`.

    Returns
    -------
    str
        Corresponding standardized path name for internal usage.

    Raises
    ------
    TypeError
        Passed name not a str.
    ValueError
        Passed name unrecognized.

    Notes
    -----
    A path name is resolvable if it is equal to
    1. The standardized path name itself
    2. The standardized path name, un-pluralized
        e.g. `output_stamp` for `output_stamps`
    3. A recognized alternative name

    See Also
    --------
    data/path_names.yaml
        List of recognized alternative path names for each path.
    """
    # Terminate if name is not str
    if not isinstance(name, str):
        raise TypeError(f"Path name {name} must be `str`, not {type(name)}.")

    # Find and return path name among recognized names
    for path_name, recognized_names in (
        PATH_NAMES["directories"] | PATH_NAMES["files"]
    ).items():
        if (
            (name == path_name)
            or (("s" == path_name[-1]) and (name == path_name[:-1]))
            or (name in recognized_names)
        ):
            return path_name

    # Terminate if name not found
    raise ValueError(f"Unrecognized path name '{name}'.")


def get_path(
    name: str,
    galwrap_config: GalWrapConfig,
    galaxy_id: int | None = None,
    filter: str | None = None,
) -> Path:
    """Get path to directory or file from a recognized name.

    Parameters
    ----------
    name : str
        Name of directory or file, e.g. `STAMPDIR` or `stamps`.
    galwrap_config : GalWrapConfig
        Configuration parameters for this program execution.
    galaxy_id : int | None, optional
        ID of target galaxy, by default None.
    filter : str | None, optional
        Filter to be used, by default None.

    Returns
    -------
    Path
        Path to specified directory or file.
    """
    # Resolve path name
    path_name = resolve_path_name(name)

    # Return corresponding path
    match path_name:
        # Directories
        ## Input
        case "input_photometry":
            return galwrap_config.input_root / "photometry_products"
        case "input_rms":
            return (
                get_path("input_photometry", galwrap_config)
                / galwrap_config.target
                / galwrap_config.field
                / galwrap_config.image_version
                / "rms_maps"
            )
        case "input_imaging":
            return galwrap_config.input_root / "imaging_products"
        case "input_data_dir":
            return (
                get_path("input_imaging", galwrap_config)
                / galwrap_config.target
                / galwrap_config.field
            )
        case "input_images":
            return (
                get_path("input_data_dir", galwrap_config)
                / "images"
                / "bcgs_out"
                / galwrap_config.image_version
            )
        case "input_psfs":
            return (
                get_path("input_data_dir", galwrap_config)
                / "psfs"
                / galwrap_config.image_version
                / f"{galwrap_config.image_version}.{galwrap_config.catalog_version}"
            )
        case "input_catalogs":
            return (
                get_path("input_data_dir", galwrap_config)
                / "catalogs"
                / galwrap_config.image_version
                / f"{galwrap_config.image_version}.{galwrap_config.catalog_version}"
            )

        ## Output
        case "output_filter_info":
            return galwrap_config.output_root / "MORPHDIR" / "FILTINFO" / "FILTALL"
        case "output_ofic":
            return (
                galwrap_config.output_root
                / "MORPHDIR"
                / "FITDIR"
                / f"{galwrap_config.target}{galwrap_config.field}"
                / f"{galwrap_config.image_version}.{galwrap_config.catalog_version}"
            )
        case "output_stamps":
            return get_path("output_ofic", galwrap_config) / "STAMPDIR"
        case "output_masks":
            return get_path("output_ofic", galwrap_config) / "MASKDIR"
        case "output_rms":
            return get_path("output_ofic", galwrap_config) / "RMSDIR"
        case "output_feedfiles":
            return get_path("output_ofic", galwrap_config) / "FEEDDIR"
        case "output_segmaps":
            return get_path("output_ofic", galwrap_config) / "SEGMAPDIR"
        case "output_galfit":
            return get_path("output_ofic", galwrap_config) / "GALFITOUTDIR"
        case "output_psfs":
            return get_path("output_ofic", galwrap_config) / "PSF"
        case "output_visualizations":
            return get_path("output_ofic", galwrap_config) / "VISDIR"
        case "output_notebooks":
            return get_path("output_ofic", galwrap_config) / "NOTEBOOKS"

        # Files
        case "file_filtinfo":
            return (
                get_path("output_ofic", galwrap_config)
                / f"{galwrap_config.target}{galwrap_config.field}"
                + f"_{galwrap_config.image_version}_FILTERINFO.dat"
            )
        case "file_depth":
            return (
                get_path("output_ofic", galwrap_config)
                / f"{galwrap_config.target}{galwrap_config.field}"
                + f"_{galwrap_config.image_version}_DEPTH.txt"
            )
        case "file_segmap":
            return (
                get_path("input_catalogs", galwrap_config)
                / f"{galwrap_config.target}{galwrap_config.field}"
                + f"_photutils_segmap_{galwrap_config.image_version}"
                + f".{galwrap_config.catalog_version}.fits"
            )
        case "file_photcat":
            return (
                get_path("input_catalogs", galwrap_config)
                / f"{galwrap_config.target}{galwrap_config.field}"
                + f"_photutils_cat_{galwrap_config.image_version}"
                + f".{galwrap_config.catalog_version}.fits"
            )
        case "file_maskname":
            return (
                get_path("output_masks", galwrap_config)
                / f"{galaxy_id}_{galwrap_config.target}{galwrap_config.field}_mask.fits"
            )
        case "file_sciname":
            return (
                get_path("output_stamps", galwrap_config)
                / filter
                / f"{galaxy_id}_{galwrap_config.target}{galwrap_config.field}"
                + f"-{filter}_sci.fits"
            )


def setup_directories(
    galwrap_config: GalWrapConfig, filters: dict[str, Any] | None = None
):
    """Validate and create directory structure for program execution.

    Parameters
    ----------
    galwrap_config : GalWrapConfig
        Configuration parameters for this program execution.
    filters : dict[str, Any] | None, optional
        Dict of filters for this program execution, by default None.

    Raises
    ------
    FileNotFoundError
        Input directory not found.
    """
    # Iterate over each expected directory
    for path_name in PATH_NAMES["directories"]:
        # Terminate if input directory does not exist
        if "input" in path_name:
            directory_path = get_path(path_name, galwrap_config)
            if not directory_path.exists():
                raise FileNotFoundError(
                    f"Input directory {path_name} expected, but not found."
                )

        # Create output directory if it does not exist
        elif "output" in path_name:
            directory_path = get_path(path_name, galwrap_config)
            if not directory_path.exists():
                directory_path.mkdir(parents=True, exist_ok=True)

    # Create filtered output subdirectories
    filtered_output_dirs = [
        "output_stamps",
        "output_rms",
        "output_feedfiles",
        "output_galfit",
    ]
    if filters is not None:
        for filtered_output_dir in filtered_output_dirs:
            for filter in filters:
                (get_path(filtered_output_dir, galwrap_config) / filter).mkdir(
                    parents=True, exist_ok=True
                )
