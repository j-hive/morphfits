"""Resolve, create, and otherwise handle directory and file structure for
GalWrap.
"""

# Imports


import logging
from pathlib import Path

from astropy.table import Table
from tqdm import tqdm

from .setup import GalWrapPath, FICLO, GalWrapConfig, GALWRAP_PATHS
from ..utils import science


# Constants


logger = logging.getLogger("PATHS")
"""Logging object for this module.
"""


# Functions


## Utility


def get_path_obj(path_like: str | Path) -> Path:
    """Get a resolved Path object for a potential string.

    Parameters
    ----------
    path_like : str | Path
        Path which may or may not be of string type.

    Returns
    -------
    Path
        Corresponding Path object.
    """
    return (
        Path(path_like).resolve() if isinstance(path_like, str) else path_like.resolve()
    )


def get_directories(path: Path) -> list[Path]:
    """Get a list of subdirectories under a path.

    Parameters
    ----------
    path : Path
        Path to be walked.

    Returns
    -------
    list[Path]
        List of subdirectories under specified path.

    Raises
    ------
    ValueError
        Specified path not a directory.
    """
    if path.is_dir():
        return [item for item in path.iterdir() if item.is_dir()]
    else:
        raise ValueError(f"Path {path} is not a directory.")


def get_files(path: Path) -> list[Path]:
    """Get a list of files in a directory.

    Parameters
    ----------
    path : Path
        Path to be walked.

    Returns
    -------
    list[Path]
        List of files in specified directory.

    Raises
    ------
    ValueError
        Specified path not a directory.
    """
    if path.is_dir():
        return [item for item in path.iterdir() if item.is_file()]
    else:
        raise ValueError(f"Path {path} is not a directory.")


## Parameter


def get_parameter(
    name: str,
    parameter: int | float | str | Path | list[float] | list[str] | None,
    instance: GalWrapConfig | FICLO | None,
) -> int | float | str | Path | list[float] | list[str]:
    """Get the hierarchically preferred value for a parameter, where parameter
    refers to a program execution's configuration variables, e.g. `field` or
    `filter`, in order of value passed from
        1. Function call parameter
        2. Attribute in object instance
        3. None if neither are passed

    Parameters
    ----------
    name : str
        Name of parameter.
    parameter : int | float | str | Path | list[float] | list[str] | None
        Value passed directly into function.
    instance : GalWrapConfig | FICLO | None
        Values stored in object generated by configuration.

    Returns
    -------
    int | float | str | Path | list[float] | list[str]
        Hierarchically preferred value, in order of directly passed value, then
        that of the instance.

    Notes
    -----
    Internal function used by :func:``get_path``. Should not be exposed to user,
    so `name` is standardized.
    """
    # Validate parameter existence and return corresponding value, in preference
    # of directly passed parameter, then object instance attribute
    ## Multiple parameters
    if name[-1] == "s":
        if parameter is None:
            if (instance is None) or (len(vars(instance)[name]) == 0):
                return None
            else:
                return vars(instance)[name]
        else:
            return parameter
    ## Single parameter
    else:
        if parameter is None:
            if instance is None:
                return None
            else:
                return (
                    get_path_obj(vars(instance)[name])
                    if "root" in name
                    else vars(instance)[name]
                )
        else:
            return get_path_obj(parameter) if "root" in name else parameter


def find_parameter_from_input(
    parameter_name: str, input_root: Path
) -> list[str] | list[int]:
    """Find parameter values from an input directory.

    Parameters
    ----------
    parameter_name : str
        Name of parameter, one of "field", "image_version", or "filter". Other
        parameters are not currently supported for discovery.
    input_root : Path
        Path to root directory of input products.

    Returns
    -------
    list[str] | list[int]
        List of discovered parameter values.

    Raises
    ------
    ValueError
        Unrecognized parameter name.
    """
    discovered = []

    match parameter_name:
        case "field":
            for field_dir in get_directories(input_root):
                if field_dir.name == "psfs":
                    continue
                elif field_dir.name not in discovered:
                    discovered.append(field_dir.name)
        case "image_version":
            for field_dir in get_directories(input_root):
                if field_dir.name == "psfs":
                    continue
                for image_dir in get_directories(field_dir):
                    if image_dir.name not in discovered:
                        discovered.append(image_dir.name)
        case "filter":
            for psf_file in get_files(input_root / "psfs"):
                filter = psf_file.name.split("_")[-1].split(".")[0].lower()
                if filter not in discovered:
                    discovered.append(filter)
            for field_dir in get_directories(input_root):
                if field_dir.name == "psfs":
                    continue
                for image_dir in get_directories(field_dir):
                    for filter_dir in get_directories(image_dir):
                        if filter_dir.name not in discovered:
                            discovered.append(filter_dir.name)
        case "object":
            catalog_paths = []
            for field_dir in get_directories(input_root):
                if field_dir.name == "psfs":
                    continue
                for image_dir in get_directories(field_dir):
                    catalog_paths.append(
                        get_path(
                            "catalog",
                            input_root=input_root,
                            field=field_dir.name,
                            image_version=image_dir.name,
                        )
                    )

            # Read all catalogs for all objects
            for catalog_path in catalog_paths:
                table = Table.read(catalog_path)
                for object in table["id"]:
                    discovered.append(int(object))
        case _:
            raise ValueError(
                f"Parameter {parameter_name} unrecognized for input discovery."
            )

    return discovered


## Resolution


def get_path_name(name: str) -> str:
    """Get internally-standardized name of path corresponding to passed name.

    Parameters
    ----------
    name: str
        Name of directory or file, e.g. `input_images`.

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
    A path name is resolvable if its casefold is equal to
    1. The standardized path name itself
        e.g `input_images`
    2. A recognized alternative name
        e.g. `images` for `input_images`
    3. The standardized path name, separated by spaces rather than underscores
        e.g. `input images` for `input_images`
    4. The standardized path name, space-separated, with a corresponding `dir`
       or `file` suffix
        e.g. `input images dir` for `input_images`
    5. The standardized path name, space-separated, suffixed, un-pluralized
        e.g. `input image dir` for `input_images`

    See Also
    --------
    data/galwrap_path_names.yaml
        List of recognized alternative path names for each path.
    """
    # Terminate if name is not str
    if not isinstance(name, str):
        raise TypeError(f"Path name {name} must be `str`, not {type(name)}.")

    # Set name for case-less comparison
    name = name.casefold()

    # Find and return path name among recognized names
    for path_name, path_item in GALWRAP_PATHS.items():
        # 1. Exact match
        if name == path_name:
            return path_name

        # 2. Alternate name match
        if name in path_item.alts:
            return path_name

        # 3. Space rather than underscore delimiter
        path_name_case_3 = " ".join(path_name.split("_"))
        if ("_" in path_name) and (name == path_name_case_3):
            return path_name

        # 4. Space delimiter and `dir` or `file` suffix
        if name == path_name_case_3 + (" file" if path_item.file else " dir"):
            return path_name

        # 5. Space delimiter, `dir` or `file` suffix, and un-pluralized
        if ("s" == path_name[-1]) and (
            name == path_name_case_3[:-1] + (" file" if path_item.file else " dir")
        ):
            return path_name

    # Terminate if name not found
    raise ValueError(f"Unrecognized path name '{name}'.")


def get_path(
    name: str,
    galwrap_config: GalWrapConfig | None = None,
    galwrap_root: Path | None = None,
    input_root: Path | None = None,
    product_root: Path | None = None,
    output_root: Path | None = None,
    ficlo: FICLO | None = None,
    field: str | None = None,
    image_version: str | None = None,
    catalog_version: str | None = None,
    filter: str | None = None,
    object: int | None = None,
    pixscale: float | None = None,
) -> Path | None:
    # Resolve name and parameters
    path_name = get_path_name(name)

    ## Paths
    galwrap_root = get_parameter("galwrap_root", galwrap_root, galwrap_config)
    input_root = get_parameter("input_root", input_root, galwrap_config)
    product_root = get_parameter("product_root", product_root, galwrap_config)
    output_root = get_parameter("output_root", output_root, galwrap_config)

    ## FICLOs
    field = get_parameter("field", field, ficlo)
    image_version = get_parameter("image_version", image_version, ficlo)
    catalog_version = get_parameter("catalog_version", catalog_version, ficlo)
    filter = get_parameter("filter", filter, ficlo)
    object = get_parameter("object", object, ficlo)
    pixscale = get_parameter("pixscale", pixscale, ficlo)

    # Resolve path for given parameters
    return GALWRAP_PATHS[path_name].resolve(
        galwrap_root=galwrap_root,
        input_root=input_root,
        product_root=product_root,
        output_root=output_root,
        field=field,
        image_version=image_version,
        catalog_version=catalog_version,
        filter=filter,
        object=object,
        pixname=None if pixscale is None else science.get_pixname(pixscale),
    )


## Setup


def setup_galwrap_paths(galwrap_config: GalWrapConfig, display_progress: bool = False):
    """Create GalWrap product and output directories for the FICLOs of a program
    run.

    Parameters
    ----------
    galwrap_config : GalWrapConfig
        Configuration object whose roots to create directories for.
    """
    logger.info("Creating product and output directories where missing.")

    # Iterate over each possible FICLO from configurations
    for ficl in galwrap_config.get_FICLs():
        # Make leaf FICL directories (will also make parents if nonexistent)
        for path_name in ["product_psfs", "output_objects"]:
            # Create directory if it does not exist
            GALWRAP_PATHS[path_name].resolve(
                galwrap_root=galwrap_config.galwrap_root,
                product_root=galwrap_config.product_root,
                output_root=galwrap_config.output_root,
                field=ficl.field,
                image_version=ficl.image_version,
                catalog_version=ficl.catalog_version,
                filter=ficl.filter,
            ).mkdir(parents=True, exist_ok=True)

        # Iterate over each object in FICL
        for object in tqdm(ficl.objects) if display_progress else ficl.objects:
            # Make leaf FICLO directories
            for path_name in ["product_ficlo", "output_galfit", "output_plots"]:
                # Create directory if it does not exist
                GALWRAP_PATHS[path_name].resolve(
                    galwrap_root=galwrap_config.galwrap_root,
                    product_root=galwrap_config.product_root,
                    output_root=galwrap_config.output_root,
                    field=ficl.field,
                    image_version=ficl.image_version,
                    catalog_version=ficl.catalog_version,
                    filter=ficl.filter,
                    object=object,
                ).mkdir(parents=True, exist_ok=True)
