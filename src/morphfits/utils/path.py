"""Resolve, create, and otherwise handle directory and file structure for
GalWrap.
"""

# Imports


import logging
import re
from pathlib import Path
from datetime import datetime as dt

import yaml
from pydantic import BaseModel

from .. import DATA_ROOT


# Constants


logger = logging.getLogger("PATHS")
"""Logging object for this module.
"""


PATH_STANDARDS = DATA_ROOT / "paths.yaml"
"""Path to data standard detailing paths for the MorphFITS filesystem structure.
"""


TEMPLATE_MAPPINGS = {
    "F": "field",
    "I": "image_version",
    "C": "catalog_version",
    "L": "filter",
    "O": "object",
    "D": "datetime",
    "N": "run_number",
}
"""Dict mapping from template abbreviations to parameter names.
"""


# Classes


class MorphFITSPath(BaseModel):
    """Path model for a single path in the MorphFITS filesystem structure.

    Parameters
    ----------
    BaseModel : class
        Base pydantic model class to enforce type validation upon creation.

    Attributes
    ----------
    file : bool
        Flag for whether path is a file.
    path : str
        Path to object (directory or file) as a template, where other paths are
        wrapped by square brackets and parameters are wrapped by curly brackets,
        e.g. `[input_root]/{F}/{I}`.
    alts : list[sr]
        List of recognized alternate names for this path.
    """

    file: bool
    path: str
    alts: list[str]

    def resolve(
        self,
        morphfits_root: Path | None = None,
        input_root: Path | None = None,
        output_root: Path | None = None,
        product_root: Path | None = None,
        run_root: Path | None = None,
        field: str | None = None,
        image_version: str | None = None,
        catalog_version: str | None = None,
        filter: str | None = None,
        object: int | None = None,
        datetime: dt | None = None,
        run_number: int | None = None,
    ) -> Path | None:
        """Get the full path for this path object for passed configuration
        parameters.

        Parameters
        ----------
        morphfits_root : Path | None, optional
            Path to root of MorphFITS filesystem, by default None.
        input_root : Path | None, optional
            Path to root input directory, by default None.
        output_root : Path | None, optional
            Path to root output directory, by default None.
        product_root : Path | None, optional
            Path to root products directory, by default None.
        run_root : Path | None, optional
            Path to root runs directory, by default None.
        field : str | None, optional
            Field of observation, by default None.
        image_version : str | None, optional
            Image version of science frame, by default None.
        catalog_version : str | None, optional
            Catalog version of science frame, by default None.
        filter : str | None, optional
            Filter used in observation, by default None.
        object : int | None, optional
            Target galaxy or cluster ID in catalog, by default None.
        datetime : datetime | None, optional
            Datetime at start of program run, by default None.
        run_number : int | None, optional
            Number of run in collection with same datetime, by default None.

        Returns
        -------
        Path
            Full path to directory or file corresponding to this path object.
        """
        # Convert start datetime and run number to string
        if datetime is not None:
            datetime_object = datetime
            datetime = datetime.strftime("%Y%m%dT%H%M%S")
        if run_number is not None:
            run_number_int = run_number
            run_number = str(run_number).rjust(2, "0")

        # Get passed values reference-able by name
        parameters = locals()

        # Initialize return value to template path
        resolved_path = self.path

        # Start by resolving other paths
        if "[" in resolved_path:
            pattern = "(\[.*\])"
            path_name = re.match(pattern, resolved_path).group()[1:-1]

            ## Replace base, input, products, or output paths in place
            if "root" in path_name:
                if "morphfits" in path_name:
                    resolved_path = re.sub(pattern, str(morphfits_root), resolved_path)
                elif "input" in path_name:
                    resolved_path = re.sub(pattern, str(input_root), resolved_path)
                elif "product" in path_name:
                    resolved_path = re.sub(pattern, str(product_root), resolved_path)
                elif "output" in path_name:
                    resolved_path = re.sub(pattern, str(output_root), resolved_path)
                elif "run" in path_name:
                    resolved_path = re.sub(pattern, str(run_root), resolved_path)
            ## Replace all other paths recursively
            else:
                resolved_path = re.sub(
                    pattern,
                    str(
                        MORPHFITS_PATHS[path_name].resolve(
                            morphfits_root=morphfits_root,
                            input_root=input_root,
                            output_root=output_root,
                            product_root=product_root,
                            run_root=run_root,
                            field=field,
                            image_version=image_version,
                            catalog_version=catalog_version,
                            filter=filter,
                            object=object,
                            datetime=(
                                datetime if datetime is None else datetime_object
                            ),
                            run_number=(
                                run_number if run_number is None else run_number_int
                            ),
                        )
                    ),
                    resolved_path,
                )

        # Finish by replacing all FICLO templates with corresponding values
        for template, parameter_name in TEMPLATE_MAPPINGS.items():
            if "{" + template + "}" in resolved_path:
                ## Handle special filter case
                ## Simulated STSci PSFs are named by single filter and in uppercase
                if (template == "L") and ("PSF_NIRCam" in resolved_path):
                    if "-" in filter:
                        filter_1, filter_2 = str(parameters[parameter_name]).split("-")
                        main_filter = filter_1 if "clear" in filter_2 else filter_2
                    else:
                        main_filter = filter
                    resolved_path = re.sub(
                        "({" + template + "})",
                        main_filter.upper(),
                        resolved_path,
                    )
                ## Otherwise, replace template with value in place
                else:
                    resolved_path = re.sub(
                        "({" + template + "})",
                        str(parameters[parameter_name]),
                        resolved_path,
                    )

        resolved_path_obj = get_path_obj(resolved_path)

        # Resolve glob paths to first file discovered (e.g. drc/drz)
        if "*" in resolved_path_obj.name:
            try:
                resolved_path_obj = list(
                    resolved_path_obj.parent.glob(resolved_path_obj.name)
                )[0]
            except:
                # logger.warning(
                #     f"File {resolved_path_obj} expected but not found, skipping."
                # )
                # raise FileNotFoundError(f"No file found at {resolved_path_obj}.")
                resolved_path_obj = Path(str(resolved_path_obj).replace("*", "c"))

        # Return resolved path
        return resolved_path_obj


## Instants


MORPHFITS_PATHS = {
    path_name: MorphFITSPath(**path_dict)
    for path_name, path_dict in yaml.safe_load(open(DATA_ROOT / "paths.yaml")).items()
}
"""Dict of paths used in MorphFITS, where the key, value pair is the path's
`path_name`, then a `MorphFITSPath` instance.
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
    for path_name, path_item in MORPHFITS_PATHS.items():
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


## Main


def get_path(
    name: str,
    morphfits_root: Path | None = None,
    input_root: Path | None = None,
    output_root: Path | None = None,
    product_root: Path | None = None,
    run_root: Path | None = None,
    field: str | None = None,
    image_version: str | None = None,
    catalog_version: str | None = None,
    filter: str | None = None,
    object: int | None = None,
    datetime: dt | None = None,
    run_number: int | None = None,
) -> Path | None:
    """Get the path to a MorphFITS file or directory.

    Parameters
    ----------
    name : str
        Name of path to get.
    morphfits_root : Path | None, optional
        Path to root of MorphFITS filesystem, by default None.
    input_root : Path | None, optional
        Path to root input directory, by default None.
    output_root : Path | None, optional
        Path to root output directory, by default None.
    product_root : Path | None, optional
        Path to root products directory, by default None.
    run_root : Path | None, optional
        Path to root runs directory, by default None.
    field : str | None, optional
        Field of observation, by default None.
    image_version : str | None, optional
        Image version of science frame, by default None.
    catalog_version : str | None, optional
        Catalog version of science frame, by default None.
    filter : str | None, optional
        Filter used in observation, by default None.
    object : int | None, optional
        Target galaxy or cluster ID in catalog, by default None.
    datetime : datetime | None, optional
        Datetime at start of program run, by default None.
    run_number : int | None, optional
        Number of run in collection with same datetime, by default None.

    Returns
    -------
    Path
        Path to file or directory.

    See Also
    --------
    data/paths.yaml
        Data standards dictionary detailing MorphFITS paths.
    """
    # Resolve name
    path_name = get_path_name(name)

    # Resolve path for given parameters
    return MORPHFITS_PATHS[path_name].resolve(
        morphfits_root=morphfits_root,
        input_root=input_root,
        output_root=output_root,
        product_root=product_root,
        run_root=run_root,
        field=field,
        image_version=image_version,
        catalog_version=catalog_version,
        filter=filter,
        object=object,
        datetime=datetime,
        run_number=run_number,
    )
