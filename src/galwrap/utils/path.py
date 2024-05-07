"""Resolve, create, and otherwise handle directory and file structure for
GalWrap.
"""

# Imports


from pathlib import Path
from typing import Optional, Self

from pydantic import BaseModel
import yaml

from .. import PATH_NAMES, DATA_ROOT
from ..config import GalWrapConfig, OFIC


# Constants


# Classes


class GalWrapPath:
    is_dir: bool
    path: Path
    children: list[Self]
    directories: list[Self]
    files: list[Self]
    filename: str
    pathname: str | None
    alt_names: list[str] | None
    description: str | None


# Functions


def read_path_standards(
    path_standards_path: str | Path = DATA_ROOT / "paths.yaml",
) -> tuple[GalWrapPath, GalWrapPath, GalWrapPath]:
    path_standards = yaml.safe_load(open(path_standards_path))


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
    A path name is resolvable if its casefold is equal to
    1. The standardized path name itself
        e.g `output_stamps`
    2. A recognized alternative name
        e.g. `stamps` for `output_stamps`
    3. The standardized path name, un-pluralized
        e.g. `output_stamp` for `output_stamps`
    4. The standardized path name, separated by spaces rather than underscores
        e.g. `output stamps` for `output_stamps`
    5. The standardized path name, space-separated, and with its word order
       reversed
        e.g. `stamps output` for `output_stamps`
    6. The standardized path name, un-pluralized, space-separated, and with its
       word order reversed
        e.g. `stamp output` for `output_stamps`

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
        PATH_NAMES["photometry"] | PATH_NAMES["imaging"] | PATH_NAMES["output"]
    ).items():
        # 1. Exact name match
        if name.casefold() == path_name.casefold():
            matching_name = True
        # 2. Alternative name match
        elif name.casefold() in [r_n.casefold() for r_n in recognized_names]:
            matching_name = True
        # 3. Un-pluralized match
        elif ("s" == path_name[-1]) and (name.casefold() == path_name[:-1].casefold()):
            matching_name = True
        # 4. Space-separated match
        elif name.casefold() == " ".join(path_name.casefold().split("_")):
            matching_name = True
        # 5. Case 4 and word order reversed
        elif name.casefold() == " ".join(
            path_name.casefold().split("_")[1:] + [path_name.casefold().split("_")[0]]
        ):
            matching_name = True
        # 6. Cases 3 and 5
        elif ("s" == path_name[-1]) and (
            name.casefold()
            == " ".join(
                path_name[:-1].casefold().split("_")[1:]
                + [path_name[:-1].casefold().split("_")[0]]
            )
        ):
            matching_name = True
        # Otherwise not matching, continue search
        else:
            matching_name = False

        # Return standardized path name if matching
        if matching_name:
            return path_name

    # Terminate if name not found
    raise ValueError(f"Unrecognized path name '{name}'.")


def get_path(
    name: str,
    galwrap_config: GalWrapConfig | None = None,
    photometry_root: Path | str | None = None,
    imaging_root: Path | str | None = None,
    output_root: Path | str | None = None,
    ofic: OFIC | None = None,
    object: str | None = None,
    field: str | None = None,
    image_version: str | None = None,
    catalog_version: int | None = None,
    filter: str | None = None,
    galaxy_id: int | None = None,
) -> Path:
    """Get the path to a directory or file via a recognized name.

    Parameters
    ----------
    name : str
        Name of the directory or file.
    galwrap_config : GalWrapConfig | None, optional
        Configuration parameters for program execution, by default None.
    photometry_root : Path | str | None, optional
        Path to root of input photometry products, by default unspecified,
        meaning it must be set by a passed GalWrapConfig object.
    imaging_root : Path | str | None, optional
        Path to root of input imaging products, by default unspecified, meaning
        it must be set by a passed GalWrapConfig object.
    output_root : Path | str | None, optional
        Path to root directory of output GalWrap products, by default
        unspecified, meaning it must be set by a passed GalWrapConfig object.
    ofic : OFIC | None, optional
        Data object containing the object, field, image version, and catalog
        version for a program execution, by default None.
    object : str | None, optional
        Center of cluster to be fitted, by default unspecified, meaning it must
        be set by a passed OFIC object.
    field : str | None, optional
        Field of cluster to be fitted, by default unspecified, meaning it must
        be set by a passed OFIC object.
    image_version : str | None, optional
        Image version of input data, by default unspecified, meaning it must be
        set by a passed OFIC object.
    catalog_version : int | None, optional
        Catalog version of input data, by default unspecified, meaning it must
        be set by a passed OFIC object.
    filter : str | None, optional
        Filter of input image data, by default None (unspecified).
    galaxy_id : int | None, optional
        ID of galaxy, by default None.

    Returns
    -------
    Path
        Path to directory or file corresponding to passed name.

    Raises
    ------
    AttributeError
        Necessary input passed neither directly nor via object, e.g.
        `object=None` and `ofic=None`.

    See Also
    --------
    data.path_names.yaml
        Standardized path names and their recognized alternative names.
    """
    # Get corresponding standardized path name
    path_name = resolve_path_name(name)

    # All paths require an object, field, and image version
    # Set by preference of parameter, then config/object
    # Terminate if neither value found
    if object is None:
        if ofic is None:
            raise AttributeError("Object not specified.")
        else:
            object = ofic.object
    if field is None:
        if ofic is None:
            raise AttributeError("Field not specified.")
        else:
            field = ofic.field
    if image_version is None:
        if ofic is None:
            raise AttributeError("Image version not specified.")
        else:
            image_version = ofic.image_version

    # Return corresponding path
    match path_name:
        # Photometry
        ## Parent
        case "photometry_ofi":
            # Ensure photometry root path set
            if photometry_root is None:
                if galwrap_config is None:
                    raise AttributeError("Photometry root not specified.")
                else:
                    photometry_root = galwrap_config.photometry_root
            elif isinstance(photometry_root, str):
                photometry_root = Path(photometry_root).resolve()

            return photometry_root / object / field / image_version
        ## OFI Directories
        case "photometry_rms":
            return (
                get_path(
                    name="photometry_ofi",
                    object=object,
                    field=field,
                    image_version=image_version,
                    photometry_root=photometry_root,
                    galwrap_config=galwrap_config,
                    ofic=ofic,
                )
                / "rms"
            )

        # Imaging
        ## Parent
        case "imaging_of":
            # Ensure imaging root path set
            if imaging_root is None:
                if galwrap_config is None:
                    raise AttributeError("Imaging root not specified.")
                else:
                    imaging_root = galwrap_config.imaging_root
            elif isinstance(imaging_root, str):
                imaging_root = Path(imaging_root).resolve()

            return imaging_root / object / field
        case "imaging_ofic":
            # Ensure catalog version set
            if catalog_version is None:
                if ofic is None:
                    raise AttributeError("Catalog version not specified.")
                else:
                    catalog_version = ofic.catalog_version
            return (
                get_path(
                    name="imaging_of",
                    object=object,
                    field=field,
                    image_version=image_version,
                    imaging_root=imaging_root,
                    galwrap_config=galwrap_config,
                    ofic=ofic,
                )
                / f"{image_version}.{catalog_version}"
            )
        ## OFI Directories
        case "imaging_sci":
            return (
                get_path(
                    name="imaging_of",
                    object=object,
                    field=field,
                    image_version=image_version,
                    imaging_root=imaging_root,
                    galwrap_config=galwrap_config,
                    ofic=ofic,
                )
                / image_version
                / "science"
            )
        case "imaging_bcgs":
            return (
                get_path(
                    name="imaging_of",
                    object=object,
                    field=field,
                    image_version=image_version,
                    imaging_root=imaging_root,
                    galwrap_config=galwrap_config,
                    ofic=ofic,
                )
                / image_version
                / "bcgs"
            )
        ## OFIC Directories
        case "imaging_psfs":
            return (
                get_path(
                    name="imaging_ofic",
                    object=object,
                    field=field,
                    image_version=image_version,
                    imaging_root=imaging_root,
                    galwrap_config=galwrap_config,
                    catalog_version=catalog_version,
                    ofic=ofic,
                )
                / "psfs"
            )
        case "imaging_catalogs":
            return (
                get_path(
                    name="imaging_ofic",
                    object=object,
                    field=field,
                    image_version=image_version,
                    imaging_root=imaging_root,
                    galwrap_config=galwrap_config,
                    catalog_version=catalog_version,
                    ofic=ofic,
                )
                / "catalogs"
            )
        ## OFIC Files
        case "file_segmap":
            return (
                get_path(
                    name="imaging_catalogs",
                    object=object,
                    field=field,
                    image_version=image_version,
                    imaging_root=imaging_root,
                    galwrap_config=galwrap_config,
                    catalog_version=catalog_version,
                    ofic=ofic,
                )
                / f"{object}{field}_photutils_segmap_{image_version}.{catalog_version}.fits"
            )
        case "file_photometry_catalog":
            return (
                get_path(
                    name="imaging_catalogs",
                    object=object,
                    field=field,
                    image_version=image_version,
                    imaging_root=imaging_root,
                    galwrap_config=galwrap_config,
                    catalog_version=catalog_version,
                    ofic=ofic,
                )
                / f"{object}{field}_photutils_cat_{image_version}.{catalog_version}.fits"
            )

        # Output
        ## Parent
        case "output_ofic":
            # Ensure output root path set
            if output_root is None:
                if galwrap_config is None:
                    raise AttributeError("Output root not specified.")
                else:
                    output_root = galwrap_config.output_root
            elif isinstance(output_root, str):
                output_root = Path(output_root).resolve()

            # Ensure catalog version set
            if catalog_version is None:
                if ofic is None:
                    raise AttributeError("Catalog version not specified.")
                else:
                    catalog_version = ofic.catalog_version

            return output_root / object / field / f"{image_version}.{catalog_version}"
        ## OFIC Directories
        case "output_segmaps":
            return (
                get_path(
                    name="output_ofic",
                    object=object,
                    field=field,
                    image_version=image_version,
                    output_root=output_root,
                    galwrap_config=galwrap_config,
                    catalog_version=catalog_version,
                    ofic=ofic,
                )
                / "segmaps"
            )
        case "output_masks":
            return (
                get_path(
                    name="output_ofic",
                    object=object,
                    field=field,
                    image_version=image_version,
                    output_root=output_root,
                    galwrap_config=galwrap_config,
                    catalog_version=catalog_version,
                    ofic=ofic,
                )
                / "masks"
            )
        case "output_psfs":
            return (
                get_path(
                    name="output_ofic",
                    object=object,
                    field=field,
                    image_version=image_version,
                    output_root=output_root,
                    galwrap_config=galwrap_config,
                    catalog_version=catalog_version,
                    ofic=ofic,
                )
                / "psfs"
            )
        case "output_rms":
            return (
                get_path(
                    name="output_ofic",
                    object=object,
                    field=field,
                    image_version=image_version,
                    output_root=output_root,
                    galwrap_config=galwrap_config,
                    catalog_version=catalog_version,
                    ofic=ofic,
                )
                / "rms"
            )
        case "output_stamps":
            return (
                get_path(
                    name="output_ofic",
                    object=object,
                    field=field,
                    image_version=image_version,
                    output_root=output_root,
                    galwrap_config=galwrap_config,
                    catalog_version=catalog_version,
                    ofic=ofic,
                )
                / "stamps"
            )
        case "output_feedfiles":
            return (
                get_path(
                    name="output_ofic",
                    object=object,
                    field=field,
                    image_version=image_version,
                    output_root=output_root,
                    galwrap_config=galwrap_config,
                    catalog_version=catalog_version,
                    ofic=ofic,
                )
                / "feedfiles"
            )
        case "output_galfit":
            return (
                get_path(
                    name="output_ofic",
                    object=object,
                    field=field,
                    image_version=image_version,
                    output_root=output_root,
                    galwrap_config=galwrap_config,
                    catalog_version=catalog_version,
                    ofic=ofic,
                )
                / "galfit_output"
            )
        case "output_visualizations":
            return (
                get_path(
                    name="output_ofic",
                    object=object,
                    field=field,
                    image_version=image_version,
                    output_root=output_root,
                    galwrap_config=galwrap_config,
                    catalog_version=catalog_version,
                    ofic=ofic,
                )
                / "visualizations"
            )
        ## OFIC Files
        case "file_filter_info":
            return (
                get_path(
                    name="output_ofic",
                    object=object,
                    field=field,
                    image_version=image_version,
                    output_root=output_root,
                    galwrap_config=galwrap_config,
                    catalog_version=catalog_version,
                    ofic=ofic,
                )
                / f"{object}{field}_{image_version}_filter_info.dat"
            )
        case "file_depth":
            return (
                get_path(
                    name="output_ofic",
                    object=object,
                    field=field,
                    image_version=image_version,
                    output_root=output_root,
                    galwrap_config=galwrap_config,
                    catalog_version=catalog_version,
                    ofic=ofic,
                )
                / f"{object}{field}_{image_version}_depth.txt"
            )
        case "file_mask":
            return (
                get_path(
                    name="output_masks",
                    object=object,
                    field=field,
                    image_version=image_version,
                    output_root=output_root,
                    galwrap_config=galwrap_config,
                    catalog_version=catalog_version,
                    ofic=ofic,
                )
                / f"{galaxy_id}_{object}{field}_mask.fits"
            )
        case "file_science":
            # Ensure filter is set
            if filter is None:
                if (ofic is None) or (ofic.filter is None):
                    raise AttributeError("Filter not set.")
                else:
                    filter = ofic.filter

            return (
                get_path(
                    name="output_stamps",
                    object=object,
                    field=field,
                    image_version=image_version,
                    output_root=output_root,
                    galwrap_config=galwrap_config,
                    catalog_version=catalog_version,
                    ofic=ofic,
                )
                / filter
                / f"{galaxy_id}_{object}{field}-{filter}_sci.fits"
            )


def setup_directories(
    galwrap_config: GalWrapConfig | None = None,
    photometry_root: Path | str | None = None,
    imaging_root: Path | str | None = None,
    output_root: Path | str | None = None,
    ofic: OFIC | None = None,
    object: str | None = None,
    field: str | None = None,
    image_version: str | None = None,
    catalog_version: int | None = None,
    filter: str | None = None,
    filters: list[str] | None = None,
):
    """Validate input directory existence and create output directories where
    nonexistent, for a given program configuration.

    Parameters
    ----------
    galwrap_config : GalWrapConfig | None, optional
        Configuration parameters for program execution, by default None.
    photometry_root : Path | str | None, optional
        Path to root of input photometry products, by default unspecified,
        meaning it must be set by a passed GalWrapConfig object.
    imaging_root : Path | str | None, optional
        Path to root of input imaging products, by default unspecified, meaning
        it must be set by a passed GalWrapConfig object.
    output_root : Path | str | None, optional
        Path to root directory of output GalWrap products, by default
        unspecified, meaning it must be set by a passed GalWrapConfig object.
    ofic : OFIC | None, optional
        Data object containing the object, field, image version, and catalog
        version for a program execution, by default None.
    object : str | None, optional
        Center of cluster to be fitted, by default unspecified, meaning it must
        be set by a passed OFIC object.
    field : str | None, optional
        Field of cluster to be fitted, by default unspecified, meaning it must
        be set by a passed OFIC object.
    image_version : str | None, optional
        Image version of input data, by default unspecified, meaning it must be
        set by a passed OFIC object.
    catalog_version : int | None, optional
        Catalog version of input data, by default unspecified, meaning it must
        be set by a passed OFIC object.
    filter : str | None, optional
        Filter of input image data, by default None (unspecified).
    filters : list[str] | None, optional
        List of filters of input image data, by default None (unspecified).

    Raises
    ------
    FileNotFoundError
        Input directory not found.
    """
    # Validate each expected input directory exists
    for path_name in PATH_NAMES["photometry"] | PATH_NAMES["imaging"]:
        # Skip files
        if "file" in path_name:
            continue

        # Terminate if input directory does not exist
        directory_path = get_path(
            path_name,
            galwrap_config=galwrap_config,
            photometry_root=photometry_root,
            imaging_root=imaging_root,
            ofic=ofic,
            object=object,
            field=field,
            image_version=image_version,
            catalog_version=catalog_version,
        )
        if not (directory_path.exists() and directory_path.is_dir()):
            raise FileNotFoundError(
                f"Input directory {path_name} expected, but not found."
            )

        # TODO check file existence

    # Iterate over each expected output directory
    for path_name in PATH_NAMES["output"]:
        # Skip files
        if "file" in path_name:
            continue

        # Create output directory if does not exist
        get_path(
            path_name,
            galwrap_config=galwrap_config,
            output_root=output_root,
            ofic=ofic,
            object=object,
            field=field,
            image_version=image_version,
            catalog_version=catalog_version,
        ).mkdir(parents=True, exist_ok=True)

    # Create filtered output subdirectories
    filtered_output_dirs = [
        "output_feedfiles",
        "output_galfit",
        "output_rms",
        "output_stamps",
    ]

    ## If list of filters is provided
    if filters is not None:
        for filtered_output_dir in filtered_output_dirs:
            for filter in filters:
                (
                    get_path(
                        filtered_output_dir,
                        galwrap_config=galwrap_config,
                        output_root=output_root,
                        ofic=ofic,
                        object=object,
                        field=field,
                        image_version=image_version,
                        catalog_version=catalog_version,
                    )
                    / filter
                ).mkdir(parents=True, exist_ok=True)

    ## If single filter is provided
    if filter is not None:
        for filtered_output_dir in filtered_output_dirs:
            (
                get_path(
                    filtered_output_dir,
                    galwrap_config=galwrap_config,
                    output_root=output_root,
                    ofic=ofic,
                    object=object,
                    field=field,
                    image_version=image_version,
                    catalog_version=catalog_version,
                )
                / filter
            ).mkdir(parents=True, exist_ok=True)
