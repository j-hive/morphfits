"""Resolve, create, and otherwise handle directory and file structure for
GalWrap.
"""

# Imports


from pathlib import Path

from .. import PATH_NAMES
from ..config import GalWrapConfig, OFIC
from . import utils


# Functions


def resolve_single_parameter(
    name: str,
    parameter: int | float | str | Path | list[float] | list[str] | None,
    instance: GalWrapConfig | OFIC | None,
) -> int | float | str | Path | list[float] | list[str] | None:
    name = name.lower()
    if (parameter is None) and (instance is None):
        raise AttributeError(f"{name.capitalize()} not specified.")
    else:
        return vars(instance)[name] if parameter is None else parameter


def resolve_parameter(
    name: str,
    parameter: int | float | str | Path | list[float] | list[str] | None,
    instance: GalWrapConfig | OFIC | None,
) -> int | float | str | Path | list[float] | list[str]:
    # Set name for case-less comparison
    name = name.casefold()

    # Validate parameter existence and return corresponding value, in preference
    # of directly passed parameter, then object instance attribute

    # OFIC
    ## Object
    if (name == "o") or (name == "obj") or (name == "object"):
        return resolve_single_parameter("object", parameter, instance)
    ## Field
    elif (name == "f") or (name == "field"):
        return resolve_single_parameter("field", parameter, instance)
    ## Image version
    elif (name == "i") or (name == "imver") or ("image" in name):
        return resolve_single_parameter("image_version", parameter, instance)
    ## Catalog version
    elif (name == "c") or (name == "catver") or ("catalog" in name):
        return resolve_single_parameter("catalog_version", parameter, instance)

    # Paths
    ## Photometry
    elif "photometry" in name:
        photometry_root = resolve_single_parameter(
            "photometry_root", parameter, instance
        )
        return (
            Path(photometry_root).resolve()
            if isinstance(photometry_root, str)
            else photometry_root
        )
    ## Imaging
    elif "imaging" in name:
        imaging_root = resolve_single_parameter("imaging_root", parameter, instance)
        return (
            Path(imaging_root).resolve()
            if isinstance(imaging_root, str)
            else imaging_root
        )
    ## Output
    elif "output" in name:
        output_root = resolve_single_parameter("output_root", parameter, instance)
        return (
            Path(output_root).resolve() if isinstance(output_root, str) else output_root
        )

    # Other
    ## Single filter
    elif (name == "l") or (name == "filt") or (name == "filter"):
        filter = resolve_single_parameter("filter", parameter, instance)
        if filter is None:
            raise AttributeError("Filter not specified.")
        else:
            return filter
    ## Multiple filters
    elif (name == "filts") or (name == "filters"):
        filters = resolve_single_parameter("filters", parameter, instance)
        if (filters is None) or (len(filters) == 0):
            raise AttributeError("Filters not specified.")
        else:
            return filters
    ## Single pixscale
    elif (name == "p") or (name == "pixscale"):
        pixscale = resolve_single_parameter("pixscale", parameter, instance)
        if pixscale is None:
            raise AttributeError("Pixscale not specified.")
        else:
            return pixscale
    ## Multiple pixscales
    elif name == "pixscales":
        pixscales = resolve_single_parameter("pixscales", parameter, instance)
        if (pixscales is None) or (len(pixscales) == 0):
            raise AttributeError("Pixscales not specified.")
        else:
            return pixscales
    ## Galaxy ID
    elif (name == "g") or (name == "id") or ("galaxy" in name):
        return resolve_single_parameter("galaxy_id", parameter, instance)

    # Unrecognized
    else:
        raise NotImplementedError(f"Parameter {name} unrecognized.")


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
    pixscale: float | None = None,
    pixscales: list[float] | None = None,
) -> Path | list[Path]:
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
    pixscale : float | None, optional
        Pixel scale resolution, by default None.
    pixscales : list[float] | None, optional
        Pixel scale resolutions, by default None.

    Returns
    -------
    Path | list[Path]
        Path(s) to directory or file corresponding to passed name.

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
    object = resolve_parameter("object", object, ofic)
    field = resolve_parameter("field", field, ofic)
    image_version = resolve_parameter("image_version", image_version, ofic)

    # Return corresponding path
    match path_name:
        # Photometry
        ## Parent
        case "photometry_ofi":
            photometry_root = resolve_parameter(
                "photometry_root", photometry_root, galwrap_config
            )
            return photometry_root / object / field / image_version
        ## OFI Directories
        case "photometry_rms":
            return (
                get_path(
                    name="photometry_ofi",
                    object=object,
                    field=field,
                    image_version=image_version,
                    ofic=ofic,
                    photometry_root=photometry_root,
                    galwrap_config=galwrap_config,
                )
                / "rms"
            )

        # Imaging
        ## Parent
        case "imaging_of":
            imaging_root = resolve_parameter(
                "imaging_root", imaging_root, galwrap_config
            )
            return imaging_root / object / field
        case "imaging_ofic":
            catalog_version = resolve_parameter(
                "catalog_version", catalog_version, ofic
            )
            return (
                get_path(
                    name="imaging_of",
                    object=object,
                    field=field,
                    image_version=image_version,
                    ofic=ofic,
                    imaging_root=imaging_root,
                    galwrap_config=galwrap_config,
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
                    ofic=ofic,
                    imaging_root=imaging_root,
                    galwrap_config=galwrap_config,
                )
                / image_version
                / "science"
            )
        case "file_science_images":
            # TODO handle filters too
            # Paths to science images for single pixscale
            try:
                pixscale = resolve_parameter("pixscale", pixscale, ofic)
                return list(
                    get_path(
                        name="imaging_sci",
                        object=object,
                        field=field,
                        image_version=image_version,
                        ofic=ofic,
                        imaging_root=imaging_root,
                        galwrap_config=galwrap_config,
                    ).glob(f"*{utils.scale_to_name(pixscale)}*_sci.fits")
                )
            # Paths to science images for multiple pixscales
            except AttributeError:
                pixscales = resolve_parameter("pixscales", pixscales, galwrap_config)
                science_images = []
                for pixscale in pixscales:
                    # Get paths to each science image
                    science_images_pixscale = list(
                        get_path(
                            name="imaging_sci",
                            object=object,
                            field=field,
                            image_version=image_version,
                            ofic=ofic,
                            imaging_root=imaging_root,
                            galwrap_config=galwrap_config,
                        ).glob(f"*{utils.scale_to_name(pixscale)}*_sci.fits")
                    )

                    # Add paths to total list
                    for science_image in science_images_pixscale:
                        science_images.append(science_image)
                return science_images
        case "imaging_bcgs":
            return (
                get_path(
                    name="imaging_of",
                    object=object,
                    field=field,
                    image_version=image_version,
                    ofic=ofic,
                    imaging_root=imaging_root,
                    galwrap_config=galwrap_config,
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
                    catalog_version=catalog_version,
                    ofic=ofic,
                    imaging_root=imaging_root,
                    galwrap_config=galwrap_config,
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
                    catalog_version=catalog_version,
                    ofic=ofic,
                    imaging_root=imaging_root,
                    galwrap_config=galwrap_config,
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
                    catalog_version=catalog_version,
                    ofic=ofic,
                    imaging_root=imaging_root,
                    galwrap_config=galwrap_config,
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
                    catalog_version=catalog_version,
                    ofic=ofic,
                    imaging_root=imaging_root,
                    galwrap_config=galwrap_config,
                )
                / f"{object}{field}_photutils_cat_{image_version}.{catalog_version}.fits"
            )

        # Output
        ## Parent
        case "output_ofic":
            output_root = resolve_parameter("output_root", output_root, galwrap_config)
            catalog_version = resolve_parameter(
                "catalog_version", catalog_version, ofic
            )
            return output_root / object / field / f"{image_version}.{catalog_version}"
        ## OFIC Directories
        case "output_segmaps":
            return (
                get_path(
                    name="output_ofic",
                    object=object,
                    field=field,
                    image_version=image_version,
                    catalog_version=catalog_version,
                    ofic=ofic,
                    output_root=output_root,
                    galwrap_config=galwrap_config,
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
                    catalog_version=catalog_version,
                    ofic=ofic,
                    output_root=output_root,
                    galwrap_config=galwrap_config,
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
                    catalog_version=catalog_version,
                    ofic=ofic,
                    output_root=output_root,
                    galwrap_config=galwrap_config,
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
                    catalog_version=catalog_version,
                    ofic=ofic,
                    output_root=output_root,
                    galwrap_config=galwrap_config,
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
                    catalog_version=catalog_version,
                    ofic=ofic,
                    output_root=output_root,
                    galwrap_config=galwrap_config,
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
                    catalog_version=catalog_version,
                    ofic=ofic,
                    output_root=output_root,
                    galwrap_config=galwrap_config,
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
                    catalog_version=catalog_version,
                    ofic=ofic,
                    output_root=output_root,
                    galwrap_config=galwrap_config,
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
                    catalog_version=catalog_version,
                    ofic=ofic,
                    output_root=output_root,
                    galwrap_config=galwrap_config,
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
                    catalog_version=catalog_version,
                    ofic=ofic,
                    output_root=output_root,
                    galwrap_config=galwrap_config,
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
                    catalog_version=catalog_version,
                    ofic=ofic,
                    output_root=output_root,
                    galwrap_config=galwrap_config,
                )
                / f"{galaxy_id}_{object}{field}_mask.fits"
            )
        case "file_science":
            filter = resolve_parameter("filter", filter, ofic)
            galaxy_id = resolve_parameter("galaxy_id", galaxy_id, ofic)
            return (
                get_path(
                    name="output_stamps",
                    object=object,
                    field=field,
                    image_version=image_version,
                    catalog_version=catalog_version,
                    ofic=ofic,
                    output_root=output_root,
                    galwrap_config=galwrap_config,
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
