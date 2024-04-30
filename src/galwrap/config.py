"""Configure and setup a program execution of the GalWrap package.
"""

# Imports


from typing import Optional, Any, Generator
from pathlib import Path
import itertools

from pydantic import BaseModel
import yaml

from astropy.io import ascii

from . import CONFIG_ROOT, PATH_NAMES


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
    objects : list[str]
        List of objects to execute this program over.
    fields : list[str]
        List of fields to execute this program over.
    image_versions : list[str]
        List of image versions to execute this program over.
    catalog_versions : list[int]
        List of catalog versions to execute this program over.
    morphology_versions : list[str]
        List of morphology versions to execute this program over.
    filters : list[str]
        List of filters to execute this program over.
    pixscales : list[float]
        List of pixscales to execute this program over.
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

    # Create GalWrapConfig from dict
    galwrap_config = GalWrapConfig(**config_dict)

    # Set filter info lists
    # TODO might need to be grouped with other variables
    filter_info = ascii.read(get_path("file_filtinfo", galwrap_config))
    galwrap_config.filters = filter_info["FILTNAME"]
    galwrap_config.pixscales = filter_info["PIXSCALES"]


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
                )
                / f"{image_version}.{catalog_version}"
            )
        ## OFI Directories
        case "imaging_bcgs":
            return (
                get_path(
                    name="imaging_of",
                    object=object,
                    field=field,
                    image_version=image_version,
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
                    imaging_root=imaging_root,
                    galwrap_config=galwrap_config,
                    catalog_version=catalog_version,
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
                )
                / filter
                / f"{galaxy_id}_{object}{field}-{filter}_sci.fits"
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
