"""Configure and setup a program executionOptional\[ of the GalWrap package.
"""

# Imports


from typing import Optional, Generator
from pathlib import Path
import itertools

from pydantic import BaseModel
import yaml
from astropy.table import Table

from .utils import utils


# Classes


class OFIC(BaseModel):
    """Configuration model for a single program execution of GalFit.

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
    galaxy_id : int | None, optional
        ID of galaxy to be fitted, by default None.
    """

    object: str
    field: str
    image_version: str
    catalog_version: int
    morphology_version: Optional[str] = None
    filter: Optional[str] = None
    pixscale: Optional[float] = None
    galaxy_id: Optional[int] = None


class GalWrapConfig(BaseModel):
    """Configuration model for a program execution of GalWrap.

    Parameters
    ----------
    BaseModel : class
        Base pydantic model class to enforce type validation upon creation.

    Attributes
    ----------
    input_root : Path
        Path to root of input products, e.g. science, RMS, PSF images, and
        catalogs.
    product_root : Path
        Path to root of products produced by this program to run GALFIT, e.g.
        stamp, sigma, mask, or psf images.
    output_root : Path
        Path to root of directory to which to write GALFIT output products.
    objects : list[str] | None, optional
        List of objects to execute this program over, by default None.
    fields : list[str] | None, optional
        List of fields to execute this program over, by default None.
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
    galaxy_ids : list[int] | None, optional
        List of object indices to execute this program over, by default None.
    """

    input_root: Path
    product_root: Path
    output_root: Path
    objects: Optional[list[str]] = None
    fields: Optional[list[str]] = None
    image_versions: Optional[list[str]] = None
    catalog_versions: Optional[list[int]] = None
    morphology_versions: Optional[list[str]] = None
    filters: Optional[list[str]] = None
    pixscales: Optional[list[float]] = None
    galaxy_ids: Optional[list[int]] = None

    def get_ofics(
        self,
        objects: list[str] | None = None,
        fields: list[str] | None = None,
        image_versions: list[str] | None = None,
        catalog_versions: list[int] | None = None,
        morphology_versions: list[str] | None = None,
        filters: list[str] | None = None,
        pixscales: list[float] | None = None,
        galaxy_ids: list[int] | None = None,
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
        galaxy_ids : list[int] | None, optional
            List of galaxy object indices over which to execute this program, by
            default None (all of them).

        Yields
        ------
        OFIC
            OFIC permutation from specified objects, fields, image, catalog, and
            morphology versions, filters, pixel scales, and galaxy IDs.
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
            or ((galaxy_ids is None) and (self.galaxy_ids is None))
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
            galaxy_id,
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
            self.galaxy_ids if galaxy_ids is None else galaxy_ids,
        ):
            yield OFIC(
                object=object,
                field=field,
                image_version=image_version,
                catalog_version=catalog_version,
                morphology_version=morphology_version,
                filter=filter,
                pixscale=pixscale,
                galaxy_id=galaxy_id,
            )


# Functions


def create_config(
    config_path: str | Path | None = None,
    input_root: str | Path | None = None,
    product_root: str | Path | None = None,
    output_root: str | Path | None = None,
    object: str | None = None,
    objects: list[str] | None = None,
    field: str | None = None,
    fields: list[str] | None = None,
    image_version: str | None = None,
    image_versions: list[str] | None = None,
    catalog_version: int | None = None,
    catalog_versions: list[int] | None = None,
    morphology_version: str | None = None,
    morphology_versions: list[str] | None = None,
    filter: str | None = None,
    filters: list[str] | None = None,
    pixscale: float | None = None,
    pixscales: list[float] | None = None,
    galaxy_id: int | None = None,
    galaxy_ids: list[int] | None = None,
) -> GalWrapConfig:
    """Create a configuration object from hierarchically preferred variables, in
    order of CLI passed values, then config file declared values, then values
    found by directory discovery.

    Parameters
    ----------
    config_path : str | Path | None, optional
        Path to user config yaml file, by default None (no user config file
        provided).

    Returns
    -------
    GalWrapConfig
        A configuration object for this program execution.
    """
    # Load config file values
    config_dict = {} if config_path is None else yaml.safe_load(open(config_path))

    # Set any parameters passed through CLI call
    ## Paths
    if input_root is not None:
        config_dict["input_root"] = utils.get_path(input_root)
    if product_root is not None:
        config_dict["product_root"] = utils.get_path(product_root)
    if output_root is not None:
        config_dict["output_root"] = utils.get_path(output_root)

    ## Multiple OFICs
    if objects is not None:
        config_dict["objects"] = objects
    if fields is not None:
        config_dict["fields"] = fields
    if image_versions is not None:
        config_dict["image_versions"] = image_versions
    if catalog_versions is not None:
        config_dict["catalog_versions"] = catalog_versions
    if morphology_versions is not None:
        config_dict["morphology_versions"] = morphology_versions
    if filters is not None:
        config_dict["filters"] = filters
    if pixscales is not None:
        config_dict["pixscales"] = pixscales
    if galaxy_ids is not None:
        config_dict["galaxy_ids"] = galaxy_ids

    ## Single OFICs - note this will override multiple OFICs if set
    if object is not None:
        config_dict["objects"] = [object]
    if field is not None:
        config_dict["fields"] = [field]
    if image_version is not None:
        config_dict["image_versions"] = [image_version]
    if catalog_version is not None:
        config_dict["catalog_versions"] = [catalog_version]
    if morphology_version is not None:
        config_dict["morphology_versions"] = [morphology_version]
    if filter is not None:
        config_dict["filters"] = [filter]
    if pixscale is not None:
        config_dict["pixscales"] = [pixscale]
    if galaxy_id is not None:
        config_dict["galaxy_ids"] = [galaxy_id]

    # If parameters are still unset, assume program execution over all
    # discovered values in input directory
    if "objects" not in config_dict:
        config_dict["objects"] = 2

    # Create GalWrapConfig from dict
    galwrap_config = GalWrapConfig(**config_dict)

    # # Setup directories
    # path.setup_directories(galwrap_config=galwrap_config)

    # # Create filter info tables
    # ## TODO set how ofic is chosen
    # input_science_files: list[Path] = path.get_path(
    #     "file_science_images", galwrap_config=galwrap_config, ofic=ofic
    # )
    # for input_science_file in input_science_files:
    #     galwrap_config.filters.append(
    #         input_science_file.name.split("-")[1].split("_")[0]
    #     )
    # ## TODO is this where pixscales are from?
    # galwrap_config.pixscales.append("40mas")

    # ## Create table of three columns in order of filters, pixscales, and pixnames
    # num_filters = len(galwrap_config.filters)
    # filter_info = Table(
    #     [
    #         galwrap_config.filters,
    #         [galwrap_config.pixscales[0] for i in range(num_filters)],
    #         [
    #             utils.scale_to_name(galwrap_config.pixscales[0])
    #             for i in range(num_filters)
    #         ],
    #     ]
    # )

    # ## Write table to file
    # ascii.write(
    #     filter_info,
    #     path.get_path("file_filter_info", galwrap_config=galwrap_config, ofic=ofic),
    # )

    # Return created config object
    return galwrap_config


# Instantiations


# galwrap_config = create_config()
"""Config object instantiation.
"""
