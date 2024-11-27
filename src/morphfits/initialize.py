"""Obtain input from the DJA archive.

References
----------
1. [DJA v7 Mosaic Release Details](https://dawn-cph.github.io/dja/imaging/v7/)
2. [DJA v7 Mosaic
   Release](https://s3.amazonaws.com/grizli-v2/JwstMosaics/v7/index.html)
"""

# Imports


import logging
import gzip
import shutil, os
import tempfile
from urllib import request
from pathlib import Path
import csv

from tqdm import tqdm

from . import settings
from .settings import RuntimeSettings
from .utils import misc


# Constants


logger = logging.getLogger("INITIALIZE")
"""Logger object for this module.
"""


DJA_BASE_URL = "https://s3.amazonaws.com/grizli-v2/JwstMosaics/v7"
"""URL to DJA v7 Mosaics archive.
"""


DJA_INDEX_ENDPOINT = "index.csv"
"""Endpoint for filelist CSV.
"""


# Classes


class DownloadProgressBar(tqdm):
    """Progress bar for download progress display."""

    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


# Functions


def get_dja_catalog() -> dict[str, str]:
    """Get a list of files available for download from the DJA archive as a dict
    from a filename to its corresponding download link.

    Returns
    -------
    dict[str, str]
        DJA download catalog, as download links indexed by their filenames.
    """
    logger.info("Downloading DJA file list.")

    # Try to get DJA file list
    try:
        # Create temporary file to store index CSV
        temp_csv = tempfile.NamedTemporaryFile(delete=False)

        # Get table with links to all files from DJA archive
        request.urlretrieve(
            url=f"{DJA_BASE_URL}/{DJA_INDEX_ENDPOINT}", filename=temp_csv.name
        )

        # Get dict mapping filenames to their download links
        dja_catalog = {}
        with open(temp_csv.name, mode="r", newline="") as csv_file:
            reader = csv.reader(csv_file)

            # Iterate over each line in CSV
            skipped_headers = False
            for line in reader:
                # Skip header row
                if not skipped_headers:
                    skipped_headers = True
                    continue

                # Skip non-FITS files
                if "fits" not in line[2]:
                    continue

                # Add file name and link to dict
                dja_catalog[line[2]] = line[3].split(">")[1].split("<")[0].strip()

        # Delete CSV and return dictionary
        temp_csv.close()
        os.unlink(temp_csv.name)
        return dja_catalog

    # Catch any errors and return empty dict
    except Exception as e:
        logger.error(f"Skipping: {e}.")
        return {}


def get_src_dest(
    runtime_settings: RuntimeSettings, dja_catalog: dict[str, str]
) -> dict[str, str]:
    """Get the sources and destinations for download for this program run, as a
    dict.

    Parameters
    ----------
    runtime_settings : RuntimeSettings
        Settings for this program run.
    dja_catalog : dict[str, str]
        Dict mapping filenames to their DJA download links.

    Returns
    -------
    dict[str, str]
        Dict containing local path destinations mapped by their DJA download
        link sources.
    """
    # Initialize empty dict to store source/destination pairs
    src_dest = {}

    # Iterate over each FICL in this program run
    for ficl in runtime_settings.ficls:
        # Get paths to each required input file
        input_segmap_path = settings.get_path(
            name="input_segmap", path_settings=runtime_settings.roots, ficl=ficl
        )
        input_catalog_path = settings.get_path(
            name="input_catalog", path_settings=runtime_settings.roots, ficl=ficl
        )
        exposure_path = settings.get_path(
            name="exposure", path_settings=runtime_settings.roots, ficl=ficl
        )
        science_path = settings.get_path(
            name="science", path_settings=runtime_settings.roots, ficl=ficl
        )
        weights_path = settings.get_path(
            name="weights", path_settings=runtime_settings.roots, ficl=ficl
        )

        # Get paths to alternate files where 'drc' is replaced by 'drz'
        alt_exposure_path = Path(str(exposure_path).replace("drc", "drz")).resolve()
        alt_science_path = Path(str(science_path).replace("drc", "drz")).resolve()
        alt_weights_path = Path(str(weights_path).replace("drc", "drz")).resolve()
        required_files = [
            input_segmap_path,
            input_catalog_path,
            exposure_path,
            science_path,
            weights_path,
            alt_exposure_path,
            alt_science_path,
            alt_weights_path,
        ]

        # Iterate over each required input file
        for required_file in required_files:
            # Add source/destination pair if file is found in download catalog
            if required_file.name in dja_catalog:
                src = dja_catalog[required_file.name]
                dest = str(required_file)
                src_dest[src] = dest

            # Add source/destination pair if zipped file is found in download
            # catalog
            elif required_file.name + ".gz" in dja_catalog:
                src = dja_catalog[required_file.name + ".gz"]
                dest = str(required_file) + ".gz"
                src_dest[src] = dest

            # Skip file if not found in download catalog
            else:
                if "drz" not in required_file.name:
                    logger.warning(
                        f"Skipping file '{required_file.name}': missing from DJA. "
                    )

    # Return source/destination pairs
    return src_dest


def unzip(path: Path) -> tuple[bool, float, float]:
    """Unzip a file with gzip.

    Parameters
    ----------
    path : Path
        Path to zipped file.

    Returns
    -------
    tuple[bool, float, float]
        Success of unzipping, compressed file size, uncompressed file size.
    """
    # Store original file size
    compressed_size = path.stat().st_size

    # Get path to which to write unzipped file
    input_fil_file_unzipped = Path(str(path)[:-3])

    # Try unzipping file with gzip
    try:
        with gzip.open(path, mode="rb") as zipped:
            with open(input_fil_file_unzipped, mode="wb") as unzipped:
                shutil.copyfileobj(zipped, unzipped)
        path.unlink()
    except Exception as e:
        logger.error(f"Skipping file '{path.name}': {e}.")
        return False, compressed_size, 0

    # Return file sizes
    uncompressed_size = input_fil_file_unzipped.stat().st_size
    return True, compressed_size, uncompressed_size


## Primary


def get_input(src_dest: dict[str, str]):
    """Download the required input files for a MorphFITS program run from the
    DJA archive.

    Parameters
    ----------
    src_dest : dict[str, str]
        Dict containing local path destinations mapped by their DJA download
        link sources.
    """
    logger.info(f"Downloading files: {len(src_dest)} files.")

    # Iterate over each source/destination pair for this initialize run
    num_files, total_file_size = 0, 0
    for src, dest in src_dest.items():
        # Try downloading from source and writing to destination
        try:
            # Skip files which already exist
            if (Path(dest).exists()) or (Path(dest[:-3]).exists()):
                num_files += 1
                raise FileExistsError("exists")

            # Download from source and display progress bar
            file_name = src.split("/")[-1]
            with DownloadProgressBar(
                unit="B",
                unit_scale=True,
                miniters=1,
                desc=file_name,
                leave=False,
            ) as t:
                request.urlretrieve(url=src, filename=dest, reporthook=t.update_to)

            # Log progress and append to file size
            file_size = Path(dest).stat().st_size
            total_file_size += file_size
            num_files += 1
            logger.debug(
                f"Downloaded file '{file_name}': "
                + f"{misc.get_str_from_file_size(file_size)} "
                + f"({num_files+1} / {len(src_dest)})."
            )

        # Catch errors downloading and skip to next pair
        except Exception as e:
            logger.debug(f"Skipping file '{Path(dest).name}': {e}.")

    # Log number of files downloaded
    if num_files > 0:
        logger.info(
            f"Downloaded {num_files} files: "
            + f"{misc.get_str_from_file_size(total_file_size)}."
        )
    else:
        logger.debug("Skipping: FICLs already initialized.")


def unzip_all(runtime_settings: RuntimeSettings):
    """Unzip all zipped FITS files under input root directory.

    Parameters
    ----------
    runtime_settings : RuntimeSettings
        Settings for this program run.
    """
    logger.info("Unzipping compressed input files.")

    # Initialize monitoring variables as 0
    num_files, total_compressed_size, total_uncompressed_size = 0, 0, 0

    # Iterate over each field subdirectory
    for field_dir in misc.get_subdirectories(runtime_settings.roots.input):
        # Skip PSFs subdirectory, which does not contain zipped files
        if field_dir.name == "psfs":
            continue

        # Iterate over each image version subdirectory
        for imver_dir in misc.get_subdirectories(field_dir):
            # Iterate over each file in image version subdirectory to unzip
            for input_fi_file in imver_dir.iterdir():
                # Skip unzipped files
                if ".fits.gz" not in input_fi_file.name:
                    continue

                # Unzip file and get back file sizes
                unzipped, compressed_size, uncompressed_size = unzip(input_fi_file)

                # Display status and save file sizes if successfully
                # unzipped, skip otherwise
                if unzipped:
                    logger.debug(
                        f"Unzipped file '{input_fi_file.name[:-3]}': "
                        + f"{misc.get_str_from_file_size(compressed_size)} -> "
                        + f"{misc.get_str_from_file_size(uncompressed_size)}."
                    )

                    # Store values to monitoring variables
                    total_compressed_size += compressed_size
                    total_uncompressed_size += uncompressed_size
                    num_files += 1

            # Iterate over each filter subdirectory
            for filter_dir in misc.get_subdirectories(imver_dir):
                # Iterate over each file in filter subdirectory to unzip
                for input_fil_file in filter_dir.iterdir():
                    # Skip unzipped files
                    if ".fits.gz" not in input_fil_file.name:
                        continue

                    # Unzip file and get back file sizes
                    unzipped, compressed_size, uncompressed_size = unzip(input_fil_file)

                    # Display status and save file sizes if successfully
                    # unzipped, skip otherwise
                    if unzipped:
                        logger.debug(
                            f"Unzipped file '{input_fil_file.name[:-3]}': "
                            + f"{misc.get_str_from_file_size(compressed_size)} -> "
                            + f"{misc.get_str_from_file_size(uncompressed_size)}."
                        )

                        # Store values to monitoring variables
                        total_compressed_size += compressed_size
                        total_uncompressed_size += uncompressed_size
                        num_files += 1

    # Log number of files and file sizes uncompressed
    if num_files > 0:
        logger.info(
            f"Unzipped {num_files} files: "
            + f"{misc.get_str_from_file_size(total_compressed_size)} -> "
            + f"{misc.get_str_from_file_size(total_uncompressed_size)}."
        )
    else:
        logger.debug("Skipping: missing compressed files.")
