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


## Secondary


## Primary


def get_dja_catalog() -> dict[str, str]:
    logger.info("Downloading file list from DJA.")

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
        logger.error(f"Skipped downloading DJA file list - {e}.")
        return {}


def get_src_dest(
    runtime_settings: RuntimeSettings, dja_catalog: dict[str, str]
) -> dict[str, str]:
    #
    src_dest = {}

    #
    for ficl in runtime_settings.ficls:
        #
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

        #
        for required_file in required_files:
            #
            if required_file.name in dja_catalog:
                src = dja_catalog[required_file.name]
                dest = str(required_file)
                src_dest[src] = dest

            #
            elif required_file.name + ".gz" in dja_catalog:
                src = dja_catalog[required_file.name + ".gz"]
                dest = str(required_file) + ".gz"
                src_dest[src] = dest

            #
            else:
                if "drz" not in required_file.name:
                    logger.warning(
                        f"Skipping download for {required_file.name} "
                        + "- failed to locate in DJA catalog."
                    )

    #
    return src_dest


def get_input(src_dest: dict[str, str]):
    logger.info(f"Downloading {len(src_dest)} files.")

    #
    num_files, total_file_size = 0, 0
    for src, dest in src_dest.items():
        #
        try:
            #
            if (Path(dest).exists()) or (Path(dest[:-3]).exists()):
                num_files += 1
                raise FileExistsError("exists")

            #
            file_name = src.split("/")[-1]
            with DownloadProgressBar(
                unit="B",
                unit_scale=True,
                miniters=1,
                desc=file_name,
                leave=False,
            ) as t:
                request.urlretrieve(url=src, filename=dest, reporthook=t.update_to)

            #
            file_size = Path(dest).stat().st_size
            total_file_size += file_size
            num_files += 1
            logger.debug(
                f"Downloaded '{file_name}' "
                + f"({misc.get_file_size_str(file_size)}) "
                + f"({num_files+1}/{len(src_dest)})."
            )

        #
        except Exception as e:
            logger.debug(f"Skipping downloading '{Path(dest).name} - {e}.")

    #
    if num_files > 0:
        logger.info(
            f"Downloaded {num_files} files "
            + f"({misc.get_file_size_str(total_file_size)})."
        )
    else:
        logger.info("Skipping downloading - FICLs already initialized.")


def unzip(runtime_settings: RuntimeSettings):
    logger.info("Unzipping zipped input files.")

    #
    num_files, total_compressed_size, total_uncompressed_size = 0, 0, 0

    #
    for field_dir in misc.get_subdirectories(runtime_settings.roots.input):
        #
        if field_dir.name == "psfs":
            continue

        #
        for imver_dir in misc.get_subdirectories(field_dir):
            #
            for filter_dir in misc.get_subdirectories(imver_dir):
                #
                for input_fil_file in filter_dir.iterdir():
                    #
                    if ".fits.gz" not in input_fil_file.name:
                        continue

                    #
                    compressed_size = input_fil_file.stat().st_size

                    #
                    input_fil_file_unzipped = Path(str(input_fil_file)[:-3])

                    #
                    try:
                        with gzip.open(input_fil_file, mode="rb") as zipped:
                            with open(input_fil_file_unzipped, mode="wb") as unzipped:
                                shutil.copyfileobj(zipped, unzipped)
                        input_fil_file.unlink()
                    except Exception as e:
                        logger.error(f"Skipping unzipping {input_fil_file.name} - {e}.")

                    #
                    uncompressed_size = input_fil_file_unzipped.stat().st_size
                    logger.info(
                        f"Unzipped '{input_fil_file_unzipped.name} "
                        + f"({misc.get_file_size_str(compressed_size)} -> "
                        + f"{misc.get_file_size_str(uncompressed_size)})."
                    )

                    #
                    total_compressed_size += compressed_size
                    total_uncompressed_size += uncompressed_size
                    num_files += 1

    #
    if num_files > 0:
        logger.info(
            f"Unzipped {num_files} files "
            + f"({misc.get_file_size_str(total_compressed_size)} -> "
            + f"{misc.get_file_size_str(total_uncompressed_size)})."
        )
    else:
        logger.info("No zipped input files to unzip.")
