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
import shutil
from urllib import request
from pathlib import Path
import csv

from tqdm import tqdm

from morphfits import config, paths, ROOT


# Constants


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


BASE_URL = "https://s3.amazonaws.com/grizli-v2/JwstMosaics/v7"
"""URL to DJA v7 Mosaics archive.
"""


FILE_LIST_ENDPOINT = "index.csv"
"""Endpoint for filelist CSV.
"""


FILE_LIST_PATH = ROOT / "index.csv"
"""Path to downloaded filelist CSV.
"""


logger = logging.getLogger("DOWNLOAD")
"""Logger object for this module.
"""


# Functions


def get_file_list() -> dict[str, dict]:
    """Get the DJA file list as a dictionary indexed by filename.

    Returns
    -------
    dict[str, dict]
        DJA file list as a dictionary indexed by filename.
    """

    # Download file list CSV from DJA index
    logger.info("Downloading file list.")
    request.urlretrieve(
        url=BASE_URL + "/" + FILE_LIST_ENDPOINT, filename=str(FILE_LIST_PATH)
    )

    # Add each file and corresponding details to a dictionary
    file_list = {}
    with open(FILE_LIST_PATH, mode="r", newline="") as csv_file:
        reader = csv.reader(csv_file)
        skipped_headers = False
        for line in reader:
            if not skipped_headers:
                skipped_headers = True
                continue
            if "fits" not in line[2]:
                continue
            file_list[line[2]] = {
                "datetime": line[0],
                "filesize": float(line[1]),
                "download": line[3].split(">")[1].split("<")[0].strip(),
            }

    # Delete CSV and return dictionary
    FILE_LIST_PATH.unlink()
    return file_list


def get_src_dest(
    morphfits_config: config.MorphFITSConfig, file_list: dict[str, dict]
) -> list[tuple[str, str]]:
    """Get the required filenames for the given FICLs.

    Parameters
    ----------
    morphfits_config : config.MorphFITSConfig
        Configuration object for this program run.
    file_list : dict[str, dict]
        DJA file list as a dictionary indexed by filename.

    Returns
    -------
    list[tuple[str, str]]
        List of source URLs and their corresponding destination paths.
    """

    # Get filenames of all required files from configuration
    logger.info("Getting list of files to download.")
    filenames = []
    for ficl in morphfits_config.get_FICLs(pre_input=True):
        # Get required destination paths for FICL
        required_paths = [
            paths.get_path(
                "segmap",
                input_root=morphfits_config.input_root,
                field=ficl.field,
                image_version=ficl.image_version,
            ),
            paths.get_path(
                "catalog",
                input_root=morphfits_config.input_root,
                field=ficl.field,
                image_version=ficl.image_version,
            ),
            paths.get_path(
                "exposure",
                input_root=morphfits_config.input_root,
                field=ficl.field,
                image_version=ficl.image_version,
                filter=ficl.filter,
            ),
            Path(
                str(
                    paths.get_path(
                        "exposure",
                        input_root=morphfits_config.input_root,
                        field=ficl.field,
                        image_version=ficl.image_version,
                        filter=ficl.filter,
                    )
                ).replace("drc", "drz")
            ).resolve(),
            paths.get_path(
                "science",
                input_root=morphfits_config.input_root,
                field=ficl.field,
                image_version=ficl.image_version,
                filter=ficl.filter,
            ),
            Path(
                str(
                    paths.get_path(
                        "science",
                        input_root=morphfits_config.input_root,
                        field=ficl.field,
                        image_version=ficl.image_version,
                        filter=ficl.filter,
                    )
                ).replace("drc", "drz")
            ).resolve(),
            paths.get_path(
                "weights",
                input_root=morphfits_config.input_root,
                field=ficl.field,
                image_version=ficl.image_version,
                filter=ficl.filter,
            ),
            Path(
                str(
                    paths.get_path(
                        "weights",
                        input_root=morphfits_config.input_root,
                        field=ficl.field,
                        image_version=ficl.image_version,
                        filter=ficl.filter,
                    )
                ).replace("drc", "drz")
            ).resolve(),
        ]

        # Search required filenames from file list dictionary
        for file in file_list:
            for required_path in required_paths:
                if required_path.name in file:
                    filenames.append(
                        (
                            BASE_URL + "/" + file,
                            "/".join(str(required_path).split("/")[:-1] + [file]),
                        )
                    )

    return filenames


def get_files(
    src_dest: list[tuple[str, str]],
    file_list: dict[str, dict],
):
    """Download files from DJA.

    Parameters
    ----------
    src_dest : list[tuple[str, str]]
        List of source URLs to download from, and their corresponding
        destination paths.
    file_list : dict[str, dict]
        DJA file list as dictionary indexed by filename.
    """
    # Display size of download
    megabytes = 0
    for file in src_dest:
        filename = file[0].split("/")[-1]
        megabytes += file_list[filename]["filesize"]
    if megabytes >= 1024**2:
        file_size = str(round(megabytes / 1024**2, 2)) + " TB"
    elif megabytes >= 1024:
        file_size = str(round(megabytes / 1024, 2)) + " GB"
    else:
        file_size = str(round(megabytes, 2)) + " MB"
    logger.info(f"Downloading {len(src_dest)} files ({file_size}).")

    # Download files
    for url, out in tqdm(src_dest, unit="file", leave=False):
        with DownloadProgressBar(
            unit="B",
            unit_scale=True,
            miniters=1,
            desc=url.split("/")[-1],
            leave=False,
        ) as t:
            request.urlretrieve(url, filename=out, reporthook=t.update_to)


def unzip_files(
    src_dest: list[tuple[str, str]],
    file_list: dict[str, dict],
):
    """Unzip downloaded files.

    Parameters
    ----------
    src_dest : list[tuple[str, Path]]
        List of source URLs to download from, and their corresponding
        destination paths.
    file_list : dict[str, dict]
        DJA file list as dictionary indexed by filename.
    """

    # Unzip files
    logger.info("Unzipping files.")
    for url, out in tqdm(src_dest, unit="file", leave=False):
        if ".gz" not in out:
            continue
        with gzip.open(out, mode="rb") as zipped:
            with open(out[:-3], mode="wb") as unzipped:
                shutil.copyfileobj(zipped, unzipped)
        Path(out).resolve().unlink()

    # Display size of download
    old_mb, new_kb = 0, 0
    for src, dest in src_dest:
        filename = src.split("/")[-1]
        old_mb += file_list[filename]["filesize"]
        if ".gz" in filename:
            new_kb += Path(dest[:-3]).stat().st_size / 1024
        else:
            new_kb += old_mb

    if old_mb >= 1024**2:
        old_size = str(round(old_mb / 1024**2, 2)) + " TB"
    elif old_mb >= 1024:
        old_size = str(round(old_mb / 1024, 2)) + " GB"
    else:
        old_size = str(round(old_mb, 2)) + " MB"

    if new_kb >= 1024**3:
        new_size = str(round(new_kb / 1024**3, 2)) + " TB"
    elif new_kb >= 1024**2:
        new_size = str(round(new_kb / 1024**2, 2)) + " GB"
    else:
        new_size = str(round(new_kb / 1024, 2)) + " MB"

    logger.info(f"Unzipped files ({old_size} -> {new_size}).")


def main(morphfits_config: config.MorphFITSConfig):
    """Identify, download, and unzip files from the DJA archive for given FICLs.

    Parameters
    ----------
    morphfits_config : config.MorphFITSConfig
        Configuration object for this program run.
    """
    file_list = get_file_list()
    src_dest = get_src_dest(morphfits_config=morphfits_config, file_list=file_list)
    get_files(src_dest=src_dest, file_list=file_list)
    unzip_files(src_dest=src_dest, file_list=file_list)
