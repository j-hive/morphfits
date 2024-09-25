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


LIST_DJA_ENDPOINT = "index.csv"
"""Endpoint for filelist CSV.
"""


LIST_DJA_PATH = ROOT / "index.csv"
"""Path to downloaded filelist CSV.
"""


logger = logging.getLogger("DOWNLOAD")
"""Logger object for this module.
"""


# Functions


def get_dja_list() -> dict[str, dict]:
    """Get the DJA file list as a dictionary with details indexed by filename.

    Returns
    -------
    dict[str, dict]
        DJA file list as a dictionary with details indexed by filename.
    """

    # Download file list CSV from DJA index
    logger.info("Downloading file list from DJA.")
    request.urlretrieve(
        url=BASE_URL + "/" + LIST_DJA_ENDPOINT, filename=str(LIST_DJA_PATH)
    )

    # Add each file and corresponding details to a dictionary
    list_dja = {}
    with open(LIST_DJA_PATH, mode="r", newline="") as csv_file:
        reader = csv.reader(csv_file)
        skipped_headers = False
        for line in reader:
            if not skipped_headers:
                skipped_headers = True
                continue
            if "fits" not in line[2]:
                continue
            list_dja[line[2]] = {
                "datetime": line[0],
                "filesize": float(line[1]),
                "download": line[3].split(">")[1].split("<")[0].strip(),
            }

    # Delete CSV and return dictionary
    LIST_DJA_PATH.unlink()
    return list_dja


def get_download_list(
    morphfits_config: config.MorphFITSConfig,
    list_dja: dict[str, dict],
    overwrite: bool = False,
) -> list[tuple[str, str]]:
    """Get a list of the required filenames for the given FICLs.

    Parameters
    ----------
    morphfits_config : config.MorphFITSConfig
        Configuration object for this program run.
    list_dja : dict[str, dict]
        DJA file list as a dictionary with details indexed by filename.
    overwrite : bool, optional
        Overwrite existing input files with new downloads, by default False.

    Returns
    -------
    list[tuple[str, str]]
        List of source URLs and their corresponding destination paths.
    """

    # Get filenames of all required files from configuration
    logger.info("Getting list of files to download from configuration.")
    list_download = []
    for ficl in morphfits_config.get_FICLs(pre_input=True):
        # Get required destination paths for FICL
        required_paths: list[Path] = [
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

        # Search for required files from DJA list
        for file in list_dja:
            for required_path in required_paths:
                if required_path.name in file:
                    ## Skip file if already in download list
                    source = BASE_URL + "/" + file
                    destination = "/".join(str(required_path).split("/")[:-1] + [file])
                    list_sources = [download[0] for download in list_download]
                    if source in list_sources:
                        continue

                    ## Skip existing files if not overwrite mode
                    zip_path = Path(str(required_path) + ".gz").resolve()
                    if required_path.exists() and not overwrite:
                        logger.info(
                            f"Skipping existing unzipped file '{required_path.name}'."
                        )
                        continue
                    if zip_path.exists() and not overwrite:
                        logger.info(
                            f"Skipping existing compressed file '{zip_path.name}'."
                        )
                        continue

                    ## Add filename otherwise as source/destination pair
                    list_download.append((source, destination))

    return list_download


def get_zip_list(node: Path) -> list[Path]:
    """Get a list of paths to files with the extension '.gz' under a directory.

    Parameters
    ----------
    node : Path
        Path to current directory or file.

    Returns
    -------
    list[Path]
        List of paths to '.gz' input files.
    """
    if ".fits" in node.name:
        if ".gz" in node.name:
            return [node]
        else:
            return []
    elif node.is_dir():
        list_zip = []
        for child_node in node.iterdir():
            list_zip += get_zip_list(node=child_node)
        return list_zip
    else:
        return []


def get_size_str(size_file: float) -> str:
    """Get a filesize, in MB, as a string with its appropriate size appended.

    Parameters
    ----------
    size_file : float
        Size of file, in MB.

    Returns
    -------
    str
        Size rounded to 2 decimal places, with its appropriate size unit.
    """
    size_byte = 1024
    if size_file >= size_byte**2:
        size_str = str(round(size_file / size_byte**2, 2)) + " TB"
    elif size_file >= size_byte:
        size_str = str(round(size_file / size_byte, 2)) + " GB"
    else:
        size_str = str(round(size_file, 2)) + " MB"
    return size_str


def download_files(
    list_download: list[tuple[str, str]],
    list_dja: dict[str, dict],
):
    """Download files from DJA.

    Parameters
    ----------
    list_download : list[tuple[str, str]]
        List of source URLs to download from, and their corresponding
        destination paths.
    list_dja : dict[str, dict]
        DJA file list as dictionary indexed by filename.
    """
    # Display size of download
    n_download = len(list_download)
    megabytes = 0
    for file in list_download:
        filename = file[0].split("/")[-1]
        megabytes += list_dja[filename]["filesize"]
    size_file = get_size_str(size_file=megabytes)
    logger.info(f"Downloading {n_download} file(s) ({size_file}).")

    # Download files
    for url, out in tqdm(list_download, unit="file", leave=False):
        filename = url.split("/")[-1]
        with DownloadProgressBar(
            unit="B",
            unit_scale=True,
            miniters=1,
            desc=filename,
            leave=False,
        ) as t:
            request.urlretrieve(url, filename=out, reporthook=t.update_to)
        size_file = get_size_str(size_file=list_dja[filename]["filesize"])
        logger.info(f"Downloaded '{filename}' ({size_file}).")


def unzip_files(morphfits_config: config.MorphFITSConfig):
    """Unzip downloaded files.

    Parameters
    ----------
    morphfits_config : MorphFITSConfig
        Configuration object for this program run.
    """
    # Get list of paths to zipped files
    logger.info("Getting list of files to unzip.")
    list_zip = get_zip_list(node=morphfits_config.input_root)

    # Get list of zipped files to unzip and return if there are none
    n_zip = len(list_zip)
    if n_zip == 0:
        logger.info("No zipped files to unzip.")
        return

    # Iterate over each zipped file
    total_original, total_new = 0, 0
    logger.info(f"Unzipping {n_zip} file(s).")
    for zip_path in tqdm(list_zip, unit="file", leave=False):
        unzip_path = Path(str(zip_path)[:-3]).resolve()

        ## Save original size
        size_original = zip_path.stat().st_size / 1024**2
        total_original += size_original

        ## Unzip file and delete zip
        with gzip.open(zip_path, mode="rb") as zipped:
            with open(unzip_path, mode="wb") as unzipped:
                shutil.copyfileobj(zipped, unzipped)
        zip_path.unlink()

        ## Save new size
        size_new = unzip_path.stat().st_size / 1024**2
        total_new += size_new

        # Display decompression size difference
        str_original = get_size_str(size_file=size_original)
        str_new = get_size_str(size_file=size_new)
        logger.info(f"Unzipped '{unzip_path.name}' ({str_original} -> {str_new}).")

    # Display total decompression size difference
    str_total_original = get_size_str(size_file=total_original)
    str_total_new = get_size_str(size_file=total_new)
    logger.info(f"Unzipped {n_zip} files ({str_total_original} -> {str_total_new}).")


def main(
    morphfits_config: config.MorphFITSConfig,
    skip_download: bool = False,
    skip_unzip: bool = False,
    overwrite: bool = False,
):
    """Identify, download, and unzip files from the DJA archive for given FICLs.

    Parameters
    ----------
    morphfits_config : config.MorphFITSConfig
        Configuration object for this program run.
    skip_download : bool, optional
        Skip downloading files, i.e. only unzip files, by default False.
    skip_unzip : bool, optional
        Skip unzipping files, i.e. only download files, by default False.
    overwrite : bool, optional
        Overwrite existing input files with new downloads, by default False.
    """
    list_dja = get_dja_list()
    list_download = get_download_list(
        morphfits_config=morphfits_config, list_dja=list_dja, overwrite=overwrite
    )
    if not skip_download:
        if len(list_download) > 0:
            download_files(list_download=list_download, list_dja=list_dja)
        else:
            logger.info("No files to download.")
    if not skip_unzip:
        unzip_files(morphfits_config=morphfits_config)
