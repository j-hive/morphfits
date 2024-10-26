"""Miscellaneous utility functions.
"""

# Imports


from pathlib import Path
from datetime import datetime

import numpy as np


# Functions


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
    if isinstance(path_like, str):
        return Path(path_like).resolve()
    else:
        return path_like.resolve()


def get_subdirectories(path: Path) -> list[Path]:
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


def get_str_from_datetime(date_time: datetime) -> str:
    """Get a string representation of a datetime, in the format
    'YYYYMMDDTHHMMSS'.

    Parameters
    ----------
    date_time : datetime
        Datetime to convert to str.

    Returns
    -------
    str
        String representation of datetime.
    """
    return date_time.strftime("%Y%m%dT%H%M%S")


def get_str_from_run_number(run_number: int) -> str:
    """Get a string representation of a run number, in the format '01', with one
    leading zero if the number is one digit.

    Parameters
    ----------
    run_number : int
        Number to convert to string.

    Returns
    -------
    str
        Run number as string with leading zero.
    """
    return str(run_number).rjust(2, "0")


def get_unique_batch_limits(
    process_id: int, n_process: int, n_items: int
) -> tuple[int, int]:
    """Produce the minimum and maximum items for a process
    given the number of items to process, the number of processes,
    and the process id. The min and max indices will be unique based
    on only these three parameters.

    Parameters
    ----------
    process_id : int
        The process id (ranging from 0 to n_process-1)
    n_process : int
        The number of processes
    n_items : int
        The number of items to process

    Returns
    -------
    tuple[int, int]
        The min and max index to process, of the form [min, max)

    """

    # Checking if valid process
    if process_id >= n_process:
        raise ValueError(
            f"process_id ({process_id}) can not be greater than {n_process - 1}"
        )

    # Setting number of items in this process
    n_items_process = n_items // n_process
    if process_id < (n_items % n_process):
        n_items_process += 1

    # Setting Start, Stop Indices
    start_index = process_id * (n_items // n_process) + min(
        process_id, n_items % n_process
    )
    stop_index = start_index + n_items_process

    return start_index, stop_index


def get_unique(items: list) -> list:
    """Get the unique elements in a list of elements, as a sorted list.

    Parameters
    ----------
    items : list
        List of elements to be sorted.

    Returns
    -------
    list
        Sorted list of unique elements.
    """
    not_nan_items = []

    # Remove any NaNs from list
    for item in items:
        if isinstance(item, float) and np.isnan(item):
            continue
        else:
            not_nan_items.append(item)

    # Return unique items in list, sorted
    return sorted(set(not_nan_items))
