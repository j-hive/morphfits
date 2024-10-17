"""Miscellaneous utility functions.
"""

# Imports


import numpy as np


# Functions


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
