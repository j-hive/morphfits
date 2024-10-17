"""Tests for the path utilities module.
"""

# Imports


from morphfits.config import MorphFITSConfig
from morphfits.utils import path


# Tests


def test_get_parameter(configuration: MorphFITSConfig):
    # Positive
    unresolved = [["input_root", "/foo/bar/", None], ["fields", None, configuration]]

    # Negative


def test_get_path_name():
    """Test `path.get_path_name` on valid and invalid path names."""
    # Positive
    ## Case 1
    valid_names = ["input_root", "input_catalog", "product_root", "product_ficlo"]
    for valid_name in valid_names:
        if path.get_path_name(valid_name) != valid_name:
            assert ValueError(
                f"Expected {valid_name}, got {path.get_path_name(valid_name)}."
            )

    ## Case 2
    valid_alt_names = {
        "input_root": "inputs",
        "input_catalog": "photometric catalog",
        "product_root": "products",
        "product_ficlo": "ficlo products",
    }
    for valid_name, valid_alt_name in valid_alt_names.items():
        if path.get_path_name(valid_alt_name) != valid_name:
            assert ValueError(
                f"Expected {valid_name} for {valid_alt_name}, "
                + f"got {path.get_path_name(valid_alt_name)}."
            )

    ## Case 3
    valid_space_names = {
        "input_root": "input root",
        "product_root": "product root",
        "product_ficlo": "product ficlo",
    }
    for valid_name, valid_space_name in valid_space_names.items():
        if path.get_path_name(valid_space_name) != valid_name:
            assert ValueError(
                f"Expected {valid_name} for {valid_space_name}, "
                + f"got {path.get_path_name(valid_space_name)}."
            )

    ## Case 4
    valid_suffix_names = {
        "input_root": "input root dir",
        "input_catalog": "input catalog file",
        "weight": "weight file",
        "product_ficlo": "product ficlo dir",
        "plot_galfit": "plot galfit file",
    }
    for valid_name, valid_suffix_name in valid_suffix_names.items():
        if path.get_path_name(valid_suffix_name) != valid_name:
            assert ValueError(
                f"Expected {valid_name} for {valid_suffix_name}, "
                + f"got {path.get_path_name(valid_suffix_name)}."
            )

    ## Case 5
    valid_unpluralized_names = {
        "input_psfs": "input psf dir",
        "input_images": "input image dir",
        "product_ficlo": "product ficlo dir",
    }
    for valid_name, valid_unpluralized_name in valid_unpluralized_names.items():
        if path.get_path_name(valid_unpluralized_name) != valid_name:
            assert ValueError(
                f"Expected {valid_name} for {valid_unpluralized_name}, "
                + f"got {path.get_path_name(valid_unpluralized_name)}."
            )

    # Negative
    ## TypeError
    invalid_names = [0, 0.0, True]
    for invalid_name in invalid_names:
        try:
            path.get_path_name(invalid_name)
            assert AssertionError(
                f"Name {invalid_name} of type "
                + f"{type(invalid_name)} expected to raise TypeError."
            )
        except TypeError:
            continue

    ## ValueError
    invalid_names = ["hey", "sigma image cutout", "sigma_root", "VISDIR"]
    for invalid_name in invalid_names:
        try:
            path.get_path_name(invalid_name)
            assert AssertionError(f"Name {invalid_name} expected to raise ValueError.")
        except ValueError:
            continue
