"""Test the GalWrap paths utilities module.
"""

# Imports


from morphfits.config import MorphFITSConfig
from morphfits import paths


# Tests


def test_get_parameter(configuration: MorphFITSConfig):
    # Positive
    unresolved = [["input_root", "/foo/bar/", None], ["fields", None, configuration]]

    # Negative


def test_get_path_name():
    """Test `path.get_path_name` on valid and invalid path names."""
    # Positive
    ## Case 1
    valid_names = ["input_root", "catalog", "product_root", "ficlo_products", "plots"]
    for valid_name in valid_names:
        if paths.get_path_name(valid_name) != valid_name:
            assert ValueError(
                f"Expected {valid_name}, got {paths.get_path_name(valid_name)}."
            )

    ## Case 2
    valid_alt_names = {
        "input_root": "inputs",
        "catalog": "photometric catalog",
        "product_root": "products",
        "ficlo_products": "product ficlo",
        "plots": "viz",
    }
    for valid_name, valid_alt_name in valid_alt_names.items():
        if paths.get_path_name(valid_alt_name) != valid_name:
            assert ValueError(
                f"Expected {valid_name} for {valid_alt_name}, "
                + f"got {paths.get_path_name(valid_alt_name)}."
            )

    ## Case 3
    valid_space_names = {
        "input_root": "input root",
        "product_root": "product root",
        "ficlo_products": "ficlo products",
    }
    for valid_name, valid_space_name in valid_space_names.items():
        if paths.get_path_name(valid_space_name) != valid_name:
            assert ValueError(
                f"Expected {valid_name} for {valid_space_name}, "
                + f"got {paths.get_path_name(valid_space_name)}."
            )

    ## Case 4
    valid_suffix_names = {
        "input_root": "input root dir",
        "catalog": "catalog file",
        "weight": "weight file",
        "ficlo_products": "ficlo products dir",
        "galfit_plot": "galfit plot file",
    }
    for valid_name, valid_suffix_name in valid_suffix_names.items():
        if paths.get_path_name(valid_suffix_name) != valid_name:
            assert ValueError(
                f"Expected {valid_name} for {valid_suffix_name}, "
                + f"got {paths.get_path_name(valid_suffix_name)}."
            )

    ## Case 5
    valid_unpluralized_names = {
        "input_psfs": "input psf dir",
        "input_images": "input image dir",
        "ficlo_products": "ficlo product dir",
        "plots": "plot dir",
    }
    for valid_name, valid_unpluralized_name in valid_unpluralized_names.items():
        if paths.get_path_name(valid_unpluralized_name) != valid_name:
            assert ValueError(
                f"Expected {valid_name} for {valid_unpluralized_name}, "
                + f"got {paths.get_path_name(valid_unpluralized_name)}."
            )

    # Negative
    ## TypeError
    invalid_names = [0, 0.0, True]
    for invalid_name in invalid_names:
        try:
            paths.get_path_name(invalid_name)
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
            paths.get_path_name(invalid_name)
            assert AssertionError(f"Name {invalid_name} expected to raise ValueError.")
        except ValueError:
            continue
