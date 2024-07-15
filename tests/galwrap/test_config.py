"""Test the GalWrap configuration module.
"""

# Imports


from morphfits.wrappers.setup import config


# Tests


def test_create_config(galwrap_config: config.GalWrapConfig):
    print(galwrap_config)
    assert True


def test_resolve_path_name():
    # Successes
    ## Valid names for each case
    valid_names = [
        ["output_stamps", "PHOTOMETRY_RMS", "file_segmap"],
        ["stamps", "input rms", "segmentation FILE"],
        ["output_stamp", "photometry_rm", "oUtPuT_pSf"],
        ["output stamps", "PHOTOMETRY RmS", "file segmap"],
        ["stamps output", "rms photometry", "psfs output"],
        ["stamp output", "rm output", "psf output"],
    ]
    for case in range(len(valid_names)):
        case_names = valid_names[case]
        for name in case_names:
            try:
                config.resolve_path_name(name)
            except:
                raise AssertionError(
                    f"Path name '{name}' should be valid for case {case+    1}."
                )

    # Failures
    ## Name Type Invalid
    wrong_type_names = [5, 5.0, True]
    for name in wrong_type_names:
        try:
            config.resolve_path_name(name)
            raise AssertionError(
                f"False positive. Name of type {type(name)}"
                + f" accepted in `config.resolve_path_name`."
            )
        except TypeError:
            continue

    ## Name Unrecognized
    unrecognized_names = ["Bill", "output_output", "stamp_output", "stamps_output"]
    for name in unrecognized_names:
        try:
            config.resolve_path_name(name)
            raise AssertionError(
                f"False positive. Unrecognized name '{name}'"
                + f" accepted in `config.resolve_path_name`."
            )
        except ValueError:
            continue
