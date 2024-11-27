# MorphFITS
MorphFITS is a morphology fitter wrapper program based in Python.

This is a usage guide. We will be using the following FICLOs as an example.
|Property|Example|
|:---|:---|
|`field`|`abell2744clu`|
|`image_version`|`grizli-v7.2`|
|`catalog_version`|`dja-v7.2`|
|`filter`|`f150w-clear`|
|`objects`|`8638` to `8644`|

For further reading, please refer to the [main documentation](../../README.md),
or the [data standards](../../data/README.md). 

# Setup & Installation
To install, run
```
git clone git@github.com:j-hive/morphfits.git
cd morphfits
poetry install
```
Download the GALFIT binary for your system from [the official
page](https://users.obs.carnegiescience.edu/peng/work/galfit/galfit.html), and
move it to the location of your choosing.

Ensure you're in the root directory, `morphfits`, and run

```
poetry run morphfits initialize -c examples/batch_mode/settings.yaml
```

to download and unzip JWST data, which will take several
minutes and ~`13GB`. 

*outdated but waiting to upload new PSFs* 

Then, download [the simulated PSF for the filter
here](https://stsci.app.box.com/v/jwst-simulated-psf-library/file/1025339832742),
and move it to `examples/uncover_run/morphfits_root/input/psfs/`, for example
via

```
mv [download folder]/PSF_NIRCam_in_flight_opd_filter_F150W.fits examples/batch_mode/morphfits_root/input/psfs
```

The input directory should now be structured as such.
```
input/
├── abell2744clu
│   └── grizli-v7.2
│       ├── abell2744clu-grizli-v7.2-fix_phot_apcorr.fits
│       ├── abell2744clu-grizli-v7.2-ir_seg.fits
│       └── f150w-clear
│           ├── abell2744clu-grizli-v7.2-f150w-clear_drc_exp.fits
│           ├── abell2744clu-grizli-v7.2-f150w-clear_drc_sci.fits
│           └── abell2744clu-grizli-v7.2-f150w-clear_drc_wht.fits
└── psfs
    └── PSF_NIRCam_in_flight_opd_filter_F150W.fits
```

# Usage
To run MorphFITS, run
```
poetry run morphfits galwrap -c examples/batch_mode/settings.yaml -g [path/to/galfit]
```
replacing `[path/to/galfit]` with a path to the GALFIT file you downloaded and
moved.

Note the settings indicate `batch_n_process=10000` and `batch_process_id=12`,
which will run the 13th process in a batch of 10000.

MorphFITS will run Lamiya's GalWrap pipeline over the FICLOs found in the
configuration file, and generate the following files.
1. Product files @ `morphfits_root/products`
    1. stamp FITS
    2. sigma map FITS
    3. PSF crop FITS
    4. mask FITS
    5. GALFIT feedfiles
2. Output files @ `morphfits_root/output`
    1. merge catalog CSV (full catalog)
    2. morphology catalog CSV (per-FIC catalog)
    3. histogram PNG
    4. model FITS
    5. fit LOG
3. Run files @ `morphfits_root/runs`
    1. run catalog CSV
    2. histogram PNG
    3. program LOG
    4. settings YAML

Note that subsequent runs will overwrite files in
`products` and `output`, so files of interest are recommended to be moved
elsewhere. 

For more help, refer to [the primary documentation](../../README.md), or run
```
poetry run morphfits galwrap --help
```
