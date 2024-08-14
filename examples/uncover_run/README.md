# MorphFITS
MorphFITS is a morphology fitter wrapper program based in Python.

This is a usage guide. We will be using the following FICLO as an example.
|Property|Example|
|:---|:---|
|`field`|`abell2744clu`|
|`image_version`|`grizli-v7.2`|
|`catalog_version`|`dja-v7.2`|
|`filter`|`f150w-clear`|
|`object`|`10651`|

For further reading, please refer to the [main documentation](../../README.md),
or the [data structure specifications](../../data/README.md). 

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
poetry run morphfits download --config-path=examples/uncover_run/config.yaml
```
to download and unzip JWST data, which will take several
minutes and ~`13GB`. Then, download [the simulated PSF for the filter
here](https://stsci.app.box.com/v/jwst-simulated-psf-library/file/1025339832742),
and move it to `examples/uncover_run/morphfits_root/input/psfs/`, for example via
```
mv [download folder]/PSF_NIRCam_in_flight_opd_filter_F200W.fits ./examples/single_ficlo/morphfits_root/input/psfs
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
To run MorphFITS, use `config.yaml` as found in this directory, modifying the `galfit_path` field. Then, navigate to the project directory `morphfits/`
and run
```
poetry run morphfits galwrap --config-path=examples/uncover_run/config.yaml
```
MorphFITS will run Lamiya's GalWrap pipeline over the FICLO found in the
configuration file. 
1. It will generate a stamp, sigma map, cropped PSF, mask, and GALFIT feedfile
to `examples/uncover_run/morphfits_root/products`, over which GALFIT is run. 
2. The program will then output a log, model FITS, and visualization to
`examples/uncover_run/morphfits_root/output`. 
3. Finally, the run log, parameters, and run flags will be recorded to
`examples/uncover_run/morphfits_root/runs`. 

Note that subsequent runs will overwrite files in
`products` and `output`, so files of interest are recommended to be moved
elsewhere. 

For more help, refer to [the primary documentation](../../README.md), or run
```
poetry run morphfits galwrap --help
```
