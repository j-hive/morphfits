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

Next, setup a directory structure in the location of your choosing by running
```
cd ...
mkdir -p morphfits_root/input/abell2744clu/grizli-v7.2/f150w-clear
mkdir morphfits_root/input/psfs
```

Each FICL requires six files - the simulated PSF, segmentation map, photometric
catalog, exposure map, science frame, and weights map. To download these files
to their corresponding locations, run
```
cd morphfits_root/input/abell2744clu/grizli-v7.2
wget https://s3.amazonaws.com/grizli-v2/JwstMosaics/v7/abell2744clu-grizli-v7.2-fix_phot_apcorr.fits
wget https://s3.amazonaws.com/grizli-v2/JwstMosaics/v7/abell2744clu-grizli-v7.2-ir_seg.fits.gz
gzip -dv *
cd f150w-clear
wget https://s3.amazonaws.com/grizli-v2/JwstMosaics/v7/abell2744clu-grizli-v7.2-f150w-clear_drc_exp.fits.gz
wget https://s3.amazonaws.com/grizli-v2/JwstMosaics/v7/abell2744clu-grizli-v7.2-f150w-clear_drc_sci.fits.gz
wget https://s3.amazonaws.com/grizli-v2/JwstMosaics/v7/abell2744clu-grizli-v7.2-f150w-clear_drc_wht.fits.gz
gzip -dv *
```
Then, download the PSF from
https://stsci.app.box.com/v/jwst-simulated-psf-library/file/1025347215541, and
move it to `.../morphfits_root/input/psfs`.

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
To run MorphFITS, write a `config.yaml` file in `morphfits_root` with the
following content.
```
input_root: .../morphfits_root/input
fields:
  - "abell2744clu"
image_versions:
  - "grizli-v7.2"
catalog_versions:
  - "dja-v7.2"
filters:
  - f150w-clear
objects:
  - 10651
```
Or, use `config.yaml` as found in this directory, modifying the `input_root`
field. Then, navigate to the project directory `...morphfits` and run
```
poetry run morphfits galwrap --config-path=[path to config]
```
MorphFITS will run Lamiya's GalWrap pipeline over the FICLO found in the
configuration file. 
1. It will generate a stamp, sigma map, cropped PSF, mask, and GALFIT feedfile
to `.../morphfits_root/products`, over which GALFIT is run. 
2. The program will then output a log, model FITS, and visualization to
`.../morphfits_root/output`. 
3. Finally, the run log, parameters, and run flags will be recorded to
`.../morphfits_root/runs`. 

Please find attached the corresponding state of `morphfits_root` following
this guide, for reference. Note that subsequent runs will overwrite files in
`products` and `output`, so files of interest are recommended to be moved
elsewhere. 

For more help, see the official documentation, send me a message, or run
```
poetry run morphfits galwrap --help
```
