# MorphFITS
MorphFITS is a morphology fitter program for Python.

In its current iteration, it contains GalWrap, a wrapper for GALFIT, for JWST data.

# Installation
To install this program, clone this repository, navigate to it, and install via
[Poetry](https://python-poetry.org/docs/).
```
git clone git@github.com:j-hive/morphfits.git
cd morphfits
poetry install
```

# Quickstart
To see what the directory structure, the products and output, and the run looks
like, use one of the settings from [the examples directory](./examples/). For
example, for the field `abell2744clu`, image version `grizli-v7.2`, catalog
version `dja-v7.2`, filter `f200w`, and object `4215`, using GALFIT, run

```
sh ./examples/single_ficlo/setup.sh
```
in a bash environment to download JWST data, which will take several minutes.
Then, download [the simulated PSF for the filter
here](https://stsci.app.box.com/v/jwst-simulated-psf-library/file/1025339832742),
and move it to the appropriate location..
```
mv /mnt/c/Users/[Windows Username]/Downloads/PSF_NIRCam_in_flight_opd_filter_F200W.fits morphfits/examples/single_ficlo/morphfits_root/input/psfs
```
Then, run 
```
poetry install
poetry run morphfits galwrap --config-path=./examples/single_ficlo/config.yaml
```
and MorphFITS will run GalWrap, a wrapper for GALFIT, over the FICLOs found in `config.yaml`. The outputs can be
found at `examples/single_ficlo/morphfits_root/output`.


# Setup
MorphFITS runs are based on FICLOs, configuration objects based on the parameters
of field, image version, catalog version, filter, and object. To read more about
FICLOs, and the directory structure, read [the data documentation
here](./data/README.md).

|Letter|Name|
|:---|:---|
|`F`|Field|
|`I`|Image Version|
|`C`|Catalog Version|
|`L`|Filter|
|`O`|Object|

For a given FICLO, the program requires the following files,

1. Simulated PSF, which can be [downloaded from STSci for a
   filter](https://stsci.app.box.com/v/jwst-simulated-psf-library/folder/174723156124).
2. Segmentation map, which can be [downloaded from JWST for a field and image
   version](https://s3.amazonaws.com/grizli-v2/JwstMosaics/v7/index.html).
3. Photometric catalog, which can be [downloaded from JWST for a field and image
version](https://s3.amazonaws.com/grizli-v2/JwstMosaics/v7/index.html).
4. Exposure map, which can be [downloaded from JWST for a field, image version,
and filter](https://s3.amazonaws.com/grizli-v2/JwstMosaics/v7/index.html).
5. Science frame, which can be [downloaded from JWST for a field, image version,
and filter](https://s3.amazonaws.com/grizli-v2/JwstMosaics/v7/index.html).
6. Weights map, which can be [downloaded from JWST for a field, image version,
and filter](https://s3.amazonaws.com/grizli-v2/JwstMosaics/v7/index.html).

and the following input directory/filename structure,
<table>
<tr>
<th>
Filesystem Tree
</th>
<th>
File
</th>
</tr>

<tr>
<td>
<pre>
input/
├── psfs/
│   └── PSF_NIRCam_in_flight_opd_filter_{L}.fits
└── {F}/
    └── {I}/
        ├── {F}-{I}-ir_seg.fits
        ├── {F}-{I}-fix_phot_apcorr.fits
        └── {L}/
            ├── {F}-{I}-{L}_dr[c/z]_exp.fits
            ├── {F}-{I}-{L}_dr[c/z]_sci.fits
            └── {F}-{I}-{L}_dr[c/z]_wht.fits
</pre>
</td>

<td>
<pre>
<br><br>1. Simulated PSF
<br><br>2. Segmentation Map
3. Photometric Catalog
<br>4. Exposure Map
5. Science Frame
6. Weights Map
</pre>
</td>
</tr>
</table>

for a given filter `{F}`, image version `{I}`, and filter `{L}`, and where the
final three files contain either the string `drc` or `drz`. Note the
segmentation map, exposure map, science frame, and weights map are downloaded as
`.fits.gz` files, and must be uncompressed, e.g. via `gzip -vd *` in the
appropriate directories. MorphFITS will create product and output directories
from the root of the input directory, so to avoid git conflicts, it is
recommended to locate the input directory under an untracked directory, such as
`sandbox`, or `morphfits_root`.


# Usage
The program can be configured in three ways. If it receives parameters from multiple
sources, it will use the values in this order.

1. CLI call (see below)
2. Configuration file [(see a sample here)](./data/galwrap/sample_config.yaml)
3. Automatic input detection

The program requires the following parameters to run. Note the first cannot be
discovered, and must be declared via CLI call or configuration file. As well,
the only available wrapper is `galwrap`.

|CLI Key|File Key|Type|Description|
|:---|:---|:---|:---|
|`--input-root`|`input_root`|`str`|Path to input directory root.|
|`--fields`|`fields`|`list[str]`|Fields of frames over which to fit.|
|`--image-versions`|`image_versions`|`list[str]`|Image versions of frames over which to fit.|
|`--catalog-versions`|`catalog_versions`|`list[str]`|Catalog versions over which to fit.|
|`--filters`|`filters`|`list[str]`|Filters of frames over which to fit.|
|`--objects`|`objects`|`list[int]`|Object IDs in catalog over which to fit.|


## Terminal
To run MorphFITS via CLI call, run
```
poetry run morphfits [wrapper] [OPTIONS]
```
for the following options

|CLI Key|Type|Description|
|:---|:---|:---|
|`--config-path`|`str`|Path to configuration file.|
|`--galwrap-root`|`str`|Path to root data directory.|
|`--input-root`|`str`|Path to root input directory, under root.|
|`--product-root`|`str`|Path to root products directory.|
|`--output-root `|`str`|Path to root output directory.|
|`--fields`|`list[str]`|Fields of frames over which to fit.|
|`--image-versions`|`list[str]`|Image versions of frames over which to fit.|
|`--catalog-versions`|`list[str]`|Catalog versions of frames over which to fit.|
|`--filters`|`list[str]`|Filters of frames over which to fit.|
|`--objects`|`list[int]`|IDs of objects in frames over which to fit.|
|`--regenerate-products`|`bool`|Regenerate all products.|
|`--regenerate-stamp`|`bool`|Regenerate stamp cutouts.|
|`--regenerate-psf`|`bool`|Regenerate PSF crops.|
|`--regenerate-mask`|`bool`|Regenerate mask cutouts.|
|`--regenerate-sigma`|`bool`|Regenerate sigma maps.|
|`--regenerate-feedfile`|`bool`|Regenerate feedfiles.|
|`--apply-mask`|`bool`|Use the mask in GALFIT.|
|`--apply-psf`|`bool`|Use the PSF in GALFIT.|
|`--apply-sigma`|`bool`|Use the sigma map in GALFIT.|
|`--kron-factor`|`bool`|Stamp size factor, by default 3. The higher the number, the larger the image, and smaller the object.|
|`--display-progress`|`bool`|Display progress as a loading bar, rather than a line for each object.|
|`--help`|`None`|Display all options.|


## Configuration File
To run MorphFITS via a configuration file, run
```
poetry run morphfits [wrapper] --config-path=[config_path] [OPTIONS]
```

for the path to some configuration file.


# Stages

The program executes the following stages.

1. Load in configuration parameters
    - from terminal, file, or directory discovery
    - creates a configuration object
2. Search for input data and create product and output directories
    - creates directories for each FICLO with existing input
3. Create products from input data for each FICLO
    - science frame stamp cutouts
    - exposure and weight sigma map cutouts
    - PSF crops
    - mask cutouts
    - GALFIT feedfiles
4. Run GALFIT on created products
    - outputs model and log to output directory
5. Create plots from output model and products
    - outputs plot to output directory


# Cookbook
## Typical Operation
A typical run of MorphFITS involves collecting and structuring the correct data,
then running the program.
```
poetry run morphfits galwrap --config-path=./examples/single_ficlo/config.yaml
```
The model is visually compared in `output_ficlo/plots/...products.png`, and
stored as a FITS in `output_ficlo/galfit/...model.fits`.


## Selecting Objects
To investigate objects of interest, generate stamp cutouts of each object in a
science frame with
```
poetry run morphfits stamp [OPTIONS]
```
for the same CLI options as above. This will generate stamps of each available
object for a given FICL, and store them in the product directory corresponding
to either `product_root` or `input_root/../products`. It will then plot stamps,
50 at a time, and store them in the corresponding output directory, i.e.
`output_root/plots/F_I_C_L_O1-to-O2.png`.


## Regenerating Products
Products can be recreated after adjusting some settings. The following flags can
be used.
|Flag|Default|Product|
|:---|:---|:---|
|`--regenerate-products`|`False`|All products (overrides others)|
|`--regenerate-stamp`|`False`|Stamp cutouts|
|`--regenerate-sigma`|`False`|Sigma maps|
|`--regenerate-psf`|`False`|PSF crops|
|`--regenerate-mask`|`False`|Mask cutouts|

To regenerate all products for GALFIT, run
```
poetry run morphfits galwrap [OPTIONS] --regenerate-products
```


## Removing Products from GALFIT
Generated products can be excluded from GALFIT. The following flags can be used
to remove corresponding products from the generated feedfile. Note the feedfile
must be regenerated.
|Flag|Default|Product|Deactivate|
|:---|:---|:---|:---|
|`apply-sigma`|`True`|Sigma maps|`--no-apply-sigma`|
|`apply-psf`|`True`|PSF crops|`--no-apply-psf`|
|`apply-mask`|`True`|Masks|`--no-apply-mask`|

To exclude all generated products from GALFTI, run
```
poetry run morphfits galwrap [OPTIONS] --regenerate-feedfile --no-apply-sigma --no-apply-psf --no-apply-mask
```


# References
1. [GALFIT](https://users.obs.carnegiescience.edu/peng/work/galfit/galfit.html)
2. [DJA - The DAWN JWST Archive](https://dawn-cph.github.io/dja/index.html)
3. [The JWST v7.2 Mosaics](https://s3.amazonaws.com/grizli-v2/JwstMosaics/v7/index.html)
4. [The Library of Simulated PSFs](https://stsci.app.box.com/v/jwst-simulated-psf-library/folder/174723156124)
