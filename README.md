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
version `dja-v7.2`, filter `f200w`, and object `4215`, using GALFIT, run from
the base directory (`morphfits`)

```
sh ./examples/single_ficlo/setup.sh
```
in a bash environment to download and unzip JWST data, which will take several
minutes and ~`13GB`. Then, download [the simulated PSF for the filter
here](https://stsci.app.box.com/v/jwst-simulated-psf-library/file/1025339832742),
and move it to `examples/single_ficlo/morphfits_root/input/psfs/`, for example via
```
mv [download folder]/PSF_NIRCam_in_flight_opd_filter_F200W.fits ./examples/single_ficlo/morphfits_root/input/psfs
```
Then, run 
```
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
2. Configuration file [(see a sample here)](./data/galfit/sample_config.yaml)
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
and use the flag `--help` for details on each option.


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
    - wrapper-specific products (e.g. GALFIT feedfiles)
4. Run morphology fitter on created products
    - outputs model and log to output directory
5. Create plots from output model and products
    - outputs plot to output directory


## Program Logs
The program records the run status of each FICLO to the `run_root`, with the
following headers (along with the logs and configuration settings). For more information on the location of this records, and
other files, refer to [the data documentation](./data/README.md).

|Header|Type|Description|
|:---|:---|:---|
|`field`|`str`|Field of image.|
|`image version`|`str`|Image processing version.|
|`catalog version`|`str`|Cataloging version.|
|`filter`|`str`|Filter wavelength.|
|`object`|`int`|Object ID in catalog.|
|`use`|`bool`|Fitting successful and accurate.|
|`status`|`int`|Fitting return code.|
|`galfit flags`|`int`|Bit-mask of GALFIT flags.|
|`center x`|`float`|X-position of model center.|
|`center y`|`float`|Y-position of model center.|
|`integrated magnitude`|`float`|Total flux across image.|
|`effective radius`|`float`|Effective object radius.|
|`concentration`|`float`|`n`, concentration parameter.|
|`axis ratio`|`float`|Ratio of model axes.|
|`position angle`|`float`|Rotation angle of model.|

The header `status` records the integer code returned by the fitting program,
detailed below. Note any nonzero code represents a failure.

|Code|Description|
|:---|:---|
|`0`|Success|
|`1`|Failure|
|`2`|Missing feedfile (GALFIT only)|
|`139`|Segmentation fault|

The header `galfit flags` records the flags raised by GALFIT, detailed below.
Note not all flags result in failed fittings.

|Flag|Fails|Bit|Value|Description|
|:---|:---:|---:|---:|:---|
|`1`|:x:|0|1|Maximum number of iterations reached. Quit out early.|
|`2`|:x:|1|2|Suspected numerical convergence error in current solution.|
|`A-1`|:x:|2|4|No input data image found. Creating model only.|
|`A-2`|:x:|3|8|PSF image not found.  No convolution performed.|
|`A-3`||4|16|No CCD diffusion kernel found or applied.|
|`A-4`|:x:|5|32|No bad pixel mask image found.|
|`A-5`|:x:|6|64|No sigma image found.|
|`A-6`|:x:|7|128|No constraint file found.|
|`C-1`|:x:|8|256|Error parsing the constraint file.|
|`C-2`||9|512|Trying to constrain a parameter that is being held fixed.|
|`H-1`||10|1024|Exposure time header keyword is missing.  Default to 1 second.|
|`H-2`||11|2048|Exposure time is zero seconds.  Default to 1 second.|
|`H-3`||12|4096|`GAIN` header information is missing.|
|`H-4`||13|8192|`NCOMBINE` header information is missing.|
|`I-1`||14|16384|Convolution PSF exceeds the convolution box.|
|`I-2`||15|32768|Fitting box exceeds image boundary.|
|`I-3`||16|65536|Some pixels have infinite ADUs; set to 0.|
|`I-4`||17|131072|Sigma image has zero or negative pixels; set to 1e10.|
|`I-5`||18|262144|Pixel mask is not same size as data image.|


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
