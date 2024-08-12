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

Then, download the GALFIT binary corresponding to your system [here](https://users.obs.carnegiescience.edu/peng/work/galfit/galfit.html).


# Quickstart
To quickstart MorphFITS, please follow one of the guides in [the examples
directory](./examples/).


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

For a given FICLO, the program requires the following files.

1. [Simulated PSF](https://stsci.app.box.com/v/jwst-simulated-psf-library/folder/174723156124)
2. [Segmentation map](https://s3.amazonaws.com/grizli-v2/JwstMosaics/v7/index.html)
3. [Photometric catalog](https://s3.amazonaws.com/grizli-v2/JwstMosaics/v7/index.html)
4. [Exposure map](https://s3.amazonaws.com/grizli-v2/JwstMosaics/v7/index.html)
5. [Science frame](https://s3.amazonaws.com/grizli-v2/JwstMosaics/v7/index.html)
6. [Weights map](https://s3.amazonaws.com/grizli-v2/JwstMosaics/v7/index.html)

and the following input directory/filename structure.
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

for a given filter `{F}`, image version `{I}`, and filter `{L}`.

Please note
- the final three files contain either the string `drc` or `drz`
- MorphFITS will create product and output directories from the parent of the
  input directory
- several files are downloaded as `.fits.gz` files
  - it is recommended to unzip them, for example via `gzip -vd *`
- directories named `sandbox` and `morphfits_root` are untracked by git
  - it is recommended to use one of these directories to avoid git conflicts


# Usage
The program can be configured in three ways. If it receives parameters from multiple
sources, it will use the values in this order.

1. CLI call (see below)
2. Configuration file [(see a sample here)](./examples/config.yaml)
3. Input directory crawling

The program requires the following parameters to run the GALFIT wrapper,
GalWrap. Note a path to a GALFIT binary *must* be provided.

|CLI Key|YAML Key|Type|Description|
|:---|:---|:---|:---|
|`--config-path`||`str`|Path to configuration YAML file.|
|`--galfit-path`|`galfit_path`|`str`|Path to GALFIT binary file.|
|`--input-root`|`input_root`|`str`|Path to input directory root.|
|`--field`|`fields`|`list[str]`|Fields of frames over which to fit.|
|`--image-version`|`image_versions`|`list[str]`|Image versions of frames over which to fit.|
|`--catalog-version`|`catalog_versions`|`list[str]`|Catalog versions over which to fit.|
|`--filter`|`filters`|`list[str]`|Filters of frames over which to fit.|
|`--object`|`objects`|`list[int]`|Object IDs in catalog over which to fit.|


## Terminal
To run MorphFITS via CLI call, run
```
poetry run morphfits [wrapper] [OPTIONS]
```
and use the flag `--help` for details on each option.

To declare multiple fields, image versions, etc. via CLI call, list them in
separate flags, i.e. 
```
poetry run morphfits [wrapper] --field=deep --field=shallow --field=north [OPTIONS]
```

## Configuration File
To run MorphFITS via a configuration file, run
```
poetry run morphfits [wrapper] --config-path=[config_path] [OPTIONS]
```


# Cookbook
## Download Input
To download all the required input frames for a given FICLO, run
```
poetry run morphfits download --config-path=path/to/config.yaml
```
with a configuration file detailing FICLOs. Here's [an example of a
configuration YAML](./examples.config.yaml). Note you are still required to
download [the simulated PSF files from
STSci](https://stsci.app.box.com/v/jwst-simulated-psf-library/folder/174723156124)
and move them to the appropriate `input_root/psfs` directory.

## Typical Operation
A typical run of MorphFITS involves collecting and structuring the correct data,
then running the program.
```
poetry run morphfits galwrap --config-path=./examples/single_ficlo/config.yaml
```
The model is visually compared in `[ficlo_output]/plots/F_I_C_L_O_products.png`, and
stored as a FITS in `[ficlo_output]/galfit/F_I_C_L_O_galfit.fits`.


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
|`use`|`bool`|This model is usable for scientific analysis.|
|`field`|`str`|Field of image.|
|`image version`|`str`|Image processing version.|
|`catalog version`|`str`|Cataloging version.|
|`filter`|`str`|Filter wavelength.|
|`object`|`int`|Object ID in catalog.|
|`return code`|`int`|GALFIT return code.|
|`flags`|`int`|GALFIT flags bitmask.|
|`convergence`|`int`|Fitting parameter convergence bitmask.|
|`center x`|`float`|X-position of model center.|
|`center y`|`float`|Y-position of model center.|
|`surface brightness`|`float`|Flux at surface of object.|
|`effective radius`|`float`|Effective object radius.|
|`sersic`|`float`|`n`, concentration parameter.|
|`axis ratio`|`float`|Ratio of model axes.|
|`position angle`|`float`|Rotation angle of model.|

as well as the errors on each parameter.


### GALFIT Return Codes

|Code|Description|
|:---|:---|
|`0`|Fitting executed without termination.|
|`1`|Fitting failed.|
|`2`|Missing products.|
|`139`|Segmentation fault.|

### GALFIT Flags Bitmask

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

### GALFIT Parameter Bitmask

|Parameter|Fails|Bit|Value|
|:---|:---|:---|:---|
|`effective radius`|:x:|0|1|
|`sersic`|:x:|1|2|
|`axis ratio`|:x:|2|4|


# References
1. [GALFIT](https://users.obs.carnegiescience.edu/peng/work/galfit/galfit.html)
2. [DJA - The DAWN JWST Archive](https://dawn-cph.github.io/dja/index.html)
3. [The JWST v7.2 Mosaics](https://s3.amazonaws.com/grizli-v2/JwstMosaics/v7/index.html)
4. [The Library of Simulated PSFs](https://stsci.app.box.com/v/jwst-simulated-psf-library/folder/174723156124)
