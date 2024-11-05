# MorphFITS
MorphFITS is J-HIVE's morphology fitter program for Python.

In its current iteration, it contains GalWrap, a wrapper for GALFIT, for JWST data.


# Installation
To install this program, clone this repository, navigate to it, and install via
[Poetry](https://python-poetry.org/docs/).
```
git clone git@github.com:j-hive/morphfits.git
cd morphfits
poetry install
```

**This step is mandatory for GALFIT.** Download the GALFIT binary corresponding to your system [here](https://users.obs.carnegiescience.edu/peng/work/galfit/galfit.html), and move it to the location of your choosing (e.g., your data directory, if it exists).


# Quickstart
To quickstart MorphFITS without data, run
```
poetry run morphfits initialize -i [INPUT_ROOT] -F abell2744clu -I grizli-v7.2 -L f200w-clear
```
for an `INPUT_ROOT` of your choosing, e.g. `morphfits_root`, which is untracked.
The program will download the necessary `.fits` files from the DJA archive.
Then, download [the simulated PSF corresponding to the filter
here](https://stsci.app.box.com/v/jwst-simulated-psf-library/file/1025339832742),
and move it to the `psfs` directory under the input root, i.e. 
```
mv [DOWNLOADS]/PSF_NIRCam_in_flight_opd_filter_F200W.fits [INPUT_ROOT]/psfs
```
Now, with the input state ready for the pipeline, run
```
poetry run morphfits galwrap -i [INPUT_ROOT] -g [GALFIT_PATH] -O 4215
```
where the input root and path to the GALFIT binary *must* be provided. This fits
the object indexed as `4215` by the UNCOVER catalog. To run over all 65k+ objects,
simply exclude the `-O 4215` option. To run over more than one field, image
version, etc., it is recommended to use a configuration file instead.

For other examples, please follow one of the guides in [the examples
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
The program's settings can be configured via CLI call, or YAML file. If a
setting is found in both, MorphFITS will prefer the former.

To run MorphFITS via CLI call, run
```
poetry run morphfits [command] [OPTIONS]
```

To run MorphFITS via configuration file, run
```
poetry run morphfits [command] -c [path/to/YAML]
```

To declare multiple fields, image versions, etc. via CLI call, list them in
separate flags, i.e. 
```
poetry run morphfits [command] --field=deep --field=shallow --field=north [OPTIONS]
```

For available commands, run
```
poetry run morphfits --help
```

and for command-specific help, run
```
poetry run morphfits [command] --help
```


## Stages
MorphFITS runs the following stages. By default, all stages are run.

|Stage|Skip via CLI|Include via YAML|Description|
|:---|:---|:---|:---|
|Unzip|`--skip-unzip`|`unzip`|Un-compress zipped input files.|
|Product|`--skip-product`|`product`|Generate product files.|
|Morphology|`--skip-morphology`|`morphology`|Run morphology fitting and generate output files.|
|Catalog|`--skip-catalog`|`catalog`|Generate catalog files.|
|Histogram|`--skip-histogram`|`histogram`|Generate histograms from catalog files.|
|Plot|`--skip-plot`|`plot`|Generate plots from output and product files.|
|Cleanup|`--skip-cleanup`|`cleanup`|Remove failed directories.|

Stages can be skipped via CLI call by using one of the `skip` flags as above,
which can be seen in the `Stages` section of the `--help` menu.

Alternatively, stages can be selected to run by YAML file by the keyword
`stages`, i.e.
```
stages:
  - catalog
  - plot
  - cleanup
```
to only run the catalog, plotting, and cleanup stages.


## Remake
MorphFITS generates the following files. By default, no files are remade.

|File|Remake via CLI|Remake via YAML|Description|
|:---|:---|:---|:---|
|Stamp|`--remake-stamps`|`stamps`|Square cutout of FICLO.|
|Sigma Map|`--remake-sigmas`|`sigmas`|Square variation map of FICLO.|
|PSF Crop|`--remake-psfs`|`psfs`|Square PSF crop for FICLO.|
|Mask|`--remake-masks`|`masks`|Square mask of FICLO.|
|Model|`--remake-morphology`|`morphology`|Morphological model of FICLO.|
|Plots|`--remake-plots`|`plots`|Histogram and model plots for FICLO.|
|Others|`--remake-feedfiles`|`others`|Other files (e.g. feedfiles).|

Files can be remade via CLI call by using one of the `remake` flags as above,
which can be seen in the `Remake` section of the `--help` menu.

Alternatively, files can be remade by YAML file by the keyword `remake`, i.e.
```
remake:
  - stamps
  - morphology
```
to only remake the stamps and models.


## Settings
To configure settings via CLI options or YAML file, set by the following keys.

|CLI Full|CLI|YAML|Type|Description|
|:---|:---|:---|:---|:---|
|`--config`|`-c`|`config_path`|`Path`|Path to configuration settings YAML file.|
|`--galfit`|`-g`|`galfit_path`|`Path`|Path to GALFIT executable binary. Required for `galwrap`.|
|`--root`|`-r`|`morphfits_root`|`Path`|Path to MorphFITS filesystem root. Either this or input root required.|
|`--input`|`-i`|`input_root`|`Path`|Path to root input directory. Either this or root required.|
|`--output`||`output_root`|`Path`|Path to root output directory.|
|`--product`||`product_root`|`Path`|Path to root product directory.|
|`--run`||`run_root`|`Path`|Path to root run directory.|
|`--field`|`-F`|`fields`|`list[str]`|Fields to run.|
|`--image-version`|`-I`|`image_versions`|`list[str]`|Image versions to run.|
|`--catalog-version`|`-C`|`catalog_versions`|`list[str]`|Catalog versions to run.|
|`--filter`|`-L`|`filters`|`list[str]`|Filters to run.|
|`--object`|`-O`|`objects`|`list[int]`|Object IDs to run.|
|`--first-object`||`first_object`|`int`|ID of first object in range.|
|`--last-object`||`last_object`|`int`|ID of last object in range.|
|`--batch-n-process`|`-n`|`batch_n_process`|`int`|Number of processes in batch.|
|`--batch-process-id`|`-p`|`batch_process_id`|`int`|ID of process in batch.|


# Cookbook
## Initialize
To download all the required input frames for a given FICLO, run
```
poetry run morphfits initialize -c path/to/config.yaml
```
with a configuration file detailing FICLOs. Here's [an example of a
configuration YAML](./examples/settings.yaml). Note you are still required to
manually download [the simulated PSF files from
STSci](https://stsci.app.box.com/v/jwst-simulated-psf-library/folder/174723156124)
and move them to the appropriate `input_root/psfs` directory. For more details,
run
```
poetry run morphfits initialize --help
```


## Batch Mode
To run MorphFITS over large ranges and multiple cores, it is recommended to run
it via batch mode. This mode is activated by using the following arguments.

|CLI Key|YAML Key|Type|Default|Description|
|:---|:---|:---|---:|:---|
|`--first-object`|`first_object`|`int`|`None`|ID of first object in range.|
|`--last-object`|`first_object`|`int`|`None`|ID of last object in range.|
|`--batch-n-process`|`batch_n_process`|`int`|`1`|Total number of batch processes.|
|`--batch-process-id`|`batch_process_id`|`int`|`0`|Process ID of current run.|

To run over all available objects in a catalog, don't provide arguments for
`--object`, `first-object`, or `last-object`. 

For example, to run over objects `10000` through `19999` in 5 processes, running
the first process would be via
```
poetry run morphfits galwrap -i path/to/input -g path/to/galfit --first-object 10000 --last-object 20000 --batch-n-process 5 --batch-process-id 0
```


## Single Operation
A typical run of MorphFITS involves collecting and structuring the correct data,
then running the program.
```
poetry run morphfits galwrap -c examples/single_ficlo/settings.yaml
```
The model is visually compared in `[ficlo_output]/plots/F_I_C_L_O_products.png`, and
stored as a FITS in `[ficlo_output]/galfit/F_I_C_L_O_galfit.fits`.


# References
1. [J-HIVE](https://j-hive.org/)
2. [GALFIT](https://users.obs.carnegiescience.edu/peng/work/galfit/galfit.html)
3. [DJA - The DAWN JWST Archive](https://dawn-cph.github.io/dja/index.html)
4. [The JWST v7.2 Mosaics](https://s3.amazonaws.com/grizli-v2/JwstMosaics/v7/index.html)
5. [The Library of Simulated PSFs](https://stsci.app.box.com/v/jwst-simulated-psf-library/folder/174723156124)
