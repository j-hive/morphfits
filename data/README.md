# Data Standards
This directory contains data standards for all morphology wrappers. Data
standards are program configurations and constraints which are not expected to
change frequently.


# FICLOs
MorphFITS operates on data using the categorization structure of FICLOs, which
is an acronym for the Field, Image version, Catalog version, fiLter, and Object ID
of a galaxy or cluster to be fitted.

|Letter|Variable|Description|Example|
|:---|:---|:---|:---|
|`F`|`field`|Center of field at which JWST instrument is pointed.|`abell2744clu`|
|`I`|`image_version`|Program used to process JWST data.|`grizli-v7.2`|
|`C`|`catalog_version`|Source of photometric catalog.|`dja-v7.2`|
|`L`|`filter`|Instrument filter(s) used.|`f200w-clear`|
|`O`|`object`|Integer ID of galaxy or cluster in catalog.|`4215`|
|`x`|`input_root`|Root input directory.|`morphfits_root/input`|
|`y`|`product_root`|Root products directory.|`morphfits_root/products`|
|`z`|`output_root`|Root output directory.|`morphfits_root/output`|
|`r`|`run_root`|Root directory for each run.|`morphfits_root/runs`|
|`D`|`datetime`|Datetime at start of run, in ISO-8601.|`20230625T145928`|
|`N`|`run_num`|Number of run if multiple are started at the same datetime.|`02`|


# Filesystem Structure
The program operates in three data stages - input, products, and output. Input
files are downloaded directly from the [JWST Mosaics
index](https://s3.amazonaws.com/grizli-v2/JwstMosaics/v7/index.html) and [STSci
simulated PSFs
directory](https://stsci.app.box.com/v/jwst-simulated-psf-library/folder/174723156124),
and expected to be organized in the structure detailed below. Product and output
files and directories are generated by the program.


## Tree
<table>
<tr>
<th>
In Filesystem
</th>
<th>
In Program
</th>
</tr>

<tr>
<td>
<pre>
morphfits_root/
├── input/
│   ├── psfs/
│   │   └── PSF_NIRCam_in_flight_opd_filter_L.fits
│   └── F/
│       └── I/
│           ├── F-I-ir_seg.fits
│           ├── F-I-fix_phot_apcorr.fits
│           └── L/
│               ├── F-I-L_dr[c/z]_exp.fits
│               ├── F-I-L_dr[c/z]_sci.fits
│               └── F-I-L_dr[c/z]_wht.fits
├── output/
│   ├── catalog.csv
│   └── F/
│       └── I/
│           └── C/
│               └── L/
│                   ├── F_I_C_L_objects.png
│                   └── O/
│                       ├── logs/
│                       │   └── F_I_C_L_O_galfit.log
│                       ├── models/
│                       │   ├── F_I_C_L_O_galfit.fits
│                       │   ├── F_I_C_L_O_imcascade.fits
│                       │   └── F_I_C_L_O_pysersic.fits
│                       └── plots/
│                           ├── F_I_C_L_O_galfit.png
│                           ├── F_I_C_L_O_imcascade.png
│                           ├── F_I_C_L_O_products.png
│                           ├── F_I_C_L_O_pysersic.png
│                           └── F_I_C_L_O_wrappers.png
├── products/
│   └── F/
│       └── I/
│           └── C/
│               └── L/
│                   └── O/
│                       ├── F_I_C_L_O.feedfile
│                       ├── F_I_C_L_O_mask.fits
│                       ├── F_I_C_L_O_psf.fits
│                       ├── F_I_C_L_O_sigma.fits
│                       └── F_I_C_L_O_stamp.fits
└── runs/
    └── D.N/
        ├── config.yaml
        ├── files.csv
        ├── histogram.png
        ├── morphfits.log
        └── parameters.csv
</pre>
</td>

<td>
<pre>
morphfits_root/
├── input_root/
│   ├── input_psfs/
│   │   └── original_psf
│   └── .../
│       └── input_data/
│           ├── segmap
│           ├── catalog
│           └── input_images/
│               ├── exposure
│               ├── science
│               └── weights
├── output_root/
│   ├── morphfits_catalog
│   └── .../
│       └── .../
│           └── .../
│               └── ficl_output/
│                   ├── ficl_objects
│                   └── ficlo_output/
│                       ├── logs/
│                       │   └── galfit_log
│                       ├── models/
│                       │   ├── galfit_model
│                       │   ├── imcascade_model
│                       │   └── pysersic_model
│                       └── plots/
│                           ├── galfit_plot
│                           ├── imcascade_plot
│                           ├── products_plot
│                           ├── pysersic_plot
│                           └── wrapper_comparison
├── product_root/
│   └── .../
│       └── .../
│           └── .../
│               └── .../
│                   └── ficlo_products/
│                       ├── feedfile
│                       ├── mask
│                       ├── psf
│                       ├── sigma
│                       └── stamp
└── run_root/
    └── run/
        ├── config
        ├── files
        ├── histogram
        ├── morphfits_log
        └── parameters
</pre>
</td>
</tr>
</table>


## Paths
The following are paths used in the program. These paths can be retrieved by
running `paths.get_path` for the required variables.

|Stage|Path Name|Format|Required|Description|
|:---|:---|:---|:---|:---|
|Input|`morphfits_root`|`/`||Root directory for all MorphFITS files.|
|Input|`input_root`|`/`|`x`|Root directory for all input files.|
|Input|`input_psfs`|`/`|`x`|Directory for simulated PSFs.|
|Input|`original_psf`|`/`|`xL`|Simulated NIRCAM PSF.|
|Input|`input_data`|`/`|`xFI`|Directory for all input but PSFs.|
|Input|`segmap`|`.fits`|`xFI`|Segmentation map.|
|Input|`catalog`|`.fits`|`xFI`|Photometry catalog table.|
|Input|`input_images`|`/`|`xFIL`|Directory for input images.|
|Input|`exposure`|`.fits`|`xFIL`|Exposure map.|
|Input|`science`|`.fits`|`xFIL`|Science frame.|
|Input|`weights`|`.fits`|`xFIL`|Weights map.|
|Output|`output_root`|`/`|`z`|Root directory for all output files.|
|Output|`morphfits_catalog`|`.csv`||Catalog of all fits under root.|
|Output|`ficl_output`|`/`|`zFICLO`|Directory for all output for a FICL.|
|Output|`ficl_objects`|`.png`|`zFICLO`|Image showing all objects in a FICL.|
|Output|`ficlo_output`|`/`|`zFICLO`|Directory for all output for a FICLO.|
|Output|`logs`|`/`|`zFICLO`|Directory for logs from fit executions.|
|Output|`galfit_log`|`.log`|`zFICLO`|GALFIT parameters fit log.|
|Output|`models`|`/`|`zFICLO`|Directory for fit models.|
|Output|`galfit_model`|`.fits`|`zFICLO`|GALFIT fit model.|
|Output|`imcascade_model`|`.fits`|`zFICLO`|Imcascade fit model.|
|Output|`pysersic_model`|`.fits`|`zFICLO`|Pysersic fit model.|
|Output|`plots`|`/`|`zFICLO`|Directory for plots.|
|Output|`galfit_plot`|`.png`|`zFICLO`|GALFIT model fidelity visualization.|
|Output|`imcascade_plot`|`.png`|`zFICLO`|Imcascade model fidelity visualization.|
|Output|`products_plot`|`.png`|`zFICLO`|Display of all products.|
|Output|`pysersic_plot`|`.png`|`zFICLO`|Pysersic model fidelity visualization.|
|Output|`wrapper_comparison`|`.png`|`zFICLO`|Comparison of models.|
|Product|`product_root`|`/`|`y`|Root directory for all products.|
|Product|`ficlo_products`|`/`|`yFICLO`|Directory for all products for a FICLO.|
|Product|`feedfile`|`.feedfile`|`yFICLO`|GALFIT configurations.|
|Product|`mask`|`.fits`|`yFICLO`|Object mask.|
|Product|`psf`|`.fits`|`yFICLO`|PSF with same size as object.|
|Product|`sigma`|`.fits`|`yFICLO`|Sigma map.|
|Product|`stamp`|`.fits`|`yFICLO`|Object cutout.|
|Run|`run_root`|`/`|`r`|Directory for records from all runs.|
|Run|`run`|`/`|`rDN`|Directory for records from a single run.|
|Run|`config`|`.yaml`|`d`|Configuration settings from run.|
|Run|`files`|`.csv`|`d`|Modified or created files from run.|
|Run|`histogram`|`.png`|`d`|Histogram depicting parameter distribution.|
|Run|`morphfits_log`|`.log`|`d`|MorphFITS program log from run.|
|Run|`parameters`|`.csv`|`d`|Parameters found from run.|