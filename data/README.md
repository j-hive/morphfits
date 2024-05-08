# Data Standards

## Filesystem Structure

```
galwrap_output/
└── {object}/
    └── {field}/
        └── {image_version}.{catalog_version}/
            ├── feedfiles/
            │   └── {filter}/
            │       └── {galaxy_id}_{object}{field}-{filter}.feedfile
            ├── galfit_output/
            │   └── {filter}/
            │       └── {galaxy_id}_{object}{field}-{filter}_model.fits
            ├── rms/
            │   └── {filter}/
            │       └── {galaxy_id}_{object}{field}-{filter}_rms.fits
            ├── stamps/
            │   └── {filter}/
            │       └── {galaxy_id}_{object}{field}-{filter}_sci.fits
            ├── masks/
            │   └── {galaxy_id}_{object}{field}_mask.fits
            ├── psfs/
            │   └── {object}{field}_{filter}_{pixname}_psf_{image_version}.{catalog_version}_crop.fits
            ├── segmaps/
            ├── visualizations/
            ├── {object}{field}_{image_version}_filter_info_.dat
            ├── {object}{field}_{image_version}_depth.txt
            └── all_filters.constraints

photometry_products/
└── {object}/
    └── {field}/
        └── {image_version}/
            └── rms/

imaging_products/
└── {object}/
    └── {field}/
        ├── {image_version}/
        │   ├── bcgs/
        │   └── science/
        └── {image_version}.{catalog_version}/
            ├── catalogs/
            │   ├── {object}{field}_photutils_segmap_{image_version}.{catalog_version}.fits
            │   └── {object}{field}_photutils_cat_{image_version}.{catalog_version}.fits
            └── psfs/
```

## Path and Variable Names
Please note paths and variables from previous notebooks have been renamed for
legibility and standardization. The following table details the changes made.

|Type|Former|Current|Description|
|:---|:---|:---|:---|
|Directory|`HOME`|`output_root`|Root for all output products.|
|Directory|`FILTDIR`|`output_filter_info`|Information and details regarding filters.|
|Directory|`PATH`|`output_ofic`|Output products for a given OFIC.|
|Directory|`STAMPDIR`|`output_stamps`|Postage stamps made from science images.|
|Directory|`MASKDIR`|`output_masks`|Masks made from segmentation maps.|
|Directory|`RMSDIR`|`output_rms`|Weight image cutouts from input RMS maps.|
|Directory|`FEEDDIR`|`output_feedfiles`|Generated feedfiles to run GalFit from.|
|Directory|`SEGMAPDIR`|`output_segmaps`|Segmentation maps.|
|Directory|`GALFITOUTDIR`|`output_galfit`|Output products from GalFit.|
|Directory|`PSFDIR`|`output_psfs`|PSF cutouts from input PSFs.|
|Directory|`VISDIR`|`output_visualizations`|GalFit parameter corner plots.|
|Directory||`photometry_root`|Root for all input photometry products.|
|Directory|`RMSMAPDIR`|`photometry_rms`|Input RMS maps.|
|Directory||`imaging_root`|Root for all input imaging products.|
|Directory|`DATADIR`|`imaging_of`|Input imaging products for a given object and field.|
|Directory|`IMGDIR`|`imaging_bcgs`|Input science images with brightest cluster galaxy subtracted.|
|Directory|`ORG_PSFDIR`|`imaging_psfs`|Input PSFs.|
|Directory|`CATDIR`|`imaging_catalogs`|Input segmentation maps.|
|File|`FILTINFO`|`file_filter_info`||
|File|`DEPTH`|`file_depth`||
|File|`SEGMAP`|`file_segmap`||
|File|`PHOTCAT`|`file_photometric_catalog`||
|File|`MASKNAME`|`file_mask`||
|File|`SCINAME`|`file_science`||


# 2024-05-08 WIP New Tree

## Legend
|Letter|Variable|Description|Example|
|:---|:---|:---|:---|
|`O`|`object`|Center of cluster instrument is pointed at.|`a370`|
|`F`|`field`|Field of observation.|`ncf`|
|`I`|`image_version`|Version of imaging.|`v2p0`|
|`C`|`catalog_version`|Version of cataloguing.|`1`|
|`L`|`filter`|Filter used.|`F140W`|
|`G`|`galaxy_id`|ID of galaxy to fit.|
|`P`|`pixname`|Pixel scale in human-readable format.|`40mas`|

## Filesystem Structure

<table>
<tr>
<th>
<div align="right">Directories</div>
</th>
<th>
Tree
</th>
<th>
Files
</th>
</tr>

<tr>

<td align="right">
<pre>
input_root_dir<br><br><br>input_ofi_dir<br>input_bcgs_dir<br><br>input_science_dir<br><br>input_rms_dir<br><br>input_ofic_dir<br>input_catalogs_dir<br><br><br>input_psfs_dir<br>
middle_root_dir<br><br><br>middle_ofic_dir<br>middle_feedfiles_dir<br><br><br>middle_rms_dir<br><br><br>middle_stamps_dir<br><br><br>middle_masks_dir<br><br>middle_psfs_dir<br><br>middle_segmaps_dir<br><br><br><br>
output_root_dir<br><br><br>output_ofic_dir<br>output_galfit_dir<br><br><br>output_visualizations_dir<br><br>
</pre>
</td>

<td>
<pre>
galwrap_input/
└── {O}/
    └── {F}/
        ├── {I}/
        │   ├── bcgs/
        │   │   └── {placeholder}.fits
        │   ├── science/
        │   │   └── {placeholder}.fits
        │   └── rms/
        │       └── {placeholder}.fits
        └── {I}.{C}/
            ├── catalogs/
            │   ├── {O}{F}_photutils_segmap_{I}.{C}.fits
            │   └── {O}{F}_photutils_cat_{I}.{C}.fits
            └── psfs/
                └── {placeholder}.fits
galwrap_middle/
└── {O}/
    └── {F}/
        └── {I}.{C}/
            ├── feedfiles/
            │   └── {L}/
            │       └── {G}_{O}{F}-{L}.feedfile
            ├── rms/
            │   └── {L}/
            │       └── {G}_{O}{F}-{L}_rms.fits
            ├── stamps/
            │   └── {L}/
            │       └── {G}_{O}{F}-{L}_sci.fits
            ├── masks/
            │   └── {G}_{O}{F}_mask.fits
            ├── psfs/
            │   └── {O}{F}_{L}_{P}_psf_{I}.{C}_crop.fits
            ├── segmaps/
            │   └── {placeholder}.fits
            ├── {O}{F}_{I}_filter_info_.dat
            ├── {O}{F}_{I}_depth.txt
            └── all_filters.constraints
galwrap_output/
└── {O}/
    └── {F}/
        └── {I}.{C}/
            ├── galfit_output/
            │   └── {L}/
            │       └── {G}_{O}{F}-{L}_model.fits
            └── visualizations/
                └── {L}/
                    └── {placeholder}.fits
</pre>
</td>

<td>
<pre>
<br><br><br><br><br>input_bcgs_file<br><br>input_science_file<br><br>input_rms_file<br><br><br>input_segmap_file<br>input_catalog_file<br><br>input_psf_file<br>
<br><br><br><br><br>middle_feedfile_file<br><br><br>middle_rms_file<br><br><br>middle_stamp_file<br><br>middle_mask_file<br><br>middle_psf_file<br><br>middle_segmap_file<br>middle_filter_file<br>middle_depth_file<br>middle_constraints_file<br>
<br><br><br><br><br>output_galfit_file<br><br><br>output_visualization_file
</pre>
</td>

</tr>
</table>

## Path Names
|Former Name|Path Name|:file_folder:|Path|Description|
|:---|:---|:---:|:---|:---|
||`input_root_dir`|:white_check_mark:|`galwrap_input/`|
||`input_ofi_dir`|:white_check_mark:|`galwrap_input/{O}/{F}/{I}/`|
||`input_bcgs_dir`|:white_check_mark:|`galwrap_input/{O}/{F}/{I}/bcgs/`|
||`input_bcgs_file`||`galwrap_input/{O}/{F}/{I}/bcgs/{placeholder}.fits`|
||`input_science_dir`|:white_check_mark:|`galwrap_input/{O}/{F}/{I}/science/`|
||`input_science_file`||`galwrap_input/{O}/{F}/{I}/science/{placeholder}.fits`|
||`input_rms_dir`|:white_check_mark:|`galwrap_input/{O}/{F}/{I}/rms`|
||`input_rms_file`||`galwrap_input/{O}/{F}/{I}/rms/{placeholder}.fits`|
||`input_ofic_dir`|:white_check_mark:|`galwrap_input/`|
||`input_catalogs_dir`|:white_check_mark:|`galwrap_input/`|
||`input_segmap_file`||`galwrap_input/`|
||`input_catalog_file`||`galwrap_input/`|
||`input_psfs_dir`|:white_check_mark:|`galwrap_input/`|
||`input_psf_file`||`galwrap_input/`|
||`middle_root_dir`|:white_check_mark:|`galwrap_middle/`|
||`middle_ofic_dir`|:white_check_mark:|`galwrap_middle/`|
||`middle_feedfiles_dir`|:white_check_mark:|`galwrap_middle/`|
||`middle_feedfile_file`||`galwrap_middle/`|
||`middle_rms_dir`|:white_check_mark:|`galwrap_middle/`|
||`middle_rms_file`||`galwrap_middle/`|
|`STAMPDIR`|`middle_stamps_dir`|:white_check_mark:|`galwrap_middle/`|
||`middle_stamp_file`||`galwrap_middle/`|
||`middle_masks_dir`|:white_check_mark:|`galwrap_middle/`|
||`middle_mask_file`||`galwrap_middle/`|
||`middle_psfs_dir`|:white_check_mark:|`galwrap_middle/`|
||`middle_psf_file`||`galwrap_middle/`|
||`middle_segmaps_dir`|:white_check_mark:|`galwrap_middle/`|
||`middle_segmap_file`||`galwrap_middle/`|
||`middle_filter_file`||`galwrap_middle/`|
||`middle_depth_file`||`galwrap_middle/`|
||`middle_constraints_file`||`galwrap_middle/`|
||`output_root_dir`|:white_check_mark:|`galwrap_output/`|
||`output_ofic_dir`|:white_check_mark:|`galwrap_output/`|
||`output_galfit_dir`|:white_check_mark:|`galwrap_output/`|
||`output_galfit_file`||`galwrap_output/`|
||`output_visualizations_dir`|:white_check_mark:|`galwrap_output/`|
||`output_visualization_file`||`galwrap_output/`|