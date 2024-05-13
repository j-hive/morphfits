# Data Standards

## Filesystem Structure
### Legend
|Letter|Variable|Description|Example|
|:---|:---|:---|:---|
|`F`|`field`|Center of cluster/field instrument is pointed at.|`abell2744clu`|
|`I`|`image_version`|Image processing version.|`grizli-v7.2`|
|`C`|`catalog_version`|Catalog version.|`DJA-v7.2`|
|`L`|`filter`|Observational filter band.|`f140w`|
|`O`|`object`|Integer ID of galaxy in catalog.|`1003`|
|`P`|`pixname`|Pixel scale in human-readable format.|`40mas`|

### Tree
```
jhive_galwrap_input/
├── {L}/
│   └── {L}_psf.fits
└── {F}/
    └── {I}/
        ├── {C}/
        │   ├── crossmatches/
        │   │   └── {F1}_{F2}_{I}_{C}_crossmatch.fits
        │   ├── {F}_{I}_{C}_pht.fits
        │   └── {F}_{I}_{C}_phz.fits
        ├── {L}/
        │   ├── exposures/
        │   │   └── {F}_{I}_{L}_exp.fits
        │   ├── sciences/
        │   │   └── {F}_{I}_{L}_sci.fits
        │   └── weights/
        │       └── {F}_{I}_{L}_wht.fits
        └── {F}_{I}_seg.fits

jhive_galwrap_products/
├── {F}/
│   └── {I}/
│       └── {C}/
│           └── {L}/
│               └── {O}/
│                   ├── feedfiles/
│                   │   └── {O}_{F}_{I}_{C}_{L}.feedfile
│                   ├── masks/
│                   │   └── {O}_{F}_{I}_{C}_mask.fits
│                   ├── psfs/
│                   │   └── {O}_{F}_{I}_{C}_{L}_{P}_psf.fits
│                   ├── sigmas/
│                   │   └── {O}_{F}_{I}_{C}_{L}_sigma.fits
│                   └── stamps/
│                       └── {O}_{F}_{I}_{C}_{L}_sci.fits
├── default.constraints
└── template.feedfile

jhive_galwrap_output/
└── {F}/
    └── {I}/
        └── {C}/
            └── {L}/
                └── {O}/
                    ├── models/
                    │   └── {O}_{F}_{I}_{C}_{L}_model.fits
                    └── plots/
                        ├── {O}_{F}_{I}_{C}_{L}_model_comparison.png
                        └── *other*.png
```

### Labelled

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
input_root<br>input_psfs<br><br><br>input_fi<br>catalogs<br>crossmatches<br><br><br><br>input_fil<br>exposures<br><br>images<br><br>weights<br><br><br>
product_root<br><br><br><br><br>product_ficlo<br>feedfiles<br><br>masks<br><br>psfs<br><br>sigmas<br><br>stamps<br><br><br><br>
output_root<br><br><br><br><br>output_ficlo<br>models<br><br>plots<br><br>
</pre>
</td>

<td>
<pre>
jhive_galwrap_input/
├─L/
│ └─L_psf.fits
└─F/
  └─I/
    ├─C/
    │ ├─crossmatches/
    │ │ └─F1_F2_I_C_crossmatch.fits
    │ ├─F_I_C_cat.fits
    │ └─F_I_C_phz.fits
    ├─L/
    │ ├─exposures/
    │ │ └─F_I_L_exp.fits
    │ ├─sciences/
    │ │ └─F_I_L_sci.fits
    │ └─weights/
    │   └─F_I_L_wht.fits
    └─F_I_seg.fits<br>
jhive_galwrap_products/
├─F/
│ └─I/
│   └─C/
│     └─L/
│       └─O/
│         ├─feedfiles/
│         │ └─O_F_I_C_L.feedfile
│         ├─masks/
│         │ └─O_F_I_C_mask.fits
│         ├─psfs/
│         │ └─O_F_I_C_L_P_psf.fits
│         ├─sigmas/
│         │ └─O_F_I_C_L_sigma.fits
│         └─stamps/
│           └─O_F_I_C_L_stamp.fits
├─default.constraints
└─template.feedfile<br>
jhive_galwrap_output/
└─F/
  └─I/
    └─C/
      └─L/
        └─O/
          ├─models/
          │ └─O_F_I_C_L_model.fits
          └─plots/
            ├─O_F_I_C_L_model_comparison.png
            └─*other*.png
</pre>
</td>

<td>
<pre>
<br><br>input_psf<br><br><br><br><br>crossmatch<br>catalog<br>photoz<br><br><br>exposure<br><br>science<br><br>weight<br>segmap<br>
<br><br><br><br><br><br><br>feedfile<br><br>mask<br><br>psf<br><br>sigma<br><br>stamp<br>constraints<br>feedfile_template<br>
<br><br><br><br><br><br><br>model<br><br>comparison_plot<br>*other_plots*
</pre>
</td>

</tr>
</table>

### Descriptions
|Former|:information_source:|Path Name|Full Path|Description|
|:---|:---:|:---|:---|:---|
||:file_folder:|`input_root`|`input_root/`|
||:file_folder:|`input_psfs`|`input_root/{L}/`|
||:framed_picture:|`input_psf`|`input_root/{L}/{L}_psf.fits`|
||:file_folder:|`input_fi`|`input_root/{F}/{I}/`|
||:file_folder:|`catalogs`|`input_root/{F}/{I}/{C}/`|
||:file_folder:|`crossmatches`|`input_root/{F}/{I}/{C}/crossmatches/`|
||:framed_picture:|`crossmatch`|`input_root/{F}/{I}/{C}/crossmatches/{F}_{F2}_{I}_{C}_crossmatch.fits`|
||:framed_picture:|`catalog`|`input_root/{F}/{I}/{C}/{F}_{I}_{C}_cat.fits`|
||:framed_picture:|`photoz`|`input_root/{F}/{I}/{C}/{F}_{I}_{C}_phz.fits`|
||:file_folder:|`input_fil`|`input_root/{F}/{I}/{L}/`|
||:file_folder:|`exposures`|`input_root/{F}/{I}/{L}/exposures/`|
||:framed_picture:|`exposure`|`input_root/{F}/{I}/{L}/exposures/{F}_{I}_{L}_exp.fits`|
||:file_folder:|`images`|`input_root/{F}/{I}/{L}/sciences/`|
||:framed_picture:|`science`|`input_root/{F}/{I}/{L}/sciences/{F}_{I}_{L}_sci.fits`|
||:file_folder:|`weights`|`input_root/{F}/{I}/{L}/weights/`|
||:framed_picture:|`weight`|`input_root/{F}/{I}/{L}/weights/{F}_{I}_{L}_wht.fits`|
||:framed_picture:|`segmap`|`input_root/{F}/{I}/{F}_{I}_seg.fits`|
||:file_folder:|`product_root`|`product_root/`|
||:file_folder:|`product_ficlo`|`product_root/{F}/{I}/{C}/{L}/{O}/`|
||:file_folder:|`feedfiles`|`product_root/{F}/{I}/{C}/{L}/{O}/feedfiles/`|
||:pencil:|`feedfile`|`product_root/{F}/{I}/{C}/{L}/{O}/feedfiles/{O}_{F}_{I}_{C}_{L}.feedfile`|
||:file_folder:|`masks`|`product_root/{F}/{I}/{C}/{L}/{O}/masks/`|
||:framed_picture:|`mask`|`product_root/{F}/{I}/{C}/{L}/{O}/masks/{O}_{F}_{I}_{C}_mask.fits`|
||:file_folder:|`psfs`|`product_root/{F}/{I}/{C}/{L}/{O}/psfs/`|
||:framed_picture:|`psf`|`product_root/{F}/{I}/{C}/{L}/{O}/psfs/{O}_{F}_{I}_{C}_{L}_psf.fits`|
||:file_folder:|`sigmas`|`product_root/{F}/{I}/{C}/{L}/{O}/sigmas/`|
||:framed_picture:|`sigma`|`product_root/{F}/{I}/{C}/{L}/{O}/sigma/{O}_{F}_{I}_{C}_{L}_sigma.fits`|
||:file_folder:|`stamps`|`product_root/{F}/{I}/{C}/{L}/{O}/stamps/`|
||:framed_picture:|`stamp`|`product_root/{F}/{I}/{C}/{L}/{O}/stamps/{O}_{F}_{I}_{C}_{L}_stamp.fits`|
||:pencil:|`constraints`|`product_root/default.constraints`|
||:pencil:|`feedfile_template`|`product_root/template.feedfile`|
||:file_folder:|`output_root`|`output_root/`|
||:file_folder:|`output_ficlo`|`output_root/{F}/{I}/{C}/{L}/{O}/`|
||:file_folder:|`models`|`output_root/{F}/{I}/{C}/{L}/{O}/models/`|
||:framed_picture:|`model`|`output_root/{F}/{I}/{C}/{L}/{O}/models/{O}_{F}_{I}_{C}_{L}_model.fits`|
||:file_folder:|`plots`|`output_root{F}/{I}/{C}/{L}/{O}/plots/`|
||:bar_chart:|`comparison_plot`|`output_root/{F}/{I}/{C}/{L}/{O}/plots/{O}_{F}_{I}_{C}_{L}_comparison.png`|
||:bar_chart:|`*other_plots*`|`output_root/{F}/{I}/{C}/{L}/{O}/plots/{O}_{F}_{I}_{C}_{L}_*other_plots*.png`|


---

## [DEPRECATED] Path and Variable Names
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