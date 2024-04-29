# Data Standards

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