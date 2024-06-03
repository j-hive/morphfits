#!/bin/bash

# Setup directories
echo Setting up directories.
cd examples/single_ficlo
mkdir -p morphfits_root/input
cd morphfits_root/input
mkdir -p psfs abell2744clu/grizli-v7.2/f200w

# Download files
echo Downloading files.
cd abell2744clu/grizli-v7.2
wget https://s3.amazonaws.com/grizli-v2/JwstMosaics/v7/abell2744clu-grizli-v7.2-ir_seg.fits.gz
wget https://s3.amazonaws.com/grizli-v2/JwstMosaics/v7/abell2744clu-grizli-v7.2-fix_phot_apcorr.fits
cd f200w
wget https://s3.amazonaws.com/grizli-v2/JwstMosaics/v7/abell2744clu-grizli-v7.2-f200w-clear_drc_exp.fits.gz
wget https://s3.amazonaws.com/grizli-v2/JwstMosaics/v7/abell2744clu-grizli-v7.2-f200w-clear_drc_sci.fits.gz
wget https://s3.amazonaws.com/grizli-v2/JwstMosaics/v7/abell2744clu-grizli-v7.2-f200w-clear_drc_wht.fits.gz

# Uncompress files
echo Unzipping files.
gzip -vd *
cd ..
gzip -vd *

echo Done.
