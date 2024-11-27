# Scripts
Single-use data processing tool scripts for MorphFITS.


## Rescale
Rescale a simulated PSF by re-binning from its original pixel scale to the pixel
scale of a science frame with the same filter. 

The [PSFs from
STSci](https://stsci.app.box.com/v/jwst-simulated-psf-library/folder/174723156124)
have pixel scales in the neighborhood of 0.007775 arcseconds per pixel.
Meanwhile, the science frames with filters less than or equal to `f210m` have
pixel scales of 0.02 arcseconds per pixel, and science frames with filters
greater than `f210m` have pixel scales of 0.04 arcseconds per pixel. This script
rescales and renames the simulated PSFs so they match given science frames.

To use, run
```
cd scripts
python rescale.py [path/to/PSF] [pixel scale OR path/to/science]
```
where the second argument could be either the new pixel scale to re-bin to, as a
float, or a path to a corresponding science frame from which to extract the new
pixel scale.

For example, for filter `f210m-clear`:
|File|Pixel Scale ("/px)|Dimensions (px)|
|:---|---:|---:|
|PSF|0.007775|1288 x 1288|
|Science|0.02|irrelevant|
|Rescaled PSF|0.02|250 x 250|