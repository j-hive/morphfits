# Scripts
Single-use data processing tool scripts for MorphFITS.


## Rescale
Rescale a simulated PSF by re-binning from its original pixel scale to the pixel
scale of a science frame with the same filter. 

The [PSFs from
STSci](https://stsci.app.box.com/v/jwst-simulated-psf-library/folder/174723156124)
all have the pixel scale of 0.007775 arcseconds per pixel. 

Meanwhile, the science frames with filters less than or equal to `f210m` have
pixel scales of 0.02 arcseconds per pixel, and science frames with filters
greater than `f210m` have pixel scales of 0.04 arcseconds per pixel. 

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
|Science|0.002|irrelevant|
|Rescaled PSF|0.002|250 x 250|