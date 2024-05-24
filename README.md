# MorphFITS
MorphFITS is a morphology fitter program for Python.

In its current iteration, it runs GalWrap, a wrapper for GALFIT for JWST data.


# Quickstart
To install this program, run
```
poetry install
```

GalWrap requires a certain input structure, which is [detailed here](./data/README.md). It also requires configuration, which is [detailed here](./config/README.md).

To run the program via a configuration file, run
```
poetry run python -m morphfits galwrap --config-path [config_path]
```



# References
1. [GALFIT](https://users.obs.carnegiescience.edu/peng/work/galfit/galfit.html)
2. [DJA - The DAWN JWST Archive](https://dawn-cph.github.io/dja/index.html)
