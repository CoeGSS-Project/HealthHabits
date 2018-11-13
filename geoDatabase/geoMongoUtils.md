# geoMongoUtils

## A basic python-client for a ad-hoc database of geographical and socio-economical data

### Dependencies

In order to use this piece of software you need to install the following modules either from `pip` or from your distribution package system:

- `pymongo`
- `shapely`

If you want to run the attached notebook as well, you will also have to install:

- a `Jupyter` notebook server;
- the `scipy` stack (`scipy, numpy, matplotlib`);
- `pandas`
- `folium`
- `seaborn`

### Installation

Put the `geoMongoUtils.py` file either in your local folder containing the notebook or within your python path.

Start an instance of `jupyter-notebook` in the notebook directory or any of its parents and browse to `http://localhost:8888` for the notebook tree.

Run a cell of code by clicking `Shift+Enter` (`Wolfram Mathematica`-like).

### Credentials

We created a **_read-only_** user called `coegss_guest` whose password is `C03G55!_Gu35t!_p455!`. Enter it as a constructor's argument or enter it in the password prompt in the notebook.

### Usage

At this early stage this `python` client exposes some preliminary functions whose documentation can be read in the `geoMongoUtils.py` file.
