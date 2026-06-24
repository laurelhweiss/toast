# TOAST — The On-sky Area Stacking Tool

An interactive widget for extracting and stacking IFU spectra as a function of 2D location on-sky. Utilizes a user-drawn lasso selector on an image of a target object. 

Built for use with HETDEX data alongside [HETDEX API](https://github.com/HETDEX/hetdex_api) and [ELiXer](https://github.com/HETDEX/elixer)

**Author:** Laurel H. Weiss

---

## Features

- Displays HETDEX fiber coverage of an input coordinate (or HETDEX Detectid) on an image cutout.
- Ability to pan around to any RA/Dec with arrow controls. 
- A user drawn lasso selects locations of interest directly on the image. The corresponding fiber spectra can then be extracted and stored. 
- Two extraction modes:
  - `fibers` — raw calibrated fiber spectra 
  - `psf` — PSF-weighted extraction at the lasso center
- Calculates a stacked spectrum from the selected fibers with a specified statistic (`mean`, `median`, `biweight`, `weighted_biweight`)
- Plots individual spectra and stack in an interactive spectrum panel. 

---

## Requirements

- Python 3.8+
- [`hetdex_api`](https://github.com/HETDEX/hetdex_api)
- [`elixer`](https://github.com/HETDEX/elixer)
- `astropy`, `astroquery`
- `matplotlib`, `ipywidgets`, `ipympl`
- `sky_lasso_tool.py` (included in this repo)

---

## Installation

```bash
git clone https://github.com/laurelhweiss/toast.git
```

---

## Usage

```python
%matplotlib widget

from hetdex_api.detections import Detections
from astropy.coordinates import SkyCoord
import astropy.units as u
from toast import QueryWidget

# Load Detections once per session (~40 seconds)
detects = Detections(curated_version='5.0.1')

# By detectid
qw = QueryWidget(detectid=3001361984, detections=detects)

# Or by coordinate
qw = QueryWidget(coords=SkyCoord(32.5314 * u.deg, -0.0705 * u.deg, frame='icrs'))
```

NOTE: Detections is only necessary when querying by HETDEX Detectid

**Workflow:** Lasso Select → Confirm Selection → Extract Spectra → (Reset)

After extraction, results are accessible as:

```python
qw.spec_table      # individual fiber spectra (astropy Table)
qw.stacked_spec    # stacked spectrum array
qw.stacked_err     # stacked error array
```

### QueryWidget parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `coords` | `SkyCoord` | — | Sky coordinate to center the widget on |
| `detectid` | `int` | — | HETDEX detectid; requires `detections` |
| `detections` | `Detections` | `None` | Pre-loaded Detections instance |
| `cutout_size` | `Quantity` | `10 arcsec` | Image cutout size |
| `spec_mode` | `str` | `'fibers'` | `'fibers'` or `'psf'` |
| `wave_range` | `tuple` | `(3540, 5450)` | X-axis wavelength range for spectrum plot|
| `flux_range` | `tuple` | `None` | Y-axis flux range for spectrum plot |
| `shotids` | `list` | `None` | Restrict to specific shots |
| `stat` | `str` | `'median'` | Stacking statistic |

---

## Files

- `toast.py` — main widget
- `sky_lasso_tool.py` — lasso selector class
- `toast.ipynb` — example notebook

---

## Acknowledgements

Built for use with the [HETDEX Survey](https://hetdex.org). Based upon structure of HETDEX API's `querywidget.py` 
