# TOAST — The On-sky Area Stacking Tool

An interactive Jupyter widget for extracting and stacking fiber spectra from HETDEX data using a freehand lasso selector on a sky image.

Built on top of the [HETDEX API](https://github.com/HETDEX/hetdex_api) and [ELiXer](https://github.com/HETDEX/elixer), based on the original `querywidget.py` by Erin Mentuch Cooper.

**Author:** Laurel H. Weiss

---

## Features

- Displays a sky cutout centered on a coordinate or HETDEX detectid
- Overlays fiber positions from `get_fibers_table`
- Freehand lasso selection of fibers directly on the image
- Two extraction modes:
  - `fibers` — raw calibrated fiber spectra for all selected fibers (default)
  - `psf` — PSF-weighted extraction at the lasso center via `get_spectra`
- Stacked spectrum with selectable statistic (`mean`, `median`, `biweight`, `weighted_biweight`)
- Pan to any RA/Dec with the arrow controls; Reset returns to the original input coordinate
- 1.5" scale bar on the image; interactive pan/zoom on the spectrum panel

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

No package installation required. Clone the repo and ensure the requirements above are available in your environment, then run the example notebook.

```bash
git clone https://github.com/<your-repo>/toast.git
cd toast
```

---

## Usage

Add `%matplotlib widget` at the top of your notebook, then:

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
| `coords` | `SkyCoord` | — | Sky coordinate to centre the widget on |
| `detectid` | `int` | — | HETDEX detectid; requires `detections` |
| `detections` | `Detections` | `None` | Pre-loaded Detections instance |
| `cutout_size` | `Quantity` | `1.0 arcmin` | Image cutout size |
| `spec_mode` | `str` | `'fibers'` | `'fibers'` or `'psf'` |
| `wave_range` | `tuple` | `(3540, 5450)` | Wavelength range in Å |
| `flux_range` | `tuple` | `None` | Y-axis flux range for spectrum plot |
| `shotids` | `list` | `None` | Restrict to specific shotids |
| `stat` | `str` | `'median'` | Stacking statistic |

---

## Files

- `toast.py` — main widget
- `sky_lasso_tool.py` — lasso selector class
- `toast.ipynb` — example notebook

---

## Acknowledgements

Built for use with the [HETDEX Survey](https://hetdex.org). Based on `querywidget.py` by Erin Mentuch Cooper.
