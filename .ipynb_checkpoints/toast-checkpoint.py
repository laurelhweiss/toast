from __future__ import print_function

"""
TOAST : The On-sky Area Stacking Tool
Author: Laurel H. Weiss

Made for use with HETDEX data release. Requires HETDEX API and Elixer
software packages. Also requires sky_lasso_tool.py

Built based on HETDEX API's querywidget.py (Author: Erin Mentuch Cooper). 
uses interactive 
lasso selector instead of aperture extractions

Returns a stacked spectrum of fiber spectra that fall within the drawn lasso region. 
Also returns individual fiber spectra. Option to grab PSF-extracted spectra at 
center of lasso (for each shot) or all fiber spectra within lasso region.
"""

import asyncio
import threading
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from astropy.wcs.utils import skycoord_to_pixel, pixel_to_skycoord
from astropy.visualization import ZScaleInterval
from astropy.table import Table, vstack
import astropy.units as u
from astropy.coordinates import SkyCoord
import ipywidgets as widgets
from ipywidgets import Layout

from hetdex_api.shot import get_fibers_table
from hetdex_api.survey import FiberIndex
from hetdex_api.config import HDRconfig
from hetdex_tools.get_spec import get_spectra

from astroquery.sdss import SDSS
from elixer import catalogs
from elixer import spectrum_utilities as ESU

from sky_lasso_tool import PixelLassoSelector

try:
    CONFIG_HDR5 = HDRconfig('hdr5')
except Exception as e:
    print("Warning! Cannot find or import HDRconfig from hetdex_api!!", e)

OPEN_DET_FILE = None
DET_HANDLE = None

WAVE = np.linspace(3470, 5540, 1036)

# convert ra/dec to pixel space for speed (probably in API already)
def _build_fiber_pixel_coords(wcs, ra_arr, dec_arr):
    coords = SkyCoord(ra_arr * u.deg, dec_arr * u.deg, frame="icrs")
    xs, ys = skycoord_to_pixel(coords, wcs)
    return np.column_stack([xs, ys])


# ---------------- Main Widget Class --------------------

class QueryWidget:
    def __init__(
        self,
        coords=None,
        detectid=None,
        aperture=3.0 * u.arcsec, #this goes into get_cutouts... NOT aperture size for get_spectra, check w Erin
        cutout_size=1.0 * u.arcmin,
        spec_mode="fibers",
        wave_range=(3540, 5450),
        flux_range=None,
        shotids=None,
        stat='median'
    ):
        """
        Parameters
        ----------
        spec_mode : str, optional
            'psf'    — PSF-weighted extraction at center of lasso
                        via get_spectra, one spectrum per shot
            'fibers' — Raw calibrated fiber spectra via get_fibers_table,
                       one spectrum per fiber per shot (default).
        shotids   : list, optional
                     - option to specify which shots to use, if None, all shots 
                     in region are used
        avg_type  : str, defaults to median
                    - 'mean', 'median', 'biweight', 'weighted_biweight'
        """
        
        if spec_mode not in ("psf", "fibers"):
            raise ValueError("spec_mode must be 'psf' or 'fibers'")

        self.spec_mode = spec_mode
        self.wave_range = wave_range
        self.flux_range = flux_range
        self.shotids = [int(s) for s in shotids] if shotids is not None else None
        self.stat = stat
        self.survey = "hdr5"  # fixed to HDR5
        self.detectid = detectid
        self.aperture = aperture
        self.cutout_size = cutout_size
        self.spec_table = None
        self.cutout = None
        
        self.stacked_spec = None
        self.stacked_err  = None

        # For panning
        self._zscale_limits = None
        self._load_generation = 0

        self._fiber_ra  = None
        self._fiber_dec = None
        self._fiber_center_coords = None
        self._fiber_cutout = None

        # Lasso addition (new)
        self._lasso_selector = None  
        self._fiber_index = None 
        self._fiber_table = None
        self._selected_fibers = None

        # Suppress RA/Dec callbacks when detectid is set
        self._suppress_coord_observe = False
        
        self.bottombox = widgets.Output(layout={"border": "1px solid black"})

        self.catlib = catalogs.CatalogLibrary()

        # Open FiberIndex here so survey.py's matplotlib.use("agg") fires before widget (annoying, don't touch)
        self._open_fiber_index()
        matplotlib.use("widget")

        if detectid:
            self.detectid = detectid
            self.update_det_coords()
        elif coords:
            self.coords = coords
            self.detectid = 1000000000
        else:
            self.coords = SkyCoord(191.663132 * u.deg, 50.712696 * u.deg, frame="icrs")
            self.detectid = 3003575145

        # Snapshot the resolved input coordinate and detectid so Reset can
        # return to exactly the state the widget was first called with.
        self._init_coords   = self.coords
        self._init_detectid = self.detectid

        # ----------- Set up new panel structure (matplotlib, previously ginga) ------------
        # Persistent image figure (left panel)
        with plt.ioff():
            self._fig, self._ax = plt.subplots(figsize=(6, 6))
            
        self._fig.subplots_adjust(left=0.12, right=0.97, top=0.97, bottom=0.08)
        self._ax.set_xlabel("x (pixels)", fontsize=8)
        self._ax.set_ylabel("y (pixels)", fontsize=8)
        self._ax.set_facecolor("black")  # visible placeholder until image loads

        canvas = self._fig.canvas
        canvas.toolbar_visible = False
        canvas.header_visible = False
        canvas.resizable = False
        canvas.layout = Layout(width="600px", height="600px", flex="0 0 auto")

        # Verify we have an interactive canvas, not Agg (annoying, don't touch)
        if "Agg" in type(canvas).__name__:
            raise RuntimeError(
                "matplotlib backend is Agg, not widget. "
                "Make sure '%matplotlib widget' is called before importing QueryWidget."
            )

        # Persistent spectra figure (right panel)
        with plt.ioff():
            self._spec_fig, self._spec_ax = plt.subplots(figsize=(4, 2))
        self._spec_fig.subplots_adjust(left=0.15, right=0.97, top=0.90, bottom=0.22)
        self._spec_ax.set_xlabel(r"$\mathrm{wavelength (\AA)}$", fontsize=8)
        self._spec_ax.set_ylabel(r"$\mathrm{f_{\lambda}~(10^{-17} ergs/s/cm^2/\AA)}$", fontsize=8)
        self._spec_ax.tick_params(labelsize=7)
        self._spec_ax.set_xlim(*self.wave_range)
        if self.flux_range is not None:
            self._spec_ax.set_ylim(*self.flux_range)

        spec_canvas = self._spec_fig.canvas
        spec_canvas.toolbar_visible = False
        spec_canvas.header_visible = False
        spec_canvas.resizable = False
        spec_canvas.layout = Layout(width="400px", height="200px", flex="0 0 auto")

        # --------------- Build widgets (some changes) --------------------

        self.detectbox = widgets.BoundedIntText(
            value=self.detectid,
            min=1000000000,
            max=6000000000,
            step=1,
            description="DetectID:",
            disabled=False,
        )
        self.im_ra = widgets.FloatText(
            value=self.coords.ra.value,
            step=0.001,
            description="RA (deg):",
            layout=Layout(width="20%"),
        )
        self.im_dec = widgets.FloatText(
            value=self.coords.dec.value,
            step=0.0005,
            description="DEC (deg):",
            layout=Layout(width="20%"),
        )
        self.lasso_button = widgets.Button(
            description="Lasso Select",
            button_style="success",
            icon="pencil",
            layout=Layout(width="140px"),
        )
        self.confirm_button = widgets.Button(
            description="Confirm Selection",
            button_style="info",
            disabled=True,
            layout=Layout(width="150px"),
        )
        self.reset_lasso_button = widgets.Button(
            description="Reset",
            button_style="warning",
            layout=Layout(width="80px"),
        )
        self.extract_button = widgets.Button(
            description="Extract Spectra",
            button_style="success",
            layout=Layout(width="130px"),
        )

        self.marker_table_output = widgets.Output(
            layout={"border": "1px solid black"}
        )
        self.textimpath = widgets.Text(
            description="Source: ", value="", layout=Layout(width="90%")
        )
        self.image_panel = widgets.VBox(
            [canvas, self.textimpath],
            layout=Layout(width="620px", flex="0 0 auto"),
        )
        self.rightbox = widgets.VBox([
            widgets.HBox([
                self.lasso_button,
                self.confirm_button,
                self.reset_lasso_button,
                self.extract_button,
            ]),
            self.marker_table_output,
            self._spec_fig.canvas,
        ], layout=Layout(width="620px"))

        self.topbox = widgets.HBox([
            self.detectbox,
            self.im_ra, self.im_dec,
        ])
        self.all_box = widgets.VBox([
            self.topbox,
            widgets.HBox([self.image_panel, self.rightbox]),
            self.bottombox,
        ])

        display(self.all_box)
        self.load_image()

        # callbacks
        self.detectbox.observe(self.on_det_change)
        self.lasso_button.on_click(self._activate_lasso)
        self.im_ra.observe(self.on_coords_change, names="value")
        self.im_dec.observe(self.on_coords_change, names="value")
        self.confirm_button.on_click(self.confirm_selection_click)
        self.reset_lasso_button.on_click(self.reset_lasso_on_click)
        self.extract_button.on_click(self.extract_on_click)

    def _safe_cla(self):
        """
        Disconnect lasso before clearing axes to avoid dangling artists.
        Call instead of self._ax.cla()
        """
        if self._lasso_selector is not None:
            self._lasso_selector.disconnect(keep_selection=False)
            self._lasso_selector = None
            self._reset_lasso_button_state()
            self.confirm_button.disabled = True
        self._ax.cla()

    def _draw_image(self):
        """
        Redraw cutout on the persistent matplotlib axes in native pixel
        coordinates, with RA/Dec tick labels derived from the image WCS.
        """
        self._safe_cla()

        if self.cutout is None:
            self._ax.set_facecolor("black")
            self._ax.text(0.5, 0.5, "No image loaded",
                          transform=self._ax.transAxes, color="white",
                          ha="center", va="center", fontsize=10)
            self._fig.canvas.draw_idle()
            return

        # Re-enable autoscaling (may have been frozen by _activate_lasso's scatter)
        self._ax.set_autoscale_on(True)

        data = self.cutout["cutout"].data
        wcs  = self.cutout["cutout"].wcs
        ny, nx = data.shape
        if self._zscale_limits is None:
            self._zscale_limits = ZScaleInterval().get_limits(data)
        vmin, vmax = self._zscale_limits

        # imshow in native pixel coords: extent=[left, right, bottom, top]
        # with origin="lower".  No arcsec conversion needed here.
        self._ax.imshow(data, origin="lower", cmap="gray_r",
                        vmin=vmin, vmax=vmax, interpolation="nearest",
                        extent=[-0.5, nx - 0.5, -0.5, ny - 0.5])

        # Label axes in RA/Dec by formatting tick positions through the WCS.
        # We pick a handful of evenly-spaced pixel ticks and convert them to
        # sky coordinates so the labels stay accurate after any pan/zoom.
        import matplotlib.ticker as mticker

        def _make_ra_formatter(wcs_ref, ny_ref):
            """Return a FuncFormatter that converts pixel x → RA string."""
            def _fmt(x, pos):
                try:
                    sky = pixel_to_skycoord(x, ny_ref / 2.0, wcs_ref)
                    return "{:.4f}°".format(sky.ra.deg)
                except Exception:
                    return ""
            return mticker.FuncFormatter(_fmt)

        def _make_dec_formatter(wcs_ref, nx_ref):
            """Return a FuncFormatter that converts pixel y → Dec string."""
            def _fmt(y, pos):
                try:
                    sky = pixel_to_skycoord(nx_ref / 2.0, y, wcs_ref)
                    return "{:.4f}°".format(sky.dec.deg)
                except Exception:
                    return ""
            return mticker.FuncFormatter(_fmt)

        self._ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=5, integer=True))
        self._ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5, integer=True))
        self._ax.xaxis.set_major_formatter(_make_ra_formatter(wcs, ny))
        self._ax.yaxis.set_major_formatter(_make_dec_formatter(wcs, nx))
        self._ax.tick_params(labelsize=7)
        plt.setp(self._ax.get_xticklabels(), rotation=30, ha="right")

        self._ax.set_xlabel('RA (deg)', fontsize=8)
        self._ax.set_ylabel('Dec (deg)', fontsize=8)
        self._ax.set_title("", fontsize=8)

        # Crosshair at the widget centre coordinate
        cx, cy = skycoord_to_pixel(self.coords, wcs)
        self._ax.axhline(cy, color="white", lw=0.3, alpha=0.4)
        self._ax.axvline(cx, color="white", lw=0.3, alpha=0.4)

        # 1.5" scale bar, anchored in axes-fraction coordinates so it always
        # sits in the lower-left corner regardless of image size or pan position.
        from astropy.wcs.utils import proj_plane_pixel_scales
        pix_scale_arcsec = proj_plane_pixel_scales(wcs)[0] * 3600.0  # arcsec/pixel
        bar_len_pix = 1.5 / pix_scale_arcsec  # length of 1.5" in image pixels

        # Scale bar position: 5% from left, 5% from bottom, (axes space for 
        # consistency). Convert to data (pixel) coords using the axes limits 
        # so the bar length is correct in pixel units.
        ax_x0_frac, ax_y0_frac = 0.05, 0.05
        xlim = self._ax.get_xlim()
        ylim = self._ax.get_ylim()
        x0 = xlim[0] + ax_x0_frac * (xlim[1] - xlim[0])
        y0 = ylim[0] + ax_y0_frac * (ylim[1] - ylim[0])
        x1 = x0 + bar_len_pix

        self._ax.plot([x0, x1], [y0, y0], color='blue', lw=1.5, solid_capstyle='butt',
                      zorder=15, transform=self._ax.transData)
        self._ax.text((x0 + x1) / 2.0, y0, '1.5"',
                      color='blue', fontsize=10, ha='center', va='bottom',
                      zorder=15, transform=self._ax.transData)

        self._fig.canvas.draw_idle()

    # -------------- Survey/coordinate helpers (unchanged) -------------------

    def update_coords(self):
        self.coords = SkyCoord(
            self.im_ra.value * u.deg, self.im_dec.value * u.deg, frame="icrs"
        )

    def on_coords_change(self, b):
        if self._suppress_coord_observe:
            return
        self.bottombox.clear_output()
        self.update_coords()
        self._suppress_coord_observe = True
        self.detectbox.value = 1000000000
        self._suppress_coord_observe = False
        self.detectid = 1000000000
        self._fiber_ra = None
        self._fiber_dec = None
        self._fiber_table = None
        # NOTE: _fiber_center_coords is intentionally NOT cleared here.
        # Panning fetches a new image centered on self.coords, but the red plus
        # should stay pinned to the original input coordinate.  _fiber_center_coords
        # is only reset by on_det_change (new target) or Reset (back to origin).
        self._fiber_cutout = None
        self.load_image()

    def on_det_change(self, b):
        self.bottombox.clear_output()
        self.detectid = self.detectbox.value
        self.update_det_coords()
        self._suppress_coord_observe = True
        self.im_ra.value = self.coords.ra.value
        self.im_dec.value = self.coords.dec.value
        self._suppress_coord_observe = False
        self._zscale_limits = None
        self._fiber_ra = None
        self._fiber_dec = None
        self._fiber_table = None
        self._fiber_center_coords = None
        self._fiber_cutout = None
        self.load_image()

    def update_det_coords(self):
        import tables as tb
        detectid_i = self.detectid
        global OPEN_DET_FILE, DET_HANDLE

        if "CONFIG_HDR5" not in globals():
            with self.bottombox:
                print("HDRconfig not available — cannot resolve detectid to coordinates.")

        # leaving for option to specify an HDR later -- currently, all detectids use HDR5.
        if (self.detectid >= 3000000000) * (self.detectid < 3090000000):
            self.det_file = CONFIG_HDR5.detecth5
        elif (self.detectid >= 3090000000) * (self.detectid < 3100000000):
            self.det_file = CONFIG_HDR5.contsourceh5
        elif (self.detectid >= 4000000000) * (self.detectid < 4090000000):
            self.det_file = CONFIG_HDR5.detecth5
        elif (self.detectid >= 4090000000) * (self.detectid < 4100000000):
            self.det_file = CONFIG_HDR5.contsourceh5
        elif (self.detectid >= 5000000000) * (self.detectid < 5090000000):
            self.det_file = CONFIG_HDR5.detecth5
        elif (self.detectid >= 5090000000) * (self.detectid < 5100000000):
            self.det_file = CONFIG_HDR5.contsourceh5
        else:
            with self.bottombox:
                print("{} does not look like a valid detectid. Checking coodinate input.".format(self.detectid))
            if not hasattr(self, "coords"):
                print("No coordinate specified. Plotting detectid 3003575145.")
                self.detectid = 3003575145
                self.coords = SkyCoord(191.663132 * u.deg, 50.712696 * u.deg, frame="icrs")
            return

        if OPEN_DET_FILE is None:
            OPEN_DET_FILE = self.det_file
            DET_HANDLE = tb.open_file(self.det_file, "r")
        elif self.det_file != OPEN_DET_FILE:
            DET_HANDLE.close()
            OPEN_DET_FILE = self.det_file
            try:
                DET_HANDLE = tb.open_file(self.det_file, "r")
            except Exception:
                with self.bottombox:
                    print("Could not open {}".format(self.det_file))
        try:
            det_row = DET_HANDLE.root.Detections.read_where("detectid == detectid_i")
            if np.size(det_row) > 0:
                self.coords = SkyCoord(
                    det_row["ra"][0] * u.deg, det_row["dec"][0] * u.deg
                )
            else:
                with self.bottombox:
                    print("{} is not in the {} detect database".format(
                        detectid_i, self.survey))
        except Exception:
            with self.bottombox:
                print("{} is not in the {} detect database".format(
                    detectid_i, self.survey))

    # ---------------- Load image (some changes) --------------------

    def load_image(self):
        im_size = self.cutout_size.to(u.arcsec).value
        mag_aperture = self.aperture.to(u.arcsec).value

        self._load_generation += 1
        my_gen = self._load_generation

        def _fetch():
            try:
                self.cutout = self.catlib.get_cutouts(
                    position=self.coords,
                    side=im_size,
                    aperture=mag_aperture,
                    dynamic=False,
                    filter=["r", "g", "f606W"],
                    first=True,
                )[0]
                self.im_path = self.cutout["path"]
            except Exception:
                # catlib failed — try SDSS fallback
                try:
                    try:
                        sdss_im = SDSS.get_images(coordinates=self.coords, band="g")
                        im = sdss_im[0][0]
                    except Exception:
                        sdss_im = SDSS.get_images(
                            coordinates=self.coords, band="g", radius=30.0 * u.arcsec
                        )
                        im = sdss_im[0][0]
                    from astropy.wcs import WCS
                    self.cutout = {
                        "cutout": type("C", (), {
                            "data": im.data,
                            "wcs": WCS(im.header),
                        })()
                    }
                    self.im_path = "SDSS Astroquery result"
                except Exception as e:
                    self.cutout = None
                    self._zscale_limits = None
                    self.im_path = "Image load failed: {}".format(e)

        def _finish():
            if my_gen != self._load_generation:
                return
            self.textimpath.value = self.im_path
            self._draw_image()
            self._load_fibers_async()

        def _run():
            _fetch()
            try:
                loop = asyncio.get_event_loop()
                loop.call_soon_threadsafe(_finish)
            except Exception:
                # Fallback: if no running loop (e.g. test context), call directly
                _finish()

        threading.Thread(target=_run, daemon=True).start()

    # --------------- Lasso feature (new) ---------------------
    # Replaces aperture markers

    def _open_fiber_index(self):
        if self._fiber_index is None:
            self._fiber_index = FiberIndex()

    def _close_fiber_index(self):
        if self._fiber_index is not None:
            try:
                self._fiber_index.close()
            except Exception:
                pass
            self._fiber_index = None

    def _fiberid_to_shot(self, fid):
        if isinstance(fid, bytes):
            fid = fid.decode()
        return fid.split("_")[0]

    def _get_fiber_coords(self, coords=None, cutout=None):
        """
        Query FiberIndex for all fibers in current region.
        Stores as self._fiber_table for later use in extraction.
        Returns (ra_arr, dec_arr) for display in lasso.
        Falls back to a pixel grid if query fails.
        """
        if coords is None:
            coords = self.coords
        if cutout is None:
            cutout = self.cutout
        try:
            self._open_fiber_index()
            self._fiber_table = self._fiber_index.query_region(
                coords,
                radius=self.cutout_size.to(u.arcsec).value / 2.0 * u.arcsec,
            )
            if self._fiber_table is not None and len(self._fiber_table) > 0:
                # Filter to requested shots if specified
                if self.shotids is not None:
                    fiber_shots = np.array(
                        [int(self._fiberid_to_shot(fid))
                         for fid in self._fiber_table["fiber_id"]]
                    )
                    mask = np.isin(fiber_shots, self.shotids)
                    missing = set(self.shotids) - set(fiber_shots[mask])
                    for s in sorted(missing):
                        with self.bottombox:
                            print("Skipped {}: no fibers at this position".format(s))
                    self._fiber_table = self._fiber_table[mask]
                return (np.array(self._fiber_table["ra"]),
                        np.array(self._fiber_table["dec"]))
        except Exception as e:
            with self.bottombox:
                print("Could not load fiber positions ({}). "
                      "Using pixel grid fallback.".format(e))

        # Fallback: uniform grid across cutout pixels
        wcs = cutout["cutout"].wcs
        ny, nx = cutout["cutout"].data.shape
        xs = np.linspace(5, nx - 5, 40)
        ys = np.linspace(5, ny - 5, 40)
        xg, yg = np.meshgrid(xs, ys)
        sky = pixel_to_skycoord(xg.ravel(), yg.ravel(), wcs)
        self._fiber_table = None
        return sky.ra.deg, sky.dec.deg

    def _draw_fibers(self):
        """
        Overlay fiber positions on the image axes in native pixel coordinates.
        Avoids transformation issues
        """
        if self._fiber_ra is None or self._fiber_dec is None:
            return
        if self.cutout is None:
            return

        # Use the WCS of the currently displayed image (before/after pan) so fiber
        # positions are consistent with imshow extent.
        wcs = self.cutout["cutout"].wcs

        pixel_coords = _build_fiber_pixel_coords(wcs, self._fiber_ra, self._fiber_dec)

        # Freeze axes limits so fibers outside image extent don't trigger
        # autoscaling (black edge issue).
        self._ax.set_autoscale_on(False)

        # Plot fiber positions
        self._ax.scatter(
            pixel_coords[:, 0], pixel_coords[:, 1],
            s=25, c='blue', marker='.', linewidths=0, zorder=5
        )

        # Plus sign located at original input coordinate.
        anchor = self._fiber_center_coords if self._fiber_center_coords is not None else self.coords
        ax_px, ax_py = skycoord_to_pixel(anchor, wcs)
        self._ax.plot(ax_px, ax_py, '+', color='k', markersize=12,
                      markeredgewidth=1.5, zorder=10)

        self._fig.canvas.draw_idle()

    def _load_fibers_async(self):
        """
        Get fiber positions in the background and overlay on image.
        Results saved in self._fiber_ra/self._fiber_dec so _activate_lasso
        can reuse them.
        
        Used so quick RA/Dec arrow clicks don't change self.coords mid-query,
        causing fiber positions for one coordinate to be plotted as
        arcsec offsets new coordinate.
        """
        if self.cutout is None:
            return

        my_gen = self._load_generation
        query_coords = self.coords
        query_cutout = self.cutout

        def _run():
            try:
                ra_arr, dec_arr = self._get_fiber_coords(query_coords, query_cutout)
            except Exception as e:
                with self.bottombox:
                    print("Could not load fiber positions: {}".format(e))
                return

            def _finish():
                if my_gen != self._load_generation:
                    return 
                self._fiber_ra  = ra_arr
                self._fiber_dec = dec_arr

                if self._fiber_center_coords is None:
                    self._fiber_center_coords = query_coords
                self._fiber_cutout = query_cutout
                self._draw_fibers()

            try:
                loop = asyncio.get_event_loop()
                loop.call_soon_threadsafe(_finish)
            except Exception:
                _finish()

        threading.Thread(target=_run, daemon=True).start()

    def _activate_lasso(self, b=None):
        """
        Attach PixelLassoSelector to the image axes.
        
        Fiber positions are already shown (loaded by _load_fibers_async)
        """
        self.bottombox.clear_output()
        self.marker_table_output.clear_output()

        if self.cutout is None:
            with self.bottombox:
                print("No cutout loaded yet.")
            self._reset_lasso_button_state()
            return

        self.lasso_button.button_style = "danger"
        self.confirm_button.disabled = True

        if self._fiber_ra is None or self._fiber_dec is None:
            with self.bottombox:
                print("Fiber positions still loading, please try again in a moment.")
            self._reset_lasso_button_state()
            self._load_fibers_async()
            return

        try:
            cutout_for_wcs = self._fiber_cutout if self._fiber_cutout is not None else self.cutout
            wcs = cutout_for_wcs["cutout"].wcs

            # Convert fiber RA/Dec to pixel coords via the same WCS used by imshow.  
            # Lasso selector must use the same coordinate space as displayed image
            pixel_coords = _build_fiber_pixel_coords(wcs, self._fiber_ra, self._fiber_dec)

            from astropy.wcs.utils import proj_plane_pixel_scales
            pix_scale_arcsec = proj_plane_pixel_scales(wcs)[0] * 3600.0  # arcsec/pixel
            marker_px = 1.5 / pix_scale_arcsec  # pixels

            self._ax.set_title(
                "Draw lasso [{} mode] — click 'Confirm Selection' when done".format(
                    self.spec_mode),
                fontsize=8,
            )
            self._fig.canvas.draw_idle()

            # PixelLassoSelector works in axes (data) coordinates.
            self._lasso_selector = PixelLassoSelector(
                fig=self._fig, ax=self._ax,
                data_coords=pixel_coords,
                ra=self._fiber_ra, dec=self._fiber_dec,
                wcs=wcs,
                marker_size_pixels=marker_px,
            )

            self._fig.canvas.draw_idle()
            self.confirm_button.disabled = False
            with self.bottombox:
                print("Lasso active [{} mode] — draw a region, "
                      "then click 'Confirm Selection'.".format(self.spec_mode))

        except Exception as e:
            self._reset_lasso_button_state()
            with self.bottombox:
                print("Error activating lasso: {}".format(e))


    def _reset_lasso_button_state(self):
        """
        Usage: click/unclick lasso button for new selection. 
        
        May remove since re-drawing lasso also does this. tbd
        """
        self.lasso_button.description = "Lasso Select"
        self.lasso_button.button_style = "success"

    def _deactivate_lasso(self, confirm=True):
        """
        Grab selection, disconnect lasso
        """
        if self._lasso_selector is not None:
            if confirm:
                self._harvest_selection()
            self._lasso_selector.disconnect(keep_selection=True)
            self._lasso_selector = None

        self.confirm_button.disabled = True
        self._reset_lasso_button_state()

    def confirm_selection_click(self, b):
        """
        Grab current lasso selection and exit lasso mode.
        """
        self._deactivate_lasso(confirm=True)

    def _harvest_selection(self):
        """
        Load RA/Dec from lasso into self._lasso_coords.
        """
        if self._lasso_selector is None:
            return

        ind = self._lasso_selector.ind
        ra_sel = self._lasso_selector.selected_ra
        dec_sel = self._lasso_selector.selected_dec
        self.marker_table_output.clear_output()

        if len(ind) == 0:
            with self.marker_table_output:
                print("No fibers inside lasso region.")
            self._selected_fibers = None
            return

        self._selected_fibers = (self._fiber_table[ind]
                                  if self._fiber_table is not None
                                  else Table({"ra": ra_sel, "dec": dec_sel}))

        n = len(ind)
        with self.marker_table_output:
            print("{} fiber(s) selected [{} mode].".format(n, self.spec_mode))
            if self._fiber_table is not None:
                shots = np.unique(
                    [self._fiberid_to_shot(fid)
                     for fid in self._selected_fibers["fiber_id"]]
                )
                print("Across {} shot(s): {}".format(len(shots), ", ".join(shots)))

        self.bottombox.clear_output()
        with self.bottombox:
            print("{} fiber(s) selected. "
                  "Press 'Extract Spectra' to retrieve.".format(n))

    def reset_lasso_on_click(self, b):
        """
        Revert to the original input coordinate and reload the image,
        discarding any lasso selection, panned position, and extracted spectra.
        """
        self._selected_fibers = None
        self.spec_table = None
        self.stacked_spec = None
        self.stacked_err  = None

        # Restore the original coordinate and detectid
        self.coords   = self._init_coords
        self.detectid = self._init_detectid
        self._zscale_limits = None  # recalculate ZScale for the original image

        self._suppress_coord_observe = True
        self.im_ra.value    = self.coords.ra.value
        self.im_dec.value   = self.coords.dec.value
        self.detectbox.value = self.detectid
        self._suppress_coord_observe = False

        # Clear cached fiber data so they are re-fetched for the original coords
        self._fiber_ra           = None
        self._fiber_dec          = None
        self._fiber_table        = None
        self._fiber_center_coords = None
        self._fiber_cutout       = None

        # Clear outputs
        self.marker_table_output.clear_output()
        self.bottombox.clear_output()

        # Clear spectra plot
        self._spec_ax.cla()
        self._spec_ax.set_xlabel(r"$\mathrm{wavelength (\AA)}$", fontsize=8)
        self._spec_ax.set_xlim(*self.wave_range)
        if self.flux_range is not None:
            self._spec_ax.set_ylim(*self.flux_range)
        self._spec_ax.tick_params(labelsize=7)
        self._spec_fig.canvas.draw_idle()

        # Reload image and fibers at the original coordinate
        self.load_image()

    def close(self):
        """
        Restores normal notebook keyboard/mouse behaviour. 
        (annyoing, don't touch)
        """
        try:
            self._deactivate_lasso(confirm=False)
        except Exception:
            pass
        try:
            self._close_fiber_index()
        except Exception:
            pass
        try:
            self._fig.canvas.close()
            plt.close(self._fig)
        except Exception:
            pass
        try:
            self._spec_fig.canvas.close()
            plt.close(self._spec_fig)
        except Exception:
            pass
        try:
            for w in self.all_box.children:
                w.close()
            self.all_box.close()
        except Exception:
            pass

    # -------------------- Extract Spec (mostly unchanged) --------------------------
    # added two modes: PSF extract or use fiber spectra

    def extract_on_click(self, b):
        self.bottombox.clear_output()
        if self._selected_fibers is None or len(self._selected_fibers) == 0:
            with self.bottombox:
                print("No lasso selection — use 'Lasso Select', draw a region, "
                      "then click 'Confirm Selection'.")
            return
        if self.spec_mode == "fibers":
            self._extract_fibers()
        else:
            self._extract_psf()

    def _extract_fibers(self):
        """
        Grab individual calibrated fiber spectra via get_fibers_table,
        looping over shotids in the lasso selection. Stored in 
        self.spec_table. stack in stacked_spec / stacked_err
        """
        shotids = np.unique(
            [self._fiberid_to_shot(fid)
             for fid in self._selected_fibers["fiber_id"]]
        )
        tables = []
        with self.bottombox:
            for shot in shotids:
                print("Fetching fibers for shot {}...".format(shot))
                try:
                    t = get_fibers_table(
                        int(shot),
                        coords=self.coords,
                        radius=self.cutout_size.to(u.arcsec).value / 2.0,
                        survey=self.survey,
                    )
                    if t is not None and len(t) > 0:
                        lasso_fids = set(self._selected_fibers["fiber_id"])
                        mask = np.array([fid in lasso_fids for fid in t["fiber_id"]])
                        if mask.any():
                            tables.append(t[mask])
                except Exception as e:
                    print("  Could not fetch shot {}: {}".format(shot, e))

        if not tables:
            with self.bottombox:
                print("No fiber spectra returned.")
            return

        self.spec_table = vstack(tables)
        self._draw_spec_axes()

    def _extract_psf(self):
        """
        Single PSF extraction at the visual center of the lasso selection.
        One extraction per shot.
        """
        ra_mean = np.mean(self._selected_fibers["ra"])
        dec_mean = np.mean(self._selected_fibers["dec"])

        center = SkyCoord(ra_mean * u.deg,dec_mean * u.deg,frame="icrs")

        # Compute PSF centre in pixel space for the marker (image is now in pixel coords)
        cutout_for_wcs = self._fiber_cutout if self._fiber_cutout is not None else self.cutout
        wcs = cutout_for_wcs["cutout"].wcs

        # Use the same center the lasso/fiber dots were drawn against, not
        # live self.coords, in case the user nudged RA/Dec after confirming
        # the selection but before clicking Extract Spectra.
        px, py = skycoord_to_pixel(center, wcs)
        self._ax.scatter(px, py, s=50, marker="x", color="blue",
                         linewidths=1, zorder=100)
        self._fig.canvas.draw_idle()

        fiber_ids = np.array(self._selected_fibers["fiber_id"])
        unique_shots = np.unique([self._fiberid_to_shot(fid) for fid in fiber_ids])

        # Extract spectra
        tables = []
        with self.bottombox:
            print("PSF center: RA={:.6f}, Dec={:.6f}".format(ra_mean,dec_mean))
        
            for shot in unique_shots:
                print("PSF-extracting shot {}...".format(shot))
                try:
                    t = get_spectra(center,shotid=int(shot),survey=self.survey)
                    if t is not None and len(t) > 0:
                        tables.append(t)
                except Exception as e:
                    print("Could not extract shot {}: {}".format(shot,e))

        if not tables:
            with self.bottombox:
                print("No PSF spectra returned.")
            return

        self.spec_table = vstack(tables)
        self._draw_spec_axes()

    # -------------------- Plot Spec (some changes) --------------------------

    def _draw_spec_axes(self):
        """
        Plot all spectra in self.spec_table on right-panel axes.
        """
        if self.spec_table is None or len(self.spec_table) == 0:
            return

        self._spec_ax.cla()
        self._spec_ax.set_xlabel(r"$\mathrm{wavelength (\AA)}$", fontsize=8)
        self._spec_ax.set_xlim(*self.wave_range)
        if self.flux_range is not None:
            self._spec_ax.set_ylim(*self.flux_range)
        self._spec_ax.tick_params(labelsize=7)

        all_specs, all_errs, all_waves = [], [], []
        if self.spec_mode == "fibers":
            self._spec_ax.set_ylabel(r"$\mathrm{f_{\lambda}~(10^{-17} ergs/s/cm^2/\AA)}$", fontsize=8)
            self._spec_ax.set_title(
                "{} fiber spectra".format(len(self.spec_table)), fontsize=8
            )
            for row in self.spec_table:
                shotid = self._fiberid_to_shot(row["fiber_id"])
                self._spec_ax.plot(WAVE, row["calfib"], lw=0.7, alpha=0.8,
                                   label=str(row["fiber_id"]))
                all_specs.append(row["calfib"])
                all_errs.append(row["calfibe"])
                all_waves.append(WAVE.tolist())
        else:
            self._spec_ax.set_ylabel(r"$\mathrm{f_{\lambda}~(10^{-17} ergs/s/cm^2/\AA)}$", fontsize=8)
            self._spec_ax.set_title(
                "{} PSF spectra".format(len(self.spec_table)), fontsize=8
            )
            for row in self.spec_table:
                self._spec_ax.plot(WAVE, row["spec"], lw=0.7, alpha=0.8,
                                   label=str(row["shotid"]))
                all_specs.append(row["spec"])
                all_errs.append(row["spec_err"])
                all_waves.append(WAVE.tolist())
                
        stack_spec, stack_err, stack_wave, cc = ESU.stack_spectra(all_specs, all_errs, all_waves, avg_type=self.stat) 
        self._spec_ax.plot(stack_wave, stack_spec, lw=1, alpha=1, color='k')
        
        self.stacked_spec = stack_spec
        self.stacked_err  = stack_err
        
        self._spec_fig.canvas.draw_idle()