from __future__ import print_function

"""

Widget to query HETDEX spectra via elixer catalog API and
HETDEX API tools

Authors: Erin Mentuch Cooper

Date: November 9, 2019

----- Rewrite: (5/5/26) -----

TOAST-ER : The On-sky Area Stacking Tool - Extended Region

Include TOAST lasso selector on widget instead of aperture 
extractions - made functional for whole HETDEX dataset. Returns 
fiber spectra within drawn lasso region. Option to grab 
PSF-extracted spectra at center of lasso (for each shot) or all 
fiber spectra within lasso region.

Authors: Laurel H. Weiss

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

        # Persist ZScale limits across RA/Dec pans; reset on new DetectID
        self._zscale_limits = None

        # Generation counter: incremented on every load_image() call so that
        # callbacks from superseded (stale) fetches are silently discarded.
        self._load_generation = 0

        # Cached fiber positions for the current coords; cleared on coord/det change
        # so the next load re-fetches for the new position.
        self._fiber_ra  = None
        self._fiber_dec = None

        # Lasso addition (new)
        self._lasso_selector = None  
        self._fiber_index = None 
        self._fiber_table = None
        self._selected_fibers = None

        # Suppress RA/Dec observe callbacks when detectid sets them programmatically
        self._suppress_coord_observe = False

        # Initialised early so update_det_coords (called below) can write to it
        # before the rest of the widget layout is built
        self.bottombox = widgets.Output(layout={"border": "1px solid black"})

        self.catlib = catalogs.CatalogLibrary()

        # Open FiberIndex here so survey.py's matplotlib.use("agg") fires
        # now, before widget (annoying, don't touch)
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
        """Disconnect lasso before clearing axes to avoid dangling artist refs.
        Always call this instead of self._ax.cla() directly."""
        if self._lasso_selector is not None:
            self._lasso_selector.disconnect(keep_selection=False)
            self._lasso_selector = None
            self._reset_lasso_button_state()
            self.confirm_button.disabled = True
        self._ax.cla()

    def _draw_image(self):
        """Redraw the cutout on the persistent matplotlib axes with arcsec
        offset axes centred on self.coords."""
        # Always clear via _safe_cla so any live lasso is disconnected before
        # its artist refs are invalidated by cla().
        self._safe_cla()

        if self.cutout is None:
            # Show an explicit error state rather than leaving a stale/blank image
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

        # Pixel scale in arcsec/pixel from WCS
        from astropy.wcs.utils import proj_plane_pixel_scales
        pix_scale = proj_plane_pixel_scales(wcs)[0] * 3600.0  # arcsec/pixel

        # Centre pixel (fractional) of the input coords
        cx, cy = skycoord_to_pixel(self.coords, wcs)

        # Extent in arcsec: left, right, bottom, top
        left   = -cx * pix_scale
        right  = (nx - 1 - cx) * pix_scale
        bottom = -cy * pix_scale
        top    = (ny - 1 - cy) * pix_scale

        self._ax.imshow(data, origin="lower", cmap="gray_r",
                        vmin=vmin, vmax=vmax, interpolation="nearest",
                        extent=[right, left, bottom, top]) #double check if +RA is left and -RA is right w Dustin
        self._ax.set_xlabel('RA offset (arcsec)', fontsize=8)
        self._ax.set_ylabel('Dec offset (arcsec)', fontsize=8)
        self._ax.set_title("", fontsize=8)
        self._ax.axhline(0, color="white", lw=0.3, alpha=0.4)
        self._ax.axvline(0, color="white", lw=0.3, alpha=0.4)
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
        # Clear detectid — coords were set manually, not from a detection
        self._suppress_coord_observe = True
        self.detectbox.value = 1000000000
        self._suppress_coord_observe = False
        self.detectid = 1000000000
        self._fiber_ra = None
        self._fiber_dec = None
        self._fiber_table = None
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
        self.load_image()

    def update_det_coords(self):
        import tables as tb
        detectid_i = self.detectid
        global OPEN_DET_FILE, DET_HANDLE

        # Fixed to HDR5; detectid prefix 5xxxxxxxx = detections, 509xxxxxxx = contsource
        if "CONFIG_HDR5" not in globals():
            with self.bottombox:
                print("HDRconfig not available — cannot resolve detectid to coordinates.")
            if not hasattr(self, "coords"):
                self.coords = SkyCoord(191.663132 * u.deg, 50.712696 * u.deg, frame="icrs")
            return
        if (self.detectid >= 5000000000) and (self.detectid < 5090000000):
            self.det_file = CONFIG_HDR5.detecth5
        elif (self.detectid >= 5090000000) and (self.detectid < 5100000000):
            self.det_file = CONFIG_HDR5.contsourceh5
        else:
            with self.bottombox:
                print("{} does not look like an HDR5 detectid".format(self.detectid))
            # Ensure self.coords is always set so the widget build (im_ra/im_dec)
            # does not crash with AttributeError on self.coords.ra.value
            if not hasattr(self, "coords"):
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

    # ---------------- Load image (mostly unchaged) --------------------

    def load_image(self):
        im_size = self.cutout_size.to(u.arcsec).value
        mag_aperture = self.aperture.to(u.arcsec).value

        # Increment generation so any in-flight fetch from a previous load_image()
        # call (e.g. from a rapid RA/Dec change) is silently discarded on return.
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
                    # Both image sources failed — clear cutout and record error so
                    # _draw_image() shows the explicit error state instead of a stale image.
                    self.cutout = None
                    self._zscale_limits = None
                    self.im_path = "Image load failed: {}".format(e)

        def _finish():
            # Discard if a newer load_image() call has already superseded us
            if my_gen != self._load_generation:
                return
            self.textimpath.value = self.im_path
            self._draw_image()
            # Kick off background fiber fetch immediately after image draws
            self._load_fibers_async()

        def _run():
            _fetch()
            # Post _finish back to the main IOLoop so all canvas mutations
            # (cla, imshow, draw_idle) run on the main thread, not the daemon thread.
            # This is the fix for intermittent blank images caused by off-thread drawing.
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

    def _get_fiber_coords(self):
        """
        Query FiberIndex for all fibers in current region.
        Stores as self._fiber_table for later use in extraction.
        Returns (ra_arr, dec_arr) for display in lasso.
        Falls back to a pixel grid if query fails.
        """
        try:
            self._open_fiber_index()
            self._fiber_table = self._fiber_index.query_region(
                self.coords,
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
        wcs = self.cutout["cutout"].wcs
        ny, nx = self.cutout["cutout"].data.shape
        xs = np.linspace(5, nx - 5, 40)
        ys = np.linspace(5, ny - 5, 40)
        xg, yg = np.meshgrid(xs, ys)
        sky = pixel_to_skycoord(xg.ravel(), yg.ravel(), wcs)
        self._fiber_table = None
        return sky.ra.deg, sky.dec.deg

    def _draw_fibers(self):
        """Overlay cached fiber positions on the image axes.
        Called after image load and after reset, so fibers are always visible.
        No-op if fiber cache is empty or no cutout is loaded."""
        if self._fiber_ra is None or self._fiber_dec is None:
            return
        if self.cutout is None:
            return
        wcs = self.cutout["cutout"].wcs

        from astropy.wcs.utils import proj_plane_pixel_scales
        pixel_coords = _build_fiber_pixel_coords(wcs, self._fiber_ra, self._fiber_dec)
        pix_scale = proj_plane_pixel_scales(wcs)[0] * 3600.0
        cx, cy = skycoord_to_pixel(self.coords, wcs)
        fiber_x_arcsec = (pixel_coords[:, 0] - cx) * pix_scale
        fiber_y_arcsec = (pixel_coords[:, 1] - cy) * pix_scale

        # Freeze axes limits so fiber artists outside the image extent don't
        # trigger autoscaling and reveal black edges
        self._ax.set_autoscale_on(False)

        # Plot all fiber positions as small black dots for reference
        self._ax.scatter(
            fiber_x_arcsec, fiber_y_arcsec,
            s=20, c="black", marker=".", linewidths=0, zorder=5
        )

        # Mark the input RA/Dec with a plus sign at the origin
        self._ax.plot(0, 0, "+", color="red", markersize=12,
                      markeredgewidth=1.5, zorder=10)

        self._fig.canvas.draw_idle()

    def _load_fibers_async(self):
        """Fetch fiber positions in the background and overlay them on the image.
        Results are cached in self._fiber_ra / self._fiber_dec so _activate_lasso
        can reuse them without a second fetch."""
        if self.cutout is None:
            return

        # Capture generation so a stale fetch doesn't overwrite newer coords
        my_gen = self._load_generation

        def _run():
            try:
                ra_arr, dec_arr = self._get_fiber_coords()
            except Exception as e:
                with self.bottombox:
                    print("Could not load fiber positions: {}".format(e))
                return

            def _finish():
                if my_gen != self._load_generation:
                    return  # coords changed while we were fetching
                self._fiber_ra  = ra_arr
                self._fiber_dec = dec_arr
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
        Fiber positions are already shown (loaded by _load_fibers_async after
        image load), so this is now just a selector-attach step.
        If fibers haven't arrived yet (slow network), kicks off a fresh fetch.
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
            # Fibers haven't loaded yet (e.g. still fetching after image load).
            # Trigger a fresh fetch; _load_fibers_async will call _draw_fibers
            # when done but won't attach the lasso — inform the user to retry.
            with self.bottombox:
                print("Fiber positions still loading, please try again in a moment.")
            self._reset_lasso_button_state()
            self._load_fibers_async()
            return

        try:
            wcs = self.cutout["cutout"].wcs

            from astropy.wcs.utils import proj_plane_pixel_scales
            pixel_coords = _build_fiber_pixel_coords(wcs, self._fiber_ra, self._fiber_dec)
            pix_scale = proj_plane_pixel_scales(wcs)[0] * 3600.0
            cx, cy = skycoord_to_pixel(self.coords, wcs)
            fiber_x_arcsec = (pixel_coords[:, 0] - cx) * pix_scale #I *think* this is correct to have RA increasing to the left
            fiber_y_arcsec = (pixel_coords[:, 1] - cy) * pix_scale

            self._ax.set_title(
                "Draw lasso [{} mode] — click 'Confirm Selection' when done".format(
                    self.spec_mode),
                fontsize=8,
            )
            self._fig.canvas.draw_idle()

            # PixelLassoSelector works in display (axes) coordinates;
            # pass arcsec offsets so it matches the arcsec axes.
            arcsec_coords = np.column_stack([fiber_x_arcsec, fiber_y_arcsec])
            self._lasso_selector = PixelLassoSelector(
                fig=self._fig, ax=self._ax,
                data_coords=arcsec_coords,
                ra=self._fiber_ra, dec=self._fiber_dec,
                wcs=wcs,
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
        Usage: click/unclick lasso button for new selection
        """
        self.lasso_button.description = "Lasso Select"
        self.lasso_button.button_style = "success"

    def _deactivate_lasso(self, confirm=True):
        """
        Grab selection, disconnect the lasso, close the
        figure, and restore the ginga panel in-place.
        """
        if self._lasso_selector is not None:
            if confirm:
                self._harvest_selection()
            self._lasso_selector.disconnect(keep_selection=True)
            self._lasso_selector = None

        self.confirm_button.disabled = True
        self._reset_lasso_button_state()
        # Redraw image after disconnect so cla() doesn't hit dangling artists

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
        Close any active lasso, discard selection, clear outputs
        """
        # Clear selection state — but NOT _fiber_table, which holds the full
        # fiber position data for the current coords and is needed by
        # _harvest_selection after a new lasso draw. _fiber_table is only
        # cleared by on_coords_change / on_det_change (new position).
        self._selected_fibers = None
        self.spec_table = None

        self.stacked_spec = None
        self.stacked_err  = None

        # Clear outputs
        self.marker_table_output.clear_output()
        self.bottombox.clear_output()

        # --- Batch all cla() calls before either draw_idle() ---
        # _draw_image() calls _safe_cla() which disconnects the lasso and clears
        # the image axes. _spec_ax.cla() clears the spectra axes. Both happen
        # before either canvas flushes, so neither draw_idle() fires mid-clear.
        self._draw_image()   # internally: _safe_cla() -> lasso disconnect -> cla() -> imshow()
        self._draw_fibers()  # re-overlay cached fiber dots after the image is redrawn

        # Clear spectra plot
        self._spec_ax.cla()
        self._spec_ax.set_xlabel(r"$\mathrm{wavelength (\AA)}$", fontsize=8)
        self._spec_ax.set_xlim(*self.wave_range)
        if self.flux_range is not None:
            self._spec_ax.set_ylim(*self.flux_range)
        self._spec_ax.tick_params(labelsize=7)

        # Single flush for both canvases after all cla() work is done
        self._fig.canvas.draw_idle()
        self._spec_fig.canvas.draw_idle()

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

        # Compute visual center in displayed coordinates
        wcs = self.cutout["cutout"].wcs

        coords = SkyCoord(self._selected_fibers["ra"] * u.deg,self._selected_fibers["dec"] * u.deg,frame="icrs")
        px, py = skycoord_to_pixel(coords, wcs)

        from astropy.wcs.utils import proj_plane_pixel_scales
        pix_scale = proj_plane_pixel_scales(wcs)[0] * 3600.0

        cx, cy = skycoord_to_pixel(self.coords, wcs)

        # x-axis flipped by imshow(extent=[right,left,...])
        x_arcsec = (px - cx) * pix_scale
        y_arcsec = (py - cy) * pix_scale

        # Visual centroid in displayed coordinates
        psf_x = np.mean(x_arcsec)
        psf_y = np.mean(y_arcsec)

        # Draw PSF center marker
        self._ax.scatter(psf_x,psf_y,s=50,marker="x",color="blue",linewidths=1,zorder=100)
        self._fig.canvas.draw_idle()

        # Determine shots
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
