from matplotlib.widgets import LassoSelector
from matplotlib.path import Path
import numpy as np

class PixelLassoSelector:
    """
    Fast (er) lasso using pixel coordinates.
    Selected points stay semi-transparent; non-selected points invisible.
    Stores original RA/Dec for table indexing.
    """
    def __init__(self, fig, ax, data_coords, ra, dec,
                 marker_size_arcsec=1.5, alpha_orig=0.05, vertex_stride=3, wcs=None):
        
        self.fig = fig
        self.ax = ax
        self.data_coords = np.asarray(data_coords)
        self.ra = np.asarray(ra)
        self.dec = np.asarray(dec)
        self.alpha_orig = alpha_orig
        self.marker_size_arcsec = marker_size_arcsec
        self.vertex_stride = vertex_stride
        self.canvas = self.fig.canvas

        self.ind = np.array([], dtype=int)
        self.selected_ra = np.array([])
        self.selected_dec = np.array([])

        # ----- Compute scatter size in points^2 from arcsec ------
        # The axes are in arcsec offset space, so we compute how many
        # display points correspond to marker_size_arcsec directly from
        # the axes data-to-display transform.
        self.canvas.draw()  # ensure bbox is valid
        # Get the axes extent in both data (arcsec) and display (points) units
        bbox = ax.get_window_extent()
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        arcsec_per_pt_x = (xlim[1] - xlim[0]) / (bbox.width  * 72 / self.fig.dpi)
        arcsec_per_pt_y = (ylim[1] - ylim[0]) / (bbox.height * 72 / self.fig.dpi)
        arcsec_per_pt = np.mean([abs(arcsec_per_pt_x), abs(arcsec_per_pt_y)])
        width_pts = self.marker_size_arcsec / arcsec_per_pt
        self.s = width_pts**2  # matplotlib scatter uses points^2

        # ---- Show selection fiber coverage -----
        self.collection_sel = ax.scatter([], [], s=self.s, color='blue',
                                         alpha=self.alpha_orig,
                                         edgecolors='none', zorder=20)

        # Disable WCS grid (for speed) — only applies to WCS-projection axes
        try:
            ax.coords.grid(False)
        except Exception:
            pass

        # -------------- The actual lasso tool ----------------
        self.lasso = LassoSelector(ax, onselect=self.onselect,
                                   props=dict(color='black', linewidth=1, alpha=0.8))

    def onselect(self, verts):
        verts = np.asarray(verts)
        if self.vertex_stride > 1:
            verts = verts[::self.vertex_stride]

        path = Path(verts)

        # -------- Bounding box prefilter (for speed) ---------
        xmin, ymin = verts.min(axis=0)
        xmax, ymax = verts.max(axis=0)
        mask = ((self.data_coords[:,0] >= xmin) & (self.data_coords[:,0] <= xmax) &
                (self.data_coords[:,1] >= ymin) & (self.data_coords[:,1] <= ymax))
        candidate_inds = np.nonzero(mask)[0]
        candidate_points = self.data_coords[mask]
        inside = path.contains_points(candidate_points)
        new_ind = candidate_inds[inside]

        # ------ Update selection scatter in pixel coordinates -------
        if len(new_ind) > 0:
            self.collection_sel.set_offsets(self.data_coords[new_ind])
        else:
            self.collection_sel.set_offsets([])

        # ----------- Store selection indices and RA/Dec -------------
        self.ind = new_ind
        self.selected_ra = self.ra[new_ind]
        self.selected_dec = self.dec[new_ind]

        self.canvas.draw_idle()

    def disconnect(self, keep_selection=True):
        self.lasso.disconnect_events()

        if not keep_selection:
            try:
                self.collection_sel.remove()
            except Exception:
                pass

        self.canvas.draw_idle()
