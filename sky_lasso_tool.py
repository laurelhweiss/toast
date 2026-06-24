from matplotlib.widgets import LassoSelector
from matplotlib.path import Path
import numpy as np

class PixelLassoSelector:
    """
    Fast(er) lasso selector that works in pixel space.

    Selected points are highlighted semi-transparently in blue; the lasso
    path itself is drawn in black.

    Simple enough to rewrite ra/dec arguments for any 2D coords.
    """
    def __init__(self, fig, ax, data_coords, ra, dec,
                 marker_size_pixels=1.5, alpha_orig=0.05, vertex_stride=3, wcs=None):
        """
        Parameters
        ----------
        data_coords : array (N, 2)
            Fiber/object positions in axes data coordinates (pixel space).
            
        ra, dec : array (N, 1)
            Corresponding on-sky coordinates, needed for selection output. 
            Made for use on small on-sky areas << 1 deg, so xy plane approx ok. 
            
        marker_size_pixels : float
            Desired selection-highlight marker diameter in data (pixel) units.
        """

        self.fig = fig
        self.ax = ax
        self.data_coords = np.asarray(data_coords)
        self.ra = np.asarray(ra)
        self.dec = np.asarray(dec)
        self.alpha_orig = alpha_orig
        self.marker_size_pixels = marker_size_pixels
        self.vertex_stride = vertex_stride
        self.canvas = self.fig.canvas

        self.ind = np.array([], dtype=int)
        self.selected_ra = np.array([])
        self.selected_dec = np.array([])

        # ----- Compute scatter size in points^2 from data (pixel) units -----
        self.canvas.draw()  # ensure bbox is valid
        bbox = ax.get_window_extent()
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        data_per_pt_x = (xlim[1] - xlim[0]) / (bbox.width  * 72 / self.fig.dpi)
        data_per_pt_y = (ylim[1] - ylim[0]) / (bbox.height * 72 / self.fig.dpi)
        data_per_pt = np.mean([abs(data_per_pt_x), abs(data_per_pt_y)])
        width_pts = self.marker_size_pixels / data_per_pt
        self.s = width_pts**2  # matplotlib scatter uses points^2

        # -------------- Show selection coverage --------------
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

        # ----------- Bounding box prefilter (for speed) ------------
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

        # -------- Store selection indices, convert to RA/Dec --------
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
