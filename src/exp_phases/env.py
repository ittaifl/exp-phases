import numpy as np
from ratinabox.Neurons import FieldOfViewBVCs, FieldOfViewOVCs
from ratinabox.Agent import Agent
from ratinabox.Environment import Environment

# ---- Extracted from your notebook ----
class WallFixedFOVc(FieldOfViewOVCs):
    def __init__(self, Agent, params={}):
        super().__init__(Agent,params)
        self.history["tuning_distances"] = []
        self.history["sigma_distances"] = []
        self.history["sigma_angles"] = []
        self.history["dists"] = []

    def display_vector_cells(self,
                         fig=None,
                         ax=None,
                         t=None,
                         **kwargs):
        """Visualises the current firing rate of these cells relative to the Agent.
        Essentially this plots the "manifold" ontop of the Agent.
        Each cell is plotted as an ellipse where the alpha-value of its facecolor reflects the current firing rate
        (normalised against the approximate maximum firing rate for all cells, but, take this just as a visualisation).
        Each ellipse is an approximation of the receptive field of the cell which is a von Mises distribution in angule and a Gaussian in distance.
        The width of the ellipse in r and theta give 1 sigma of these distributions (for von Mises: kappa ~= 1/sigma^2).

        This assumes the x-axis in the Agent's frame of reference is the heading direction.
        (Or the heading diection is the X-axis for egocentric frame of reference). IN this case the Y-axis is the towards the "left" of the agent.

        Args:
        • fig, ax: the matplotlib fig, ax objects to plot on (if any), otherwise will plot the Environment
        • t (float): time to plot at
        • object_type (int): if self.cell_type=="OVC", which object type to plot

        Returns:
            fig, ax: with the
        """
        if t is None:
            t = self.Agent.history["t"][-1]
        t_id = np.argmin(np.abs(np.array(self.Agent.history["t"]) - t))

        if fig is None and ax is None:
            fig, ax = self.Agent.plot_trajectory(t_start=t - 10, t_end=t, **kwargs)

        pos = self.Agent.history["pos"][t_id]

        y_axis_wrt_agent = np.array([0, 1])
        x_axis_wrt_agent = np.array([1,0])
        head_direction = self.Agent.history["head_direction"][t_id]
        head_direction_angle = 0.0


        if self.reference_frame == "egocentric":
            head_direction = self.Agent.history["head_direction"][t_id]
            # head direction angle (CCW from true North)
            head_direction_angle = (180 / np.pi) * ratinabox.utils.get_angle(head_direction)

            # this assumes the "x" dimension is the agents head direction and "y" is to its left
            x_axis_wrt_agent = head_direction / np.linalg.norm(head_direction)
            y_axis_wrt_agent = utils.rotate(x_axis_wrt_agent, np.pi / 2)


        fr = np.array(self.history["firingrate"][t_id])

        tuning_distances = np.array(self.history["tuning_distances"][t_id])
        sigma_angles = np.array(self.history["sigma_angles"][t_id])
        sigma_distances = np.array(self.history["sigma_distances"][t_id])
        tuning_angles = self.tuning_angles #this is unchanged


        x = tuning_distances * np.cos(tuning_angles)
        y = tuning_distances * np.sin(tuning_angles)

        pos_of_cells = pos + np.outer(x, x_axis_wrt_agent) + np.outer(y, y_axis_wrt_agent)

        ww = sigma_angles * tuning_distances
        hh = sigma_distances
        aa  = 1.0 * head_direction_angle + tuning_angles * 180 / np.pi

        ec = EllipseCollection(ww,hh, aa, units = 'x',
                                offsets = pos_of_cells,
                                offset_transform = ax.transData,
                                linewidth=0.5,
                                edgecolor="dimgrey",
                                zorder = 2.1,
                                )
        if self.cell_colors is None:
            facecolor = self.color if self.color is not None else "C1"
            facecolor = np.array(matplotlib.colors.to_rgba(facecolor))
            facecolor_array = np.tile(np.array(facecolor), (self.n, 1))
        else:
            facecolor_array = self.cell_colors.copy() #made in child class init. Each cell can have a different plot color.
            # e.g. if cells are slective to different object types or however you like
        facecolor_array[:, -1] = 0.7*np.maximum(
            0, np.minimum(1, fr / (0.5 * self.max_fr))
        ) # scale alpha so firing rate shows as how "solid" (up to 0.7 so you can _just_ seen whats beneath) to color of this vector cell is.
        ec.set_facecolors(facecolor_array)
        ax.add_collection(ec)

        return fig, ax


    def ray_distances_to_walls(self, agent_pos, head_direction, thetas, ray_length, walls):
        # This function computes the distance the agent is to the wall relative to its 6 beams of light. We provide the agent position, its
        # head direction and the 6 beams angles. We get as an output the wall distance of each beam relative to the agents position and HD.
        n = len(thetas)

        heading = np.arctan2(head_direction[1], head_direction[0])

        #print(heading)
        # 1) Build the end points of each ray: shape (n,2)
        ends = np.stack([agent_pos + ray_length * np.array([np.cos(heading + θ),np.sin(heading + θ)]) for θ in thetas], axis=0)

        # 3) starts is just the agent_pos repeated: shape (n,2)
        starts = np.tile(agent_pos[np.newaxis, :], (n, 1))

        # 4) build segs as (n_rays, 2, 2), with [p0, p1] = [start, end]
        segs = np.zeros((n, 2, 2))
        segs[:, 0, :] = starts  # p_a0 = start
        segs[:, 1, :] = ends    # p_a1 = end

        # 5) get the raw line parameters
        intercepts = utils.vector_intercepts(segs, walls, return_collisions=False)
        la = intercepts[..., 0]
        lb = intercepts[..., 1]

        # 6) mask for real segment–segment hits
        valid = (la > 0) & (la < 1) & (lb > 0) & (lb < 1)

        # 7) pick the nearest hit along each ray (smallest la)
        la_valid = np.where(valid, la, np.inf)  # invalid → ∞
        min_la = la_valid.min(axis=1)           # one per ray

        # 8) convert parameter to distance
        distances = np.minimum(min_la * ray_length, ray_length)
        #print(distances)
        return distances #as 6x1 vector of distances for each of the 6 sensors

    def update_cell_locations(self):
        # we update the cell locations by forcing them to remain on the walls.
        # A cell should never wander across the open field as they were originally
        # indended to by rat in a box by default.
        thetas = self.Agent.Neurons[0].tuning_angles
        n=len(thetas)

        self.dists = self.ray_distances_to_walls(self.Agent.pos, self.Agent.head_direction, thetas, 100.0, np.array(self.Agent.Environment.walls))

        s = self.dists*16.5
        self.tuning_distances = np.ones(n)*0.06*s
        self.sigma_distances = np.ones(n)*0.05
        self.sigma_angles = np.ones(n)*0.83333333/s

    def save_to_history(self):
        super().save_to_history()  # Save standard history
        self.history["tuning_distances"].append(self.tuning_distances.copy())
        self.history["sigma_distances"].append(self.sigma_distances.copy())
        self.history["sigma_angles"].append(self.sigma_angles.copy())
        self.history["dists"].append(self.dists.copy()/np.linalg.norm([x_ratio, y_ratio])) #we divide by sqrt(2) because the model wants numbers between 0 and 1 and the max distance in a 1x1 square is sqrt(2)

    def reset_history(self):
        super().reset_history()
        self.history["dists"]=[]


def rescale_env_with_locations(old_env,
                               t0_locations, t1_locations, t2_locations,
                               x_ratio=1.0, y_ratio=1.0):
    """
    Stretch `old_env` by (x_ratio, y_ratio) and re‑insert the three
    colour‑specific location arrays at rescaled coordinates.

    Parameters
    ----------
    old_env : ratinabox.Environment
        The source rectangular arena.
    t0_locations, t1_locations, t2_locations : (N,2) NumPy arrays
        The (x,y) coordinates of red, green, and purple pucks *from the
        old arena*.
    x_ratio, y_ratio : float
        How much to multiply the width and height.

    Returns
    -------
    new_env  : ratinabox.Environment
        A brand‑new Environment whose boundary and objects are resized.
    new_t0, new_t1, new_t2 : NumPy arrays
        The three location arrays after rescaling (useful for plotting).
    """

    # -- 1 · recover old width & height ----------------------------------
    H_old = old_env.params["scale"]                 # original “height”
    W_old = H_old * old_env.params["aspect"]        # original “width”

    # -- 2 · make the stretched boundary --------------------------------
    H_new, W_new = H_old * y_ratio, W_old * x_ratio
    new_env = Environment({"boundary": [[0, 0], [W_new, 0],
                                        [W_new, H_new], [0, H_new]]})

    # -- 3 · rescale the three colour‑arrays -----------------------------
    scale_vec = np.array([x_ratio, y_ratio])
    new_t0 = t0_locations * scale_vec
    new_t1 = t1_locations * scale_vec
    new_t2 = t2_locations * scale_vec

    # -- 4 · add objects to the new environment --------------------------
    for p in new_t0:
        new_env.add_object(p, type=0)
    for p in new_t1:
        new_env.add_object(p, type=1)
    for p in new_t2:
        new_env.add_object(p, type=2)

    return new_env, new_t0, new_t1, new_t2


def get_next_move(
    pos, hd_angle, Env, *,
    # --- step length ---
    step_len_fixed=None,                 # e.g. 0.04 → fixed; None → sample
    mu_len=0.02, sigma_len=0.05,         # used only if fixed is None (trunc. Normal)
    # --- angle sampling ---
    angle_kappa=6.0,                     # von Mises concentration (higher = straighter)
    angle_dist='vonmises',               # 'vonmises' | 'uniform' | 'keep'
    # --- logging / numerics ---
    encode_relative_angles=True,         # True → log Δangle (theta - hd_angle)
    normalize_dist=True,                 # True → distance_used / max(box_w, box_h)
    eps=1e-9,                            # geometric epsilon
    detach_frac=1e-3,                    # post-hit nudge into interior (fraction of box)
    corner_tol_frac=1e-9,                # tx≈ty tolerance for "corner" hit
    jitter_after_hit_deg=0.0,            # tiny random deflection after a hit (0 = none)