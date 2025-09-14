import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import RMSprop
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.nn.init as init
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.collections import EllipseCollection
from sklearn.decomposition import PCA
from tqdm import tqdm
from ratinabox.Environment import Environment
from ratinabox.Agent import Agent
from ratinabox.Neurons import FieldOfViewBVCs, FieldOfViewOVCs
from ratinabox import utils
import ratinabox
import warnings
from ratinabox.Environment import Environment
import numpy as np
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
from matplotlib.colors import LogNorm
from matplotlib.cm import get_cmap
import torch

class WallFixedFOVc(FieldOfViewOVCs):

    def __init__(self, Agent, params={}, x_ratio: float = 1.0, y_ratio: float = 1.0):
        super().__init__(Agent, params)
        self.x_ratio = x_ratio
        self.y_ratio = y_ratio
        self.history['tuning_distances'] = []
        self.history['sigma_distances'] = []
        self.history['sigma_angles'] = []
        self.history['dists'] = []

    def display_vector_cells(self, fig=None, ax=None, t=None, **kwargs):
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
            t = self.Agent.history['t'][-1]
        t_id = np.argmin(np.abs(np.array(self.Agent.history['t']) - t))
        if fig is None and ax is None:
            fig, ax = self.Agent.plot_trajectory(t_start=t - 10, t_end=t, **kwargs)
        pos = self.Agent.history['pos'][t_id]
        y_axis_wrt_agent = np.array([0, 1])
        x_axis_wrt_agent = np.array([1, 0])
        head_direction = self.Agent.history['head_direction'][t_id]
        head_direction_angle = 0.0
        if self.reference_frame == 'egocentric':
            head_direction = self.Agent.history['head_direction'][t_id]
            head_direction_angle = 180 / np.pi * ratinabox.utils.get_angle(head_direction)
            x_axis_wrt_agent = head_direction / np.linalg.norm(head_direction)
            y_axis_wrt_agent = utils.rotate(x_axis_wrt_agent, np.pi / 2)
        fr = np.array(self.history['firingrate'][t_id])
        tuning_distances = np.array(self.history['tuning_distances'][t_id])
        sigma_angles = np.array(self.history['sigma_angles'][t_id])
        sigma_distances = np.array(self.history['sigma_distances'][t_id])
        tuning_angles = self.tuning_angles
        x = tuning_distances * np.cos(tuning_angles)
        y = tuning_distances * np.sin(tuning_angles)
        pos_of_cells = pos + np.outer(x, x_axis_wrt_agent) + np.outer(y, y_axis_wrt_agent)
        ww = sigma_angles * tuning_distances
        hh = sigma_distances
        aa = 1.0 * head_direction_angle + tuning_angles * 180 / np.pi
        ec = EllipseCollection(ww, hh, aa, units='x', offsets=pos_of_cells, offset_transform=ax.transData, linewidth=0.5, edgecolor='dimgrey', zorder=2.1)
        if self.cell_colors is None:
            facecolor = self.color if self.color is not None else 'C1'
            facecolor = np.array(matplotlib.colors.to_rgba(facecolor))
            facecolor_array = np.tile(np.array(facecolor), (self.n, 1))
        else:
            facecolor_array = self.cell_colors.copy()
        facecolor_array[:, -1] = 0.7 * np.maximum(0, np.minimum(1, fr / (0.5 * self.max_fr)))
        ec.set_facecolors(facecolor_array)
        ax.add_collection(ec)
        return (fig, ax)

    def ray_distances_to_walls(self, agent_pos, head_direction, thetas, ray_length, walls):
        n = len(thetas)
        heading = np.arctan2(head_direction[1], head_direction[0])
        ends = np.stack([agent_pos + ray_length * np.array([np.cos(heading + θ), np.sin(heading + θ)]) for θ in thetas], axis=0)
        starts = np.tile(agent_pos[np.newaxis, :], (n, 1))
        segs = np.zeros((n, 2, 2))
        segs[:, 0, :] = starts
        segs[:, 1, :] = ends
        intercepts = utils.vector_intercepts(segs, walls, return_collisions=False)
        la = intercepts[..., 0]
        lb = intercepts[..., 1]
        valid = (la > 0) & (la < 1) & (lb > 0) & (lb < 1)
        la_valid = np.where(valid, la, np.inf)
        min_la = la_valid.min(axis=1)
        distances = np.minimum(min_la * ray_length, ray_length)
        return distances

    def update_cell_locations(self):
        thetas = self.Agent.Neurons[0].tuning_angles
        n = len(thetas)
        self.dists = self.ray_distances_to_walls(self.Agent.pos, self.Agent.head_direction, thetas, 100.0, np.array(self.Agent.Environment.walls))
        s = self.dists * 16.5
        self.tuning_distances = np.ones(n) * 0.06 * s
        self.sigma_distances = np.ones(n) * 0.05
        self.sigma_angles = np.ones(n) * 0.83333333 / s

    def save_to_history(self):
        super().save_to_history()
        self.history['tuning_distances'].append(self.tuning_distances.copy())
        self.history['sigma_distances'].append(self.sigma_distances.copy())
        self.history['sigma_angles'].append(self.sigma_angles.copy())
        self.history['dists'].append(
            self.dists.copy() / np.linalg.norm([self.x_ratio, self.y_ratio])
        )

    def reset_history(self):
        super().reset_history()
        self.history['dists'] = []

def get_next_move(pos, hd_angle, Env, *, step_len_fixed=None, mu_len=0.02, sigma_len=0.05, angle_kappa=6.0, angle_dist='vonmises', encode_relative_angles=True, normalize_dist=True, eps=1e-09, detach_frac=0.001, corner_tol_frac=1e-09, jitter_after_hit_deg=0.0):
    """
    Continuous stepper:
      • sample an executed angle (theta),
      • attempt a step; clip to first wall (no in-frame bounce),
      • log [angle, distance_used] where angle is RELATIVE if encode_relative_angles=True,
      • next heading = 180° turn if clipped, with anti-sticking nudge/projection.
    Returns:
      next_pos : (2,) float
      next_angle : float radians in [-π,π)
      action_vec : (2,) float = [ angle_to_log , distance_used_(norm or raw) ]
    """
    import numpy as np
    B = np.asarray(Env.boundary) if hasattr(Env, 'boundary') else np.asarray(Env.params['boundary'])
    xmin, ymin = (B[:, 0].min(), B[:, 1].min())
    xmax, ymax = (B[:, 0].max(), B[:, 1].max())
    W, H = (xmax - xmin, ymax - ymin)
    box_side = max(W, H)
    corner_tol = corner_tol_frac * box_side
    detach_eps = detach_frac * box_side

    def _wrap(a):
        return (a + np.pi) % (2 * np.pi) - np.pi

    def _ensure_interior_angle(theta, p, margin=5 * eps):
        """If theta points outward at a boundary contact, flip offending component(s)."""
        vx, vy = (np.cos(theta), np.sin(theta))
        if p[0] <= xmin + margin and vx < 0:
            vx = abs(vx)
        if p[0] >= xmax - margin and vx > 0:
            vx = -abs(vx)
        if p[1] <= ymin + margin and vy < 0:
            vy = abs(vy)
        if p[1] >= ymax - margin and vy > 0:
            vy = -abs(vy)
        return np.arctan2(vy, vx)
    if angle_dist == 'vonmises':
        theta = _wrap(hd_angle + np.random.vonmises(0.0, angle_kappa))
    elif angle_dist == 'uniform':
        theta = np.random.uniform(-np.pi, np.pi)
    else:
        theta = _wrap(hd_angle)
    v = np.array([np.cos(theta), np.sin(theta)], float)
    if step_len_fixed is not None:
        step_req = float(step_len_fixed)
    else:
        step_req = np.random.normal(mu_len, sigma_len)
        while step_req <= 0:
            step_req = np.random.normal(mu_len, sigma_len)
    tx = np.inf
    if v[0] > 0:
        tx = (xmax - pos[0]) / v[0]
    elif v[0] < 0:
        tx = (xmin - pos[0]) / v[0]
    ty = np.inf
    if v[1] > 0:
        ty = (ymax - pos[1]) / v[1]
    elif v[1] < 0:
        ty = (ymin - pos[1]) / v[1]
    collided = step_req >= min(tx, ty) - eps
    if collided:
        if abs(tx - ty) <= corner_tol:
            t_hit = min(tx, ty)
            hit_x = hit_y = True
        elif tx < ty:
            t_hit = tx
            hit_x, hit_y = (True, False)
        else:
            t_hit = ty
            hit_x, hit_y = (False, True)
    else:
        t_hit = np.inf
        hit_x = hit_y = False
    dist_exec = min(step_req, max(0.0, t_hit - eps))
    next_pos = pos + dist_exec * v
    next_pos[0] = np.clip(next_pos[0], xmin + eps, xmax - eps)
    next_pos[1] = np.clip(next_pos[1], ymin + eps, ymax - eps)
    if collided:
        next_angle = _ensure_interior_angle(_wrap(theta + np.pi), next_pos)
        if jitter_after_hit_deg > 0:
            jitter = np.deg2rad(jitter_after_hit_deg) * (2 * np.random.rand() - 1.0)
            next_angle = _ensure_interior_angle(_wrap(next_angle + jitter), next_pos)
        u = np.array([np.cos(next_angle), np.sin(next_angle)], float)
        next_pos = next_pos + detach_eps * u
        next_pos[0] = np.clip(next_pos[0], xmin + eps, xmax - eps)
        next_pos[1] = np.clip(next_pos[1], ymin + eps, ymax - eps)
    else:
        next_angle = theta
    logged_angle = _wrap(theta - hd_angle) if encode_relative_angles else theta
    logged_dist = dist_exec / box_side if normalize_dist else dist_exec
    action_vec = np.array([logged_angle, logged_dist], float)
    return (next_pos, next_angle, action_vec)

def rescale_env_with_locations(old_env, t0_locations, t1_locations, t2_locations, x_ratio=1.0, y_ratio=1.0):
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
    H_old = old_env.params['scale']
    W_old = H_old * old_env.params['aspect']
    H_new, W_new = (H_old * y_ratio, W_old * x_ratio)
    new_env = Environment({'boundary': [[0, 0], [W_new, 0], [W_new, H_new], [0, H_new]]})
    scale_vec = np.array([x_ratio, y_ratio])
    new_t0 = t0_locations * scale_vec
    new_t1 = t1_locations * scale_vec
    new_t2 = t2_locations * scale_vec
    for p in new_t0:
        new_env.add_object(p, type=0)
    for p in new_t1:
        new_env.add_object(p, type=1)
    for p in new_t2:
        new_env.add_object(p, type=2)
    return (new_env, new_t0, new_t1, new_t2)

def masked_copy_noisy(arr, vis_cutoff, noise_std, *, noise_phase='pre', dist_power=4, eps=0.001, clip_max=None, mask_colours_with_vis=True, zero_all_distances=False, zero_colours=None, inplace=False, no_vis_val=0.0, rng=None, norm_mode='analytic', norm_axis=-1):
    """
    arr shape: (..., 24) = 6 sensors × (3 colours + 1 distance)
       [R0 G0 B0  R1 G1 B1  ... R5 G5 B5 | D0 D1 D2 D3 D4 D5]
    """
    if arr.shape[-1] != 24:
        raise ValueError('Expected last dimension = 24')
    gen = rng if rng is not None else np.random
    target = arr if inplace else arr.copy()
    colours = target[..., :18].reshape(*target.shape[:-1], 6, 3)
    dists = target[..., -6:]
    far = dists > vis_cutoff
    dists[far] = no_vis_val
    if mask_colours_with_vis:
        colours[far] = no_vis_val
    if zero_all_distances:
        dists[...] = no_vis_val
    if zero_colours:
        if isinstance(zero_colours, str):
            zero_colours = [zero_colours]
        idx_map = {'r': 0, 'g': 1, 'b': 2}
        zero_colours = [c.lower() for c in zero_colours]
        if 'all' in zero_colours:
            colours[...] = no_vis_val
        else:
            for c in zero_colours:
                colours[..., idx_map[c]] = no_vis_val
    if noise_std > 0 and noise_phase in ('pre', 'both'):
        dists += gen.normal(0.0, noise_std, size=dists.shape)
    d_safe = np.maximum(dists, eps)
    gain_raw = ((vis_cutoff + eps) / (d_safe + eps)) ** dist_power
    if clip_max is not None:
        gain_raw = np.minimum(gain_raw, clip_max)
    if norm_mode == 'analytic':
        g_far = 1.0
        g_near = ((vis_cutoff + eps) / (eps + eps)) ** dist_power
        denom = max(g_near - g_far, 1e-12)
        gain = (gain_raw - g_far) / denom
        gain = np.clip(gain, 0.0, 1.0)
    elif norm_mode == 'minmax':
        g_min = gain_raw.min(axis=norm_axis, keepdims=True)
        g_max = gain_raw.max(axis=norm_axis, keepdims=True)
        denom = np.maximum(g_max - g_min, 1e-12)
        gain = (gain_raw - g_min) / denom
    else:
        gain = gain_raw
    dists[...] = gain
    if noise_std > 0 and noise_phase in ('post', 'both'):
        target += gen.normal(0.0, noise_std, size=target.shape)
    return target

class NormReLU(nn.Module):

    def __init__(self, hidden_size, epsilon=0.01, noise_std=0.03):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.epsilon = epsilon
        self.noise_std = noise_std

    def forward(self, x):
        x_norm = (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + self.epsilon)
        noise = torch.randn_like(x) * self.noise_std
        return F.relu(x_norm + self.bias + noise)

class HardSigmoid(nn.Module):

    def __init__(self):
        super(HardSigmoid, self).__init__()
        self.g = torch.nn.ReLU()

    def forward(self, x):
        return torch.clamp(0.2 * x + 0.5, min=0.0, max=1.0)

class NextStepRNN(nn.Module):

    def __init__(self, obs_dim=144, act_dim=2, hidden_dim=500):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim
        self.W_in = nn.Linear(obs_dim, hidden_dim, bias=False)
        self.W_act = nn.Linear(act_dim, hidden_dim, bias=False)
        self.W_rec = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_out = nn.Linear(hidden_dim, obs_dim)
        self.beta = nn.Parameter(torch.zeros(1))
        self.norm_relu = NormReLU(hidden_dim)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.15)
        self.hardsigmoid = HardSigmoid()
        self.g = self.norm_relu
        self.init_weights()

    def init_weights(self):
        tau = 2.0
        k_in = 1.0 / np.sqrt(self.obs_dim + self.act_dim)
        k_out = 1.0 / np.sqrt(self.hidden_dim)
        init.uniform_(self.W_in.weight, -k_in, k_in)
        init.uniform_(self.W_act.weight, -k_in, k_in)
        init.uniform_(self.W_out.weight, -k_out, k_out)
        W_rec_data = torch.empty(self.hidden_dim, self.hidden_dim)
        init.uniform_(W_rec_data, -k_out, k_out)
        identity_boost = torch.eye(self.hidden_dim) * (1 - 1 / tau)
        W_rec_data += identity_boost
        self.W_rec.weight.data = W_rec_data
    '\n    #Init weights like in Recanatesi et al.\n    def init_weights(self):\n        # Initialize W_rec to identity matrix\n        nn.init.eye_(self.W_rec.weight)\n\n        # Initialize W_in, W_act, and W_out to normal distribution (mean=0, std=0.02)\n        nn.init.normal_(self.W_in.weight, mean=0.0, std=0.02)\n        nn.init.normal_(self.W_act.weight, mean=0.0, std=0.02)\n        nn.init.normal_(self.W_out.weight, mean=0.0, std=0.02)\n    '

    def forward(self, obs_seq, act_seq, return_hidden=False):
        T, B, _ = obs_seq.size()
        h = torch.zeros(B, self.hidden_dim, device=obs_seq.device)
        y = torch.zeros(B, self.obs_dim, device=obs_seq.device)
        outputs, hiddens = ([], [])
        for t in range(T):
            o_in = self.W_in(obs_seq[t, :, :])
            a_in = self.W_act(act_seq[t, :, :])
            h_in = self.W_rec(h)
            bias = self.beta
            g = self.g
            h = g(o_in + a_in + h_in + bias)
            if return_hidden:
                hiddens.append(h.detach().cpu())
            y = torch.sigmoid(self.W_out(h))
            outputs.append(y)
        outputs = torch.stack(outputs)
        if return_hidden:
            hiddens = torch.stack(hiddens)
            return (outputs, hiddens)
        return outputs