import numpy as np

# ---- masking/noise transform ----
def masked_copy_noisy(
        arr,
        vis_cutoff,
        noise_std,
        *,
        noise_phase="pre",            # "pre", "post", "both", "none"
        dist_power=4,                  # positive → gain ∝ 1/d^power
        eps=1e-3,                      # clamp to avoid 0
        clip_max=None,
        mask_colours_with_vis=True,
        zero_all_distances=False,
        zero_colours=None,
        inplace=False,
        no_vis_val=0.0,
        rng=None,
        # --- NEW ---
        norm_mode="analytic",          # "analytic", "minmax", or None
        norm_axis=-1                   # for "minmax": axis to reduce over (usually -1 or last distance axis)