"""Documented source defaults of the thesis 2-GPU asteroid SGR implementation.

This is not an independent record of runtime overrides or exact package versions.
"""
from spectral_geometry_refinement_gpu import SGRConfig


def asteroid_reference_config():
    return SGRConfig(
        seed=42,
        max_refine_depth=2,
        refine_kmax=4,
        min_parent_n=80,
        min_child_n=3,
        min_child_frac=0.0,
        prom_frac=0.035,
        peak_distance=3,
        landmark_min_width_channels=3.0,
        landmark_support_frac=0.50,
        landmark_support_tolerance_channels=8,
        landmark_support_prom_scale=0.75,
        delta_bic_min=10.0,
        ari_view_min=0.30,
        min_coherent_views=2,
        cons_sil_min=0.15,
        cons_ari_min=0.30,
        gmm_n_init=15,
        gmm_max_iter=1500,
        gmm_tol=1e-3,
        gpu_backend="gpu",
        gpu_dtype="float32",
        gpu_devices=(0, 1),
        parallel_root_clusters=True,
        gap_factor=2.5,
        gap_min=0.05,
        use_regional_trends=True,
        trend_window_channels=(),
        trend_step_fraction=1.0,
        trend_min_channels=5,
        trend_include_full_region=False,
        trend_regions=((0.45, 0.55), (0.55, 2.50), (0.45, 2.50)),
    )
