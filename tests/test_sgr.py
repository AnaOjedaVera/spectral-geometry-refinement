import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spectral_geometry_refinement_cpu import SGRConfig, SpectralGeometryRefiner


class TestSGR(unittest.TestCase):
    def setUp(self):
        self.wave = np.linspace(0.45, 2.45, 31)
        self.X = 1 + np.random.default_rng(42).normal(0, .001, (12, 31))
        self.labels = np.repeat([0, 1], 6)

    def test_source_trend_configuration(self):
        cfg = SGRConfig()
        self.assertEqual(cfg.trend_window_channels, ())
        self.assertEqual(cfg.trend_regions, ((0.45, 0.55), (0.55, 2.50), (0.45, 2.50)))

    def test_parent_size_stop_and_conservation(self):
        result = SpectralGeometryRefiner(SGRConfig(min_parent_n=80)).fit(self.X, self.wave, self.labels)
        self.assertEqual(result.n_global_clusters, 2)
        self.assertEqual(result.n_refined_clusters, 2)
        self.assertEqual(result.n_splits, 0)
        self.assertEqual(len(result.refined_labels), len(self.X))
        self.assertTrue(np.all(result.refined_labels >= 0))

    def test_reject_mismatched_wavelengths(self):
        with self.assertRaises(ValueError):
            SpectralGeometryRefiner().fit(self.X, self.wave[:-1], self.labels)


if __name__ == '__main__':
    unittest.main()
