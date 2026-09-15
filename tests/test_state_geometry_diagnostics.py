import importlib.util
from pathlib import Path

import numpy as np
import unittest

spec = importlib.util.spec_from_file_location("diagnostics", Path(__file__).parents[1] / "scripts/build_state_geometry_diagnostics.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class GeometryDiagnosticsTest(unittest.TestCase):
    def test_correspondence_and_metric_units(self):
        reference = np.array([[0., 0., 0.], [1., 2., 3.]])
        moved = reference + [.003, .004, 0.]
        np.testing.assert_allclose(module.paired_distances_mm(moved, reference), [5., 5.])
        np.testing.assert_array_equal(module.paired_distances_mm(reference, reference), [0., 0.])


    def test_invalid_geometry_rejected(self):
        with self.assertRaises(ValueError):
            module.paired_distances_mm(np.zeros((2, 3)), np.zeros((3, 3)))
        with self.assertRaises(ValueError):
            module.paired_distances_mm(np.full((2, 3), np.nan), np.zeros((2, 3)))


if __name__ == "__main__":
    unittest.main()
