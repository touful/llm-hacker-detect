import unittest
import numpy as np
from agenttrace.ablation import CANDIDATES, COLUMNS, VIEWS, choose
from agenttrace.study import duplicates, rowkeys


class FeatureStudyTests(unittest.TestCase):
    def test_projections(self):
        self.assertEqual(len(set(COLUMNS)), 2190)
        for key, cols in VIEWS.items():
            self.assertEqual(len(cols), len(set(cols)))
            self.assertTrue(np.all((cols >= 0) & (cols < 2190)))
            if key.startswith("no-") and key[3:] in VIEWS:
                self.assertEqual(len(np.union1d(cols, VIEWS[key[3:]])), 2190)
                self.assertEqual(len(np.intersect1d(cols, VIEWS[key[3:]])), 0)
        self.assertEqual(COLUMNS[VIEWS["mean"][0]], "mean.host")
        self.assertFalse(
            any("before" in COLUMNS[i] for i in VIEWS["no-explicit-pairs"])
        )

    def test_lock(self):
        ap = {k: 0.70 for k in CANDIDATES}
        ap["full"] = 0.80
        self.assertEqual(choose(ap), "full")
        ap.update(flags=0.79, mean=0.81)
        self.assertEqual(choose(ap), "flags")

    def test_duplicates(self):
        k = rowkeys(
            np.array([[1.0, np.nan], [1.0, np.nan], [0.5, 0.5], [np.nan, np.nan]])
        )
        self.assertEqual(
            duplicates(k, [0, 1, 1, 1], set()).tolist(), [True, True, False, False]
        )
        self.assertEqual(
            duplicates(k, [1, 1, 1, 1], {k[2]}).tolist(), [False, False, True, False]
        )
