import unittest
import numpy as np
from agenttrace.views import features


class ViewContract(unittest.TestCase):
    def test_ordinals_denominators_and_missingness(self):
        packets = [(float(i), 100, 1, 1) for i in range(32)]
        req = [
            (
                float(i),
                1,
                "GET",
                1,
                0,
                ("host", "accept") if i % 2 == 0 else ("host",),
                "other",
            )
            for i in range(32)
        ]
        v = features(packets, req, [])
        for k, shape in {
            "conv": (145,),
            "style": (24,),
            "dist": (2190,),
            "bins": (16, 103),
        }.items():
            self.assertEqual(v[k].shape, shape)
        np.testing.assert_array_equal(v["dist"][:1648], v["bins"].ravel())
        np.testing.assert_allclose(
            v["bins"][:, 12], 0.5
        )  # host before accept: divide by BOTH requests
        np.testing.assert_allclose(v["bins"][:, 101], 1.0)
        self.assertTrue(np.isnan(v["conv"][92:142]).all())  # no response transitions
        sparse = features(packets, req[:1], [])
        self.assertTrue(np.isnan(sparse["bins"][1:]).all())
        self.assertTrue(np.isnan(sparse["bins"][0, 101]))
        self.assertTrue(np.isnan(features(packets, [], [])["dist"]).all())
        self.assertTrue(np.isnan(features([], [], [])["conv"]).all())
