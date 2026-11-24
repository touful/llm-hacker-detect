"""Synthetic checks for the model API; no capture or label file is read."""

import unittest

import numpy as np

from agenttrace.models import DEFAULT_NAMES, NAMES, create
from agenttrace.classical import SUITE


class ModelTests(unittest.TestCase):
    def setUp(self):
        n = 66  # includes a short final minibatch
        self.y = np.arange(n) % 2
        rng = np.random.default_rng(781)
        self.X = {
            k: np.zeros((n, w), dtype="f")
            for k, w in [("dist", 2190), ("conv", 145), ("style", 24)]
        }
        self.X["bins"] = np.zeros((n, 16, 103), dtype="f")
        self.X["bins"][:, ::4, :] = rng.uniform(size=(n, 4, 103))
        for k in ("conv", "dist", "style"):
            self.X[k][:, 0] = 2 * self.y - 1

        self.X["dist"][:, 1648] = self.y

    def test_registry(self):
        self.assertEqual(
            set(DEFAULT_NAMES),
            {
                "conv-hgb",
                "dist-hgb",
                "dist-et",
                "dist-linear",
                "joint-hgb",
                "behavior-hgb",
                "bag-hgb",
                "view-fusion",
                "header-mil",
                "header-tcn",
                "header-gru",
            },
        )
        self.assertEqual(len(SUITE), 20)
        self.assertEqual(set(NAMES), set(DEFAULT_NAMES) | set(SUITE))
        with self.assertRaises(ValueError):
            create("unknown")
        with self.assertRaises(RuntimeError):
            create("conv-hgb").predict(self.X)
        with self.assertRaises(ValueError):
            create("header-tcn", batch_size=33)

    def test_tabular_positive_probability_and_support(self):
        for name in (n for n in NAMES if not n.startswith("header")):
            with self.subTest(name=name):
                m = create(name).fit(self.X, self.y)
                x = {k: v[[0, 1, 2]].copy() for k, v in self.X.items()}
                x["dist"][2, :] = x["conv"][2, :] = np.nan
                p = m.predict(x)
                self.assertEqual(p.shape, (3,))
                self.assertTrue(0 <= p[0] < 0.5 < p[1] <= 1)
                self.assertTrue(np.isnan(p[2]))
                if name == "joint-hgb":
                    self.assertEqual(m.estimator_.n_features_in_, 145 + 2190)
                self.assertEqual(
                    m.predict({k: v[:0] for k, v in x.items()}).shape, (0,)
                )

    def test_validate_and_refit(self):
        m = create("dist-linear").fit(self.X, self.y)
        x = {"dist": self.X["dist"].copy()}
        x["dist"][self.y == 1, :] = np.nan
        self.assertIs(m.fit(x, self.y), m)
        self.assertTrue(np.isnan(m.predict(self.X)).all())
        with self.assertRaises(ValueError):
            m.fit(self.X, self.y[:-1])
        with self.assertRaises(ValueError):
            m.fit({"dist": x["dist"][:, :-1]}, self.y)

    def test_sequences(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch unavailable")
        if not torch.cuda.is_available():
            self.skipTest("cuda:0 unavailable")
        for name, count in [
            ("header-mil", 38434),
            ("header-tcn", 44577),
            ("header-gru", 40577),
        ]:
            with self.subTest(name=name):
                m = create(name, epochs=1).fit(self.X, self.y)
                net = m.estimator_
                self.assertEqual(next(net.parameters()).device, torch.device("cuda:0"))
                self.assertEqual(
                    sum(p.numel() for p in net.parameters() if p.requires_grad), count
                )
                x = {k: self.X[k][:4].copy() for k in ("bins", "style")}
                x["bins"][1] = np.nan
                x["bins"][2] = 0  # finite zeros are meaningful observations
                x["style"][2] = np.nan
                p = m.predict(x)
                self.assertEqual(p.shape, (4,))
                self.assertTrue(np.isnan(p[1]))
                self.assertTrue(np.isfinite(p[2]))
                self.assertTrue(np.all((p[[0, 3]] >= 0) & (p[[0, 3]] <= 1)))
                np.testing.assert_allclose(
                    p[3:], m.predict({k: v[3:] for k, v in x.items()}), atol=1e-6
                )
                # The shared position channel distinguishes early from late bins.
                np.testing.assert_allclose(
                    net.position.cpu().flatten()[[0, -1]], [-1.0, 1.0]
                )
