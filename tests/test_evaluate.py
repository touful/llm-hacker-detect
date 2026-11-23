"""Offline evaluation protocol tests; no captures, GPU or network required."""

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from agenttrace.evaluate import _cutoffs, _metrics, evaluate


class EvaluationTests(unittest.TestCase):
    def test_validation_budget_and_ties(self):
        neg = np.r_[0.99, 0.90, np.full(4, 0.85), np.full(94, 0.1), np.nan]
        scores = np.r_[neg, 1.0]
        y = np.r_[np.zeros(len(neg)), 1]
        cuts = _cutoffs(y, scores)
        self.assertEqual(cuts, {1: 0.90, 5: 0.85})
        self.assertEqual([(neg > cut).sum() for cut in cuts.values()], [1, 2])
        self.assertEqual(_cutoffs(y[-1:], scores[-1:]), {1: None, 5: None})

    def test_probability_and_classification_metrics(self):
        y = np.array([0, 0, 1, 1, 1])
        m = _metrics(y, np.array([0.1, 0.6, 0.7, 0.5, np.nan]))
        self.assertEqual(m["confusion"], {"tn": 1, "fp": 1, "fn": 0, "tp": 2})
        expected = dict(
            ap=5 / 6,
            auc=0.75,
            accuracy=0.75,
            balanced_accuracy=0.75,
            mcc=1 / np.sqrt(3),
            f1=0.8,
            precision=2 / 3,
            recall=1.0,
            specificity=0.5,
            brier=(0.01 + 0.36 + 0.09 + 0.25) / 4,
            log_loss=-np.log([0.9, 0.4, 0.7, 0.5]).mean(),
        )
        for key, value in expected.items():
            with self.subTest(metric=key):
                self.assertAlmostEqual(m[key], value)
        self.assertEqual(m["support"]["by_class"]["1"]["coverage"], 2 / 3)
        empty = _metrics(y, np.full(5, np.nan))
        self.assertEqual(empty["support"]["coverage"], 0)
        self.assertIsNone(empty["accuracy"])
        self.assertEqual(sum(empty["confusion"].values()), 0)

    def test_outer_protocol_oof_common_support_and_serialization(self):
        # Interleaved groups expose accidental concatenation in fold order.
        folds = np.tile([20, 10, 50, 30, 40], 4)
        ids = np.arange(20)
        y = ids // 5 % 2
        files = [f"synthetic,{i}" for i in ids]
        scores = np.where(y == 1, 0.8, folds / 100.0)
        records = []

        class Spy:
            def __init__(self, name, seed, **kwargs):
                self.name = name
                self.calls = []
                records.append((seed, self.calls))

            def fit(self, X, labels):
                np.testing.assert_array_equal(labels, y[X["id"]])
                self.calls.append(X["id"])

            def predict(self, X):
                ix = X["id"]
                self.calls.append(ix)
                p = scores[ix].copy()
                if self.name == "partial":
                    p[(folds[ix] == 20) & (y[ix] == 0)] = np.nan
                return p

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("agenttrace.models.create", Spy),
        ):
            out = Path(tmp)
            summary = evaluate(y, folds, files, {"id": ids}, ["full", "partial"], out)
            self.assertEqual(len(records), 10)
            for k, (seed, (train, val, test)) in enumerate(records):
                expected_test = 10 * (1 + k % 5)
                expected_val = 10 * (1 + (k + 1) % 5)
                self.assertEqual(seed, 42 + k % 5)
                np.testing.assert_array_equal(
                    train, ids[~np.isin(folds, [expected_test, expected_val])]
                )
                np.testing.assert_array_equal(val, ids[folds == expected_val])
                np.testing.assert_array_equal(test, ids[folds == expected_test])
            self.assertEqual(
                summary, json.loads((out / "summary.json").read_text("utf-8"))
            )
            full, partial = (summary["models"][n] for n in ("full", "partial"))
            self.assertEqual(
                partial,
                json.loads((out / "partial" / "summary.json").read_text("utf-8")),
            )
            self.assertEqual(full["metrics"]["support"]["scored"], 20)
            self.assertEqual(
                partial["metrics"]["support"]["by_class"]["0"]["scored"], 8
            )
            self.assertEqual(
                full["common_support_metrics"], partial["common_support_metrics"]
            )
            op = full["folds"][-1]["low_fpr"]["1"]
            self.assertEqual(op["threshold_exclusive"], 0.1)  # test negatives are .5
            self.assertEqual((op["validation_fpr"], op["fpr"]), (0, 1))
            self.assertEqual(
                partial["folds"][0]["low_fpr"]["1"]["support"]["scored"], 0
            )
            with (out / "partial" / "oof.csv").open(encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual([r["file"] for r in rows], files)
            self.assertEqual((rows[0]["score"], rows[0]["common_support"]), ("", "0"))
            self.assertEqual((rows[1]["score"], rows[1]["pred_fpr_1pct"]), ("0.1", ""))

    def test_real_model_offline(self):
        y = np.tile([0, 1], 15)
        folds = np.repeat(np.arange(5), 6)
        X = np.zeros((len(y), 2190), dtype=np.float32)
        X[:, 0] = y
        X[0, :] = np.nan
        with tempfile.TemporaryDirectory() as tmp:
            r = evaluate(y, folds, range(len(y)), {"dist": X}, ["dist-linear"], tmp)
            m = r["models"]["dist-linear"]["metrics"]
            self.assertEqual(m["support"]["scored"], len(y) - 1)
            self.assertEqual(m["auc"], 1.0)
            with self.assertRaisesRegex(ValueError, "five"):
                evaluate(y, folds % 4, range(len(y)), {"dist": X}, ["dist-linear"], tmp)
