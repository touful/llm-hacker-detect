"""Small, synthetic checks; no capture data or trained model is required."""

import unittest

import numpy as np

from agenttrace.features import (
    HEADER_CATEGORIES,
    HEADER_ORDER_PAIRS,
    NAMED_HEADERS,
    _header_observations,
    _ordinal_bins,
)


class TestRequestObservations(unittest.TestCase):
    def test_rows_only_for_real_requests(self):
        for n, expected in (
            (0, []),
            (1, [0]),
            (2, [0, 8]),
            (5, [0, 3, 6, 9, 12]),
            (17, [0, *range(16)]),
        ):
            np.testing.assert_array_equal(_ordinal_bins(n), expected)

    def test_headers_do_not_imply_missing_or_repeated_pairs(self):
        present, ordered = _header_observations(
            ("x-foo", "accept", "host", "x-bar", "accept", "cookie")
        )
        self.assertEqual(
            tuple(HEADER_CATEGORIES[i] for i in present),
            ("other", "accept", "host", "cookie"),
        )
        self.assertEqual(
            tuple(HEADER_ORDER_PAIRS[i] for i in ordered),
            (("accept", "host"), ("accept", "cookie"), ("host", "cookie")),
        )
        self.assertEqual(_header_observations(()), ((), ()))

    def test_all_order_columns_are_reachable(self):
        observed = []
        for headers in (NAMED_HEADERS, NAMED_HEADERS[::-1]):
            _, indices = _header_observations(headers)
            observed.extend(indices)
        self.assertEqual(sorted(observed), list(range(90)))
