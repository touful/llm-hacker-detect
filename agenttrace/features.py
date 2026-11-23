"""Session feature extraction primitives; see docs/features.md for conventions."""

import numpy as np


# The fixed vocabulary is independent of the dataset and Python hash seed.
NAMED_HEADERS = (
    "host",
    "user-agent",
    "accept",
    "accept-encoding",
    "accept-language",
    "connection",
    "content-type",
    "content-length",
    "cookie",
    "referer",
)
HEADER_CATEGORIES = NAMED_HEADERS + ("other",)
_HEADER_INDEX = {name: i for i, name in enumerate(NAMED_HEADERS)}
HEADER_ORDER_PAIRS = tuple(
    (a, b) for a in NAMED_HEADERS for b in NAMED_HEADERS if a != b
)


def _ordinal_bins(count):
    """Return one destination row per real request, never padding the input."""
    if count == 0:
        return np.empty(0, dtype=np.intp)
    return np.arange(count, dtype=np.intp) * 16 // count


def _header_observations(headers):
    """Return present category indices and first-occurrence directed pair indices."""
    first = tuple(dict.fromkeys(_HEADER_INDEX.get(h, 10) for h in headers))
    named = tuple(i for i in first if i != 10)
    # Row-major off-diagonal entries of a 10x10 matrix (9 columns per row).
    before = tuple(
        a * 9 + b - (b > a) for k, a in enumerate(named) for b in named[k + 1 :]
    )
    return first, before
