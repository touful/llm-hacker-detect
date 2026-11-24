"""Fixed, label-independent projections of the passive-http-v1 Dist schema."""

import numpy as np
from .features import HEADER_CATEGORIES, HEADER_ORDER_PAIRS

BASE = (
    list(HEADER_CATEGORIES)
    + [f"{a}_before_{b}" for a, b in HEADER_ORDER_PAIRS]
    + ["switch", "diversity"]
)
STYLE = [
    f"phase{j}_{v}"
    for j in range(3)
    for v in ("ua", "accept", "language", "encoding", "cookie", "referer", "host_first")
] + ["drift01", "drift12", "drift02"]
COLUMNS = (
    [
        f"{block}.{name}"
        for block in [*[f"bin{j}" for j in range(16)], "min", "max", "mean", "sd"]
        for name in BASE
    ]
    + [f"ever.{a}_before_{b}" for a, b in HEADER_ORDER_PAIRS]
    + [f"deviation{j}" for j in range(16)]
    + STYLE
)
ALL = np.arange(len(COLUMNS))
BLOCKS = {
    "raw": ALL[:1648],
    "stats": ALL[1648:2060],
    "flags": ALL[2060:2150],
    "dev": ALL[2150:2166],
    "style": ALL[2166:],
}
MEAN = ALL[1854:1957]
VIEWS = {
    **BLOCKS,
    **{f"no-{k}": np.setdiff1d(ALL, v) for k, v in BLOCKS.items()},
    "fieldmean": MEAN[:11],
    "mean": MEAN,
    "mean-flags": np.r_[MEAN, BLOCKS["flags"]],
    "mean-flags-style": np.r_[MEAN, BLOCKS["flags"], BLOCKS["style"]],
    "bag": ALL[1648:2150],
    "no-explicit-pairs": np.r_[
        ALL[:2060][~np.isin(ALL[:2060] % 103, range(11, 101))], ALL[2150:]
    ],
    "field-only": ALL[:2060][ALL[:2060] % 103 < 11],
}
CANDIDATES = ("fieldmean", "flags", "mean", "mean-flags", "mean-flags-style", "bag")


def choose(ap, tolerance=0.015):
    eligible = [k for k in CANDIDATES if ap[k] >= ap["full"] - tolerance]
    return min(eligible, key=lambda k: len(VIEWS[k])) if eligible else "full"
