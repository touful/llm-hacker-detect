"""事后配对消融：从同一组有向存在标志构造无向共现对照。"""

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from .ablation import VIEWS
from .evaluate import evaluate
from .features import HEADER_ORDER_PAIRS, NAMED_HEADERS
from .models import _Model
from .study import load, paired, save, scores

_PAIRS = list(combinations(NAMED_HEADERS, 2))
_FORWARD = [HEADER_ORDER_PAIRS.index((a, b)) for a, b in _PAIRS]
_REVERSE = [HEADER_ORDER_PAIRS.index((b, a)) for a, b in _PAIRS]


def unordered(flags):
    # 两方向任一出现即为共现；零值保持可观测，全空会话保持 NaN。
    return np.maximum(flags[:, _FORWARD], flags[:, _REVERSE])


class PairModel(_Model):
    def __init__(self, view, **kwargs):
        super().__init__("dist-hgb", **kwargs)
        self.view = view

    def _inputs(self, X):
        a = X[self.view]
        return (a,), np.isfinite(a).any(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cohort", type=Path)
    parser.add_argument("pilot", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    z, pilot = load(args.cohort), load(args.pilot)
    dev = np.isin(z["files"], pilot["files"])
    if (
        z["meta"]["max_packets"] != 2048
        or z["meta"]["labels_sha256"] != pilot["meta"]["labels_sha256"]
        or not all(np.array_equal(z[k][dev], pilot[k]) for k in ("files", "y", "fold"))
    ):
        raise ValueError("unexpected observation budget or mismatched cohorts")
    ix = np.flatnonzero(~dev)
    flags = z["dist"][ix][:, VIEWS["flags"]]
    views = {"directed": flags, "unordered": unordered(flags)}
    args.out.mkdir(parents=True, exist_ok=False)
    with threadpool_limits(limits=4):
        evaluate(
            z["y"][ix],
            z["fold"][ix],
            z["files"][ix],
            views,
            list(views),
            args.out,
            factory=PairModel,
        )
        p, q = [scores(args.out / name / "oof.csv", z, ix) for name in views]
        if not np.array_equal(np.isfinite(p), np.isfinite(q)):
            raise ValueError("scoring support differs")
        save(
            args.out / "directed-minus-unordered.json",
            paired(z["y"][ix], z["fold"][ix], p, q),
        )


if __name__ == "__main__":
    main()
