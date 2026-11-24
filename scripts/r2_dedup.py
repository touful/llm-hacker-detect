"""Retrain on two deduplicated versions of the frozen confirmation cohort."""

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from agenttrace import study


# Frozen confirmation panel (flags is the locked projection; no reselection).
_MODELS = ("flags", "dist-hgb")


def dedup(z, dev, ix, out):
    """Filter once before five-fold CV, without changing any surviving fold ID."""
    dev, ix = np.asarray(dev), np.asarray(ix)
    for indices in (dev, ix):
        if (
            indices.ndim != 1
            or indices.dtype.kind not in "iu"
            or not np.all(indices[1:] > indices[:-1])
        ):
            raise ValueError("expected increasing integer indices")
    if not np.array_equal(
        np.sort(np.concatenate((dev, ix))), np.arange(len(z["files"]))
    ):
        raise ValueError("development and confirmation must partition the cohort")

    keys = study.rowkeys(z["dist"][ix])
    devkeys = set(study.rowkeys(z["dist"][dev])) - {None}
    folds = z["fold"][ix]
    ids = [int(Path(f).stem) for f in z["files"]]
    if len(ids) != len(set(ids)):
        raise ValueError("numeric file IDs are not unique")

    groups = defaultdict(list)
    for i, key in enumerate(keys):
        if key is not None:
            groups[key].append(i)
    empty = np.array([k is None for k in keys], dtype=bool)
    overlap = np.array([k in devkeys for k in keys], dtype=bool)
    cross = study.duplicates(keys, folds, set())
    strict = ~(overlap | cross)
    representative = empty.copy()
    winners = {}
    for key, members in groups.items():
        if key not in devkeys:
            i = min(members, key=lambda j: ids[ix[j]])
            winners[key] = i
            representative[i] = True

    counts = dict(
        development=len(dev),
        confirmation=len(ix),
        empty=int(empty.sum()),
        dev_overlap=int(overlap.sum()),
        cross_fold=int(cross.sum()),
        excluded_union=int((~strict).sum()),
    )
    assert counts == dict(
        development=533,
        confirmation=4795,
        empty=246,
        dev_overlap=278,
        cross_fold=455,
        excluded_union=473,
    ), counts
    designs = {"strict": strict, "representative": representative}
    for name, keep in designs.items():
        kept_keys = [keys[i] for i in np.flatnonzero(keep)]
        assert (
            keep[empty].all()
            and not study.duplicates(kept_keys, folds[keep], devkeys).any()
        )
        assert int(keep.sum()) == {"strict": 4322, "representative": 4359}[name]
    assert set(winners) == groups.keys() - devkeys
    assert representative.sum() - empty.sum() == len(winners)

    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    with (out / "audit.csv").open("w", encoding="utf8", newline="") as stream:
        w = csv.writer(stream)
        w.writerow(
            (
                "cohort_index",
                "file",
                "fold",
                "y",
                "dist_sha256",
                "dev_overlap",
                "cross_fold",
                "strict_keep",
                "representative_keep",
                "representative_file",
            )
        )
        for i, (idx, key) in enumerate(zip(ix, keys)):
            winner = i if key is None else winners.get(key)
            w.writerow(
                (
                    idx,
                    z["files"][idx],
                    folds[i],
                    z["y"][idx],
                    key.hex() if key else "",
                    int(overlap[i]),
                    int(cross[i]),
                    int(strict[i]),
                    int(representative[i]),
                    z["files"][ix[winner]] if winner is not None else "",
                )
            )

    summary = dict(
        key="study.rowkeys(full dist); empty rows are never grouped", counts=counts
    )
    with threadpool_limits(limits=4):
        for name, keep in designs.items():
            result = study.run(z, ix[keep], _MODELS, out / name)
            n, scored = int(keep.sum()), int((keep & ~empty).sum())
            for model in ("dist-hgb", "flags"):
                support = result[model]["metrics"]["support"]
                assert (support["n"], support["scored"]) == (n, scored)
            summary[name] = dict(
                n=n,
                nonempty=scored,
                removed=len(ix) - n,
                models={
                    m: {
                        k: result[m][k]
                        for k in ("metrics", "common_support_metrics", "low_fpr")
                    }
                    for m in _MODELS
                },
            )
    study.save(out / "summary.json", summary)
    return summary
