"""Export confirmation-fold fusion predictions without fitting or loading models."""

import csv
from pathlib import Path

import numpy as np

from agenttrace.evaluate import (
    _TARGETS,
    _cutoffs,
    _metrics,
    _operating_point,
    _ratio,
    _scores,
)


_CASCADE = "flags-behavior-cascade"
_MODELS = ("dist-hgb", "view-fusion", "joint-hgb", _CASCADE)


def _views(raw, n):
    dist = _scores(raw["dist-hgb"], n)
    behavior = _scores(raw["behavior-hgb"], n)
    flags = _scores(raw["flags_platt"], n)
    fallback = _scores(raw["behavior_platt"], n)
    source = np.where(
        np.isfinite(flags),
        "http",
        np.where(np.isfinite(fallback), "transport_fallback", "abstain"),
    )

    # Match _ViewFusion.predict, including summation dtype and branch order.
    p = np.stack([raw["behavior-hgb"], raw["dist-hgb"]])
    count = np.isfinite(p).sum(0)
    fused = np.divide(
        np.nansum(p, axis=0), count, out=np.full(n, np.nan), where=count > 0
    )
    return (
        {
            "dist-hgb": dist,
            "view-fusion": fused,
            "joint-hgb": _scores(raw["joint-hgb"], n),
            _CASCADE: np.where(np.isfinite(flags), flags, fallback),
        },
        source,
        np.isfinite(behavior),
    )


def _point(y, decision):
    point = _operating_point(y, decision)
    # An abstention is not a negative decision, but it never triggers an alert.
    point["unconditional_fpr"] = _ratio(point["confusion"]["fp"], int((y == 0).sum()))
    point["unconditional_tpr"] = _ratio(point["confusion"]["tp"], int((y == 1).sum()))
    return point


def _indices(raw, n):
    idx = np.asarray(raw)
    if (
        idx.ndim != 1
        or idx.dtype.kind not in "iu"
        or not len(idx)
        or np.any(idx < 0)
        or np.any(idx >= n)
        or len(np.unique(idx)) != len(idx)
    ):
        raise ValueError("expected unique, in-range confirmation indices")
    return idx


def _write_rows(path, files, fold, y, scores, decisions, source):
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        header = [
            "file",
            "fold",
            "y",
            "score",
            "pred",
            *(f"pred_fpr_{t}pct" for t in _TARGETS),
        ]
        writer.writerow(header + (["source_view"] if source is not None else []))
        for i, score in enumerate(scores):
            finite = np.isfinite(score)
            row = [
                files[i],
                fold[i],
                y[i],
                score if finite else None,
                int(score >= 0.5) if finite else None,
            ]
            for t in _TARGETS:
                d = decisions[t][i]
                row.append(int(d) if np.isfinite(d) else None)
            writer.writerow(row + ([source[i]] if source is not None else []))


def fusion_export(out, files, fold, y, byfold):
    """Write <out>/<model>/oof.csv and return a JSON-serializable summary.

    Cutoffs use all finite negative validation scores of the deployed rule;
    HTTP and transport reports slice the same test decisions, without retuning.
    Cohort masks come from finite dist-hgb and behavior-hgb test predictions,
    respectively. The all cohort retains abstentions in unconditional rates.
    """
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    files, fold, y = map(np.asarray, (files, fold, y))
    if (
        y.ndim != 1
        or not len(y)
        or files.shape != y.shape
        or fold.shape != y.shape
        or fold.dtype.kind not in "iu"
        or not np.isin(y, [0, 1]).all()
    ):
        raise ValueError(
            "expected aligned confirmation files, integer folds and binary labels"
        )
    y = y.astype(int)
    n = len(y)
    scores = {m: np.full(n, np.nan) for m in _MODELS}
    decisions = {m: {t: np.full(n, np.nan) for t in _TARGETS} for m in _MODELS}
    thresholds = {m: {} for m in _MODELS}
    transport = np.zeros(n, dtype=bool)
    source = np.empty(n, dtype=object)
    seen = np.zeros(n, dtype=bool)
    for batch in byfold:
        k = batch["k"]
        val, test = (_indices(batch[key], n) for key in ("val", "test"))
        if (
            seen[test].any()
            or not np.array_equal(np.sort(test), np.flatnonzero(fold == k))
            or np.any(fold[val] == k)
            or np.unique(fold[val]).size != 1
        ):
            raise ValueError(f"misaligned validation/test indices for test fold {k}")
        v, _, _ = _views(batch["v"], len(val))
        p, source[test], transport[test] = _views(batch["t"], len(test))
        for m in _MODELS:
            scores[m][test] = p[m]
            cuts = _cutoffs(y[val], v[m])
            thresholds[m][str(k)] = {str(t): cut for t, cut in cuts.items()}
            keep = np.isfinite(p[m])
            for t, cut in cuts.items():
                if cut is not None:
                    decisions[m][t][test[keep]] = p[m][keep] > cut
        seen[test] = True
    if not seen.all():
        raise ValueError("test folds do not cover the confirmation cohort")

    cohorts = {
        "all": seen,
        "http": np.isfinite(scores["dist-hgb"]),
        "transport": transport,
    }
    summary = {"n": n, "models": {}}
    for m in _MODELS:
        reports = {}
        for name, mask in cohorts.items():
            labels = y[mask]
            reports[name] = {
                "metrics": _metrics(labels, scores[m][mask]),
                "low_fpr": {
                    str(t): _point(labels, decisions[m][t][mask]) for t in _TARGETS
                },
            }
            if m == _CASCADE:
                by_source = {}
                for branch in ("http", "transport_fallback", "abstain"):
                    rows = mask & (source == branch)
                    by_source[branch] = {
                        "n": int(rows.sum()),
                        "human": int((rows & (y == 0)).sum()),
                        "agent": int((rows & (y == 1)).sum()),
                        "low_fpr": {
                            str(t): _point(y[rows], decisions[m][t][rows])
                            for t in _TARGETS
                        },
                    }
                reports[name]["by_source_view"] = by_source
        summary["models"][m] = {
            "thresholds_exclusive_by_fold": thresholds[m],
            "cohorts": reports,
        }

    out.mkdir(parents=True, exist_ok=False)
    for m in _MODELS:
        directory = out / m
        directory.mkdir()
        _write_rows(
            directory / "oof.csv",
            files,
            fold,
            y,
            scores[m],
            decisions[m],
            source if m == _CASCADE else None,
        )
    return summary
