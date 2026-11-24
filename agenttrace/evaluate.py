"""Evaluate session classifiers on fixed outer folds; see docs/metrics.md."""

import csv
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from sklearn import metrics as sk


_TARGETS = (1, 5)  # percentages, to keep the integer FP budget exact


def _ratio(num, den):
    return float(num / den) if den else None


def _support(y, keep):
    def counts(mask):
        n, scored = int(mask.sum()), int((mask & keep).sum())
        return {"n": n, "scored": scored, "coverage": _ratio(scored, n)}

    return {
        **counts(np.ones(len(y), dtype=bool)),
        "by_class": {str(c): counts(y == c) for c in (0, 1)},
    }


def _confusion(y, pred):
    if len(y) == 0:
        return dict(tn=0, fp=0, fn=0, tp=0)
    tn, fp, fn, tp = sk.confusion_matrix(y, pred, labels=[0, 1]).ravel().tolist()
    return {"tn": tn, "fp": fp, "fn": fn, "tp": tp}


def _metrics(y, scores):
    keep = np.isfinite(scores)
    support = _support(y, keep)
    y, p = y[keep], scores[keep]
    pred = (p >= 0.5).astype(int)
    cm = _confusion(y, pred)
    both = np.unique(y).size == 2
    tp, tn, fp, fn = (cm[k] for k in ("tp", "tn", "fp", "fn"))
    return {
        "support": support,
        "confusion": cm,
        "ap": float(sk.average_precision_score(y, p)) if tp + fn else None,
        "auc": float(sk.roc_auc_score(y, p)) if both else None,
        "accuracy": _ratio(tp + tn, len(y)),
        "balanced_accuracy": float(sk.balanced_accuracy_score(y, pred))
        if both
        else None,
        "mcc": float(sk.matthews_corrcoef(y, pred)) if both else None,
        "f1": float(sk.f1_score(y, pred, zero_division=0)) if len(y) else None,
        "precision": _ratio(tp, tp + fp),
        "recall": _ratio(tp, tp + fn),
        "specificity": _ratio(tn, tn + fp),
        "brier": float(sk.brier_score_loss(y, p, pos_label=1)) if len(y) else None,
        "log_loss": float(sk.log_loss(y, p, labels=[0, 1])) if len(y) else None,
    }


def _cutoffs(y, scores):
    # Strictly above the first excluded negative. Never split equal scores.
    neg = np.sort(scores[(y == 0) & np.isfinite(scores)])[::-1]
    return {t: float(neg[len(neg) * t // 100]) if len(neg) else None for t in _TARGETS}


def _wilson95(hits, total):
    if not total:
        return None
    p, z = hits / total, 1.959963984540054
    scale = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / scale
    half = z * (p * (1 - p) / total + z**2 / (4 * total**2)) ** 0.5 / scale
    return [max(0.0, center - half), min(1.0, center + half)]


def _operating_point(y, decisions):
    keep = np.isfinite(decisions)
    cm = _confusion(y[keep], decisions[keep].astype(int))
    return {
        "support": _support(y, keep),
        "confusion": cm,
        "fpr": _ratio(cm["fp"], cm["tn"] + cm["fp"]),
        "tpr": _ratio(cm["tp"], cm["fn"] + cm["tp"]),
        "fpr_wilson95": _wilson95(cm["fp"], cm["tn"] + cm["fp"]),
        "tpr_wilson95": _wilson95(cm["tp"], cm["fn"] + cm["tp"]),
    }


def _fold_spread(rows):
    spread = {}
    for metric in ("ap", "auc", "mcc", "f1", "balanced_accuracy", "accuracy"):
        vals = [r["metrics"][metric] for r in rows if r["metrics"][metric] is not None]
        spread[metric] = {
            "n_folds": len(vals),
            "sd": float(np.std(vals, ddof=1)) if len(vals) >= 2 else None,
        }
    return spread


def _scores(raw, n):
    p = np.asarray(raw, dtype=float)
    if p.shape != (n,):
        raise ValueError(
            f"expected {n} positive-class probabilities, got shape {p.shape}"
        )
    if np.any((p < 0) | (p > 1)):
        raise ValueError("model output must be in [0,1] or NaN for abstention")
    return p


def _write_oof(path, files, y, folds, p, decisions, common):
    def cell(value):
        return int(value) if np.isfinite(value) else None

    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "file",
                "fold",
                "y",
                "score",
                "pred",
                "pred_fpr_1pct",
                "pred_fpr_5pct",
                "common_support",
            ]
        )
        for i, score in enumerate(p):
            writer.writerow(
                [
                    files[i],
                    folds[i],
                    y[i],
                    score if np.isfinite(score) else None,
                    int(score >= 0.5) if np.isfinite(score) else None,
                    *(cell(decisions[t][i]) for t in _TARGETS),
                    int(common[i]),
                ]
            )


def _json(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def evaluate(
    y, folds, files, X, names, out, seed=42, epochs=12, batch_size=32, *, factory=None
):
    """Fit one fresh model per test fold and return the serialized summary as a dict."""
    from .models import create

    factory = create if factory is None else factory
    y, folds = np.asarray(y), np.asarray(folds)
    if y.ndim != 1 or not np.isin(y, [0, 1]).all():
        raise ValueError("y must be a vector of binary 0/1 labels")
    y = y.astype(int)
    files = list(files)
    if folds.shape != y.shape or len(files) != len(y):
        raise ValueError("y, folds, files must be aligned")
    if folds.dtype.kind not in "iu" or np.unique(folds).size != 5:
        raise ValueError("expected exactly five distinct integer fold IDs")
    fold_ids = np.unique(folds).tolist()
    features = {k: np.asarray(v) for k, v in X.items()}
    if not features or any(v.ndim == 0 or len(v) != len(y) for v in features.values()):
        raise ValueError("each feature array must have one row per label")

    def subset(indices):
        return {k: v[indices] for k, v in features.items()}

    names = list(names)
    if not names or len(set(names)) != len(names):
        raise ValueError("expected nonempty unique model names")
    for name in names:
        if (
            not isinstance(name, str)
            or not name
            or not all(c.isalnum() or c in "_-" for c in name)
        ):
            raise ValueError(f"invalid model name: {name!r}")

    results, predictions, decisions = {}, {}, {}
    for name in names:
        start = perf_counter()
        p = np.full(len(y), np.nan)
        op = {t: np.full(len(y), np.nan) for t in _TARGETS}
        rows = []
        for i, test_fold in enumerate(fold_ids):
            val_fold = fold_ids[(i + 1) % 5]
            test = np.flatnonzero(folds == test_fold)
            val = np.flatnonzero(folds == val_fold)
            train = np.flatnonzero((folds != val_fold) & (folds != test_fold))
            t0 = perf_counter()
            model = factory(name, seed=seed + i, epochs=epochs, batch_size=batch_size)
            model.fit(subset(train), y[train])
            t1 = perf_counter()
            vp = _scores(model.predict(subset(val)), len(val))
            cuts = _cutoffs(y[val], vp)
            t2 = perf_counter()
            tp = _scores(model.predict(subset(test)), len(test))
            t3 = perf_counter()
            p[test] = tp
            low = {}
            for t, cut in cuts.items():
                if cut is not None:
                    mask = np.isfinite(tp)
                    op[t][test[mask]] = tp[mask] > cut
                low[str(t)] = {
                    "threshold_exclusive": cut,
                    "validation_fpr": _ratio(
                        int(((y[val] == 0) & (vp > cut)).sum()),
                        int(((y[val] == 0) & np.isfinite(vp)).sum()),
                    )
                    if cut is not None
                    else None,
                    **_operating_point(y[test], op[t][test]),
                }
            rows.append(
                {
                    "test_fold": test_fold,
                    "validation_fold": val_fold,
                    "n_train": len(train),
                    "validation_support": _support(y[val], np.isfinite(vp)),
                    "metrics": _metrics(y[test], tp),
                    "low_fpr": low,
                    "seconds": {
                        "fit": t1 - t0,
                        "validation": t2 - t1,
                        "test_predict": t3 - t2,
                    },
                }
            )
            del model
        results[name] = {
            "metrics": _metrics(y, p),
            "folds": rows,
            "low_fpr": {str(t): _operating_point(y, op[t]) for t in _TARGETS},
            "fold_spread": _fold_spread(rows),
            "seconds": perf_counter() - start,
        }
        predictions[name], decisions[name] = p, op

    common = np.logical_and.reduce([np.isfinite(p) for p in predictions.values()])
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for name, result in results.items():
        result["common_support_metrics"] = _metrics(
            y, np.where(common, predictions[name], np.nan)
        )
        model_dir = out / name
        model_dir.mkdir(exist_ok=True)
        _write_oof(
            model_dir / "oof.csv",
            files,
            y,
            folds,
            predictions[name],
            decisions[name],
            common,
        )
        _json(model_dir / "summary.json", result)
    summary = {
        "seed": int(seed),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "models": results,
    }
    _json(out / "summary.json", summary)
    return summary
