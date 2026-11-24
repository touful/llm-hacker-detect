"""Validation-only score calibration and pooled out-of-fold SLR diagnostics."""

import csv
import json
from pathlib import Path

import numpy as np
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression


_MODELS = ("dist", "flags")
_METHODS = ("platt", "laplace_pav")
_KEYS = ("raw", *(f"{m}_{s}" for m in _METHODS for s in ("post", "log10lr")))
_EPS = 1e-12
_LN10 = np.log(10.0)
_PLATT = dict(C=1.0, solver="lbfgs", max_iter=1000, tol=1e-10)


def _prob(x):
    x = np.asarray(x, dtype=float)
    if np.any((x < 0) | (x > 1)):
        raise ValueError("probabilities must be in [0, 1] or NaN")
    return x


def _labels(y):
    y = np.asarray(y)
    if y.ndim != 1 or not np.isin(y, [0, 1]).all():
        raise ValueError("expected a vector of binary labels")
    return y.astype(int)


def _raw_logit(x):
    return logit(np.clip(x, _EPS, 1 - _EPS))


def _pav(scores, y):
    """Stack of (left score, positives, negatives); ties cannot be split."""
    order = np.argsort(scores, kind="stable")
    x, idx, counts = np.unique(scores[order], return_index=True, return_counts=True)
    positives = np.add.reduceat(y[order], idx)
    blocks = []
    for left, p, count in zip(x, positives, counts):
        p, n = int(p), int(count - p)
        # Compare odds by exact integer products, including at pure endpoints.
        while blocks and blocks[-1][1] * n >= p * blocks[-1][2]:
            left, prev_p, prev_n = blocks.pop()
            p += prev_p
            n += prev_n
        blocks.append((left, p, n))
    return blocks


def fit_cal(rawval, yval):
    """Fit on scored validation cases; apply to any shape without dropping NaNs."""
    x, y = _prob(rawval), _labels(yval)
    if x.shape != y.shape:
        raise ValueError("validation scores and labels are not aligned")
    scored = ~np.isnan(x)
    x, y = x[scored], y[scored]
    n1, n0 = int(y.sum()), int(len(y) - y.sum())
    if not min(n1, n0):
        raise ValueError("both classes must have scored validation cases")
    prior_logodds = np.log(n1 / n0)
    platt = LogisticRegression(**_PLATT).fit(_raw_logit(x)[:, None], y)
    slope, intercept = float(platt.coef_[0, 0]), float(platt.intercept_[0])
    blocks = _pav(x, y)
    left = np.array([a for a, _, _ in blocks])
    # Global add-one posterior endpoints, not smoothing of individual blocks.
    lo, hi = 1 / (len(y) + 2), (len(y) + 1) / (len(y) + 2)
    rates = np.clip([p / (p + n) for _, p, n in blocks], lo, hi)

    def apply(rawarray):
        raw = _prob(rawarray)
        mask = ~np.isnan(raw)
        x = raw[mask]
        result = {k: np.full(raw.shape, np.nan) for k in _KEYS[1:]}
        z = intercept + slope * _raw_logit(x)
        result["platt_post"][mask] = expit(z)
        result["platt_log10lr"][mask] = (z - prior_logodds) / _LN10
        i = np.maximum(0, np.searchsorted(left, x, side="right") - 1)
        result["laplace_pav_post"][mask] = rates[i]
        result["laplace_pav_log10lr"][mask] = (logit(rates[i]) - prior_logodds) / _LN10
        return result

    apply.params = dict(
        n_positive=n1,
        n_negative=n0,
        prior=n1 / len(y),
        slope=slope,
        intercept=intercept,
        pav_left=left.tolist(),
        pav_post=rates.tolist(),
        posterior_bounds=[lo, hi],
    )
    return apply


def _min_cllr(score, y):
    """Oracle on the supplied pooled ordering, with equal class mass, in bits."""
    n1, n0 = int(y.sum()), int(len(y) - y.sum())
    loss = 0.0
    for _, p, n in _pav(score, y):
        # Pure blocks take the exact optimum LR of 0 or infinity (zero loss).
        if p and n:
            a, b = p / n1, n / n0
            loss += a * np.log1p(b / a) + b * np.log1p(a / b)
    return float(loss / (2 * np.log(2.0)))


def _posterior_stats(post, y):
    edges = np.linspace(0, 1, 16)
    ids = np.searchsorted(edges[1:-1], post, side="right")
    bins, ece, mce = [], 0.0, 0.0
    for i in range(15):
        mask = ids == i
        count = int(mask.sum())
        conf = float(post[mask].mean()) if count else None
        rate = float(y[mask].mean()) if count else None
        bins.append(
            dict(
                bin_lo=edges[i],
                bin_hi=edges[i + 1],
                count=count,
                mean_conf=conf,
                emp_pos_rate=rate,
            )
        )
        if count:
            error = abs(conf - rate)
            ece += count / len(y) * error
            mce = max(mce, error)
    z = _raw_logit(post)
    return dict(
        bins=bins,
        ece15=ece,
        mce15=mce,
        brier=float(np.mean((post - y) ** 2)),
        log_loss=float(np.logaddexp(0, np.where(y, -z, z)).mean()),
    )


def _stats(post, lr, y):
    result = _posterior_stats(post, y)
    agent, human = lr[y == 1], lr[y == 0]
    loss = (
        np.logaddexp(0, -_LN10 * agent).mean() + np.logaddexp(0, _LN10 * human).mean()
    ) / (2 * np.log(2.0))
    lower = _min_cllr(lr, y)
    result.update(
        cllr_bits=float(loss), cllr_min_bits=lower, cllr_cal_bits=float(loss - lower)
    )
    result["tippett"] = dict(
        agent_ge10=float(np.mean(agent >= 1)),
        agent_ge100=float(np.mean(agent >= 2)),
        human_lt1=float(np.mean(human < 0)),
        human_le01=float(np.mean(human <= -1)),
    )
    for group, scores in [("agent", agent), ("human", human)]:
        q10, median, q90 = np.quantile(scores, [0.1, 0.5, 0.9])
        result[group + "_log10lr_quantiles"] = dict(
            q10=q10,
            median=median,
            q90=q90,
            frac_lr_lt_1=float(np.mean(scores < 0)),
            frac_lr_ge_1=float(np.mean(scores >= 0)),
            frac_lr_ge_10=float(np.mean(scores >= 1)),
        )
    return result


def slr_export(out, files, fold, y, values):
    """Write all-file oof_slr.csv, scored Tippett ECDFs and summary.json to a new directory."""
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    files, fold, y = np.asarray(files), np.asarray(fold), _labels(y)
    if files.shape != y.shape or fold.shape != y.shape or fold.dtype.kind not in "iu":
        raise ValueError("expected aligned file IDs, integer folds and binary labels")
    summary = {
        "method": {
            "platt": dict(
                input="logit(raw)", input_clip_epsilon=_EPS, l2=True, **_PLATT
            ),
            "laplace_pav": "Tie-pooled increasing validation PAV; right-continuous step, constant tails. "
            "Clip posterior to [1/(N+2),(N+1)/(N+2)], N=scored validation size.",
            "elub_pav": "CSV alias of laplace_pav; these are Laplace posterior bounds, NOT empirical ELUB.",
            "log10lr": "Subtract log10 positive:negative odds of scored validation from log10 posterior odds.",
            "cllr_min_bits": "Exact empirical pooled best nondecreasing map of each reported log10LR; "
            "class-balanced cost in bits, ties inseparable, pure blocks contribute zero.",
            "ece15_mce15": "Posterior vs positive rate; 15 equal-width [lo,hi) bins, last includes 1; "
            "count-weighted mean and maximum absolute gap over nonempty bins.",
            "log_loss": f"Natural logs; clip probabilities to [{_EPS}, 1-{_EPS}] for log loss and raw logit.",
            "tippett": "Scored class-conditional empirical CDF P(log10LR <= threshold).",
        }
    }
    columns, curves = {}, []
    for model in _MODELS:
        a = {k: np.asarray(values[model][k], dtype=float) for k in _KEYS}
        raw = _prob(a["raw"])
        mask = np.isfinite(raw)
        for k, v in a.items():
            if (
                v.shape != y.shape
                or np.isinf(v).any()
                or not np.array_equal(np.isfinite(v), mask)
            ):
                raise ValueError(
                    f"{model}/{k}: scores must align and share the raw abstention mask"
                )
            if k.endswith("_post"):
                _prob(v)
            columns[f"{model}_{k.replace('laplace_pav', 'elub_pav')}"] = v
        labels = y[mask]
        if len(np.unique(labels)) != 2:
            raise ValueError(
                f"{model}: pooled Cllr requires scored examples of both classes"
            )
        record = summary[model] = dict(
            n_total=len(y),
            n_scored=len(labels),
            n_positive=int(labels.sum()),
            n_negative=int((labels == 0).sum()),
        )
        # Bounded PAV is invertible to the validation prior via its exported odds and LR.
        # This avoids substituting the test prevalence or assuming validation fold numbering.
        prior_odds = (
            logit(a["laplace_pav_post"][mask]) / _LN10 - a["laplace_pav_log10lr"][mask]
        )
        priors, row_prior = {}, np.empty(len(labels))
        for f in np.unique(fold[mask]):
            take = fold[mask] == f
            p = float(np.median(prior_odds[take]))
            if not np.isfinite(p) or not np.allclose(
                prior_odds[take], p, atol=1e-8, rtol=0
            ):
                raise ValueError(
                    f"{model}: inconsistent validation odds in test fold {f}"
                )
            priors[str(f)] = float(expit(p * _LN10))
            row_prior[take] = p
        record["validation_prior_by_test_fold"] = priors
        record["raw_score_min_cllr_bits"] = _min_cllr(raw[mask], labels)
        for method in ("raw", *_METHODS):
            post = raw[mask] if method == "raw" else a[f"{method}_post"][mask]
            lr = (
                _raw_logit(post) / _LN10 - row_prior
                if method == "raw"
                else a[f"{method}_log10lr"][mask]
            )
            record[method] = _stats(post, lr, labels)
            for label in (0, 1):
                x, counts = np.unique(lr[labels == label], return_counts=True)
                curves.extend(
                    (model, method, label, t, n, c)
                    for t, n, c in zip(x, counts, np.cumsum(counts) / counts.sum())
                )

    out.mkdir(parents=True, exist_ok=False)
    with (out / "oof_slr.csv").open("x", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "fold", "y", *columns])
        for name, k, target, *row in zip(files, fold, y, *columns.values()):
            w.writerow([name, k, target, *(None if np.isnan(v) else v for v in row)])
    with (out / "tippett.csv").open("x", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "method", "y", "log10lr", "count", "cdf"])
        w.writerows(curves)
    with (out / "summary.json").open("x", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, allow_nan=False)
        f.write("\n")
    return summary
