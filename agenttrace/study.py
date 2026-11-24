"""Run frozen feature selection and confirmatory within-cohort CV. No packet payload I/O."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import sklearn
from sklearn.metrics import average_precision_score as ap
from threadpoolctl import threadpool_limits
from .ablation import ALL, CANDIDATES, COLUMNS, VIEWS, choose
from .evaluate import evaluate
from .models import create, _Model


class Projected(_Model):
    def __init__(self, key, **kwargs):
        super().__init__("dist-hgb", **kwargs)
        self.columns = VIEWS[key]

    def _inputs(self, X):
        a = X["dist"][:, self.columns]
        return (a,), np.isfinite(a).any(1)


def factory(name, **kwargs):
    return Projected(name, **kwargs) if name in VIEWS else create(name, **kwargs)


def save(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf8")


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in ("y", "fold", "files", "dist", "conv", "count")} | {
            "meta": json.loads(str(z["meta"]))
        }


def run(z, idx, models, out, y=None):
    return evaluate(
        z["y"][idx] if y is None else y,
        z["fold"][idx],
        z["files"][idx],
        {k: z[k][idx] for k in ("dist", "conv")},
        models,
        out,
        factory=factory,
    )["models"]


def scores(path, z, idx):
    with path.open(encoding="utf8", newline="") as f:
        rows = list(csv.DictReader(f))
    if [r["file"] for r in rows] != z["files"][idx].tolist() or [
        int(r["y"]) for r in rows
    ] != z["y"][idx].tolist():
        raise ValueError("misaligned predictions")
    return np.array([float(r["score"]) if r["score"] else np.nan for r in rows])


def paired(y, fold, p, q):
    ok = np.isfinite(p) & np.isfinite(q)
    y, fold, p, q = (a[ok] for a in (y, fold, p, q))
    rng = np.random.default_rng(2026)
    groups = [
        np.flatnonzero((fold == f) & (y == c))
        for f in np.unique(fold)
        for c in np.unique(y)
    ]
    d = []
    for _ in range(1000):
        ix = np.concatenate(
            [rng.choice(g, len(g), replace=True) for g in groups if len(g)]
        )
        d.append(ap(y[ix], p[ix]) - ap(y[ix], q[ix]))
    return dict(
        n=len(y),
        ap_first=ap(y, p),
        ap_second=ap(y, q),
        delta=ap(y, p) - ap(y, q),
        ci95=np.quantile(d, [0.025, 0.975]).tolist(),
    )


def rowkeys(a):
    # 缺失统一编码，数值范围 [0,1]，全空不视为重复流量。
    return [
        hashlib.sha256(np.nan_to_num(r, nan=-1.0).astype("<f4").tobytes()).digest()
        if np.isfinite(r).any()
        else None
        for r in a
    ]


def duplicates(keys, folds, devkeys):
    byfold = {}
    for k, f in zip(keys, folds):
        if k is not None:
            byfold.setdefault(k, set()).add(f)
    return np.array(
        [k is not None and (k in devkeys or len(byfold[k]) > 1) for k in keys]
    )


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("cohort", type=Path)
    p.add_argument("pilot", type=Path)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    z = load(args.cohort)
    e = load(args.pilot)
    ids = z["files"]
    dev = np.flatnonzero(np.isin(ids, e["files"]))
    confirm = np.flatnonzero(~np.isin(ids, e["files"]))
    if (
        len(set(ids)) != len(ids)
        or len(dev) != len(e["files"])
        or not np.array_equal(ids[dev], e["files"])
        or not np.array_equal(z["y"][dev], e["y"])
        or not np.array_equal(z["fold"][dev], e["fold"])
    ):
        raise ValueError("cohort mismatch")
    if (
        z["meta"]["max_packets"] != 2048
        or e["meta"]["max_packets"] is not None
        or z["meta"]["labels_sha256"] != e["meta"]["labels_sha256"]
    ):
        raise ValueError("unexpected cache or observation budget")
    out = args.out
    out.mkdir(parents=True, exist_ok=False)
    with threadpool_limits(limits=4):
        devdir = out / "development"
        run(z, dev, ["dist-hgb", *VIEWS], devdir)
        ref = scores(devdir / "dist-hgb/oof.csv", z, dev)
        aps = {"full": ap(z["y"][dev][np.isfinite(ref)], ref[np.isfinite(ref)])}
        for key in CANDIDATES:
            v = scores(devdir / key / "oof.csv", z, dev)
            assert np.array_equal(np.isfinite(ref), np.isfinite(v))
            aps[key] = ap(z["y"][dev][np.isfinite(v)], v[np.isfinite(v)])
        choice = choose(aps)
        selected = choice if choice != "full" else "dist-hgb"
        cols = VIEWS.get(choice, ALL)
        save(
            out / "lock.json",
            dict(
                choice=selected,
                dim=len(cols),
                tolerance=0.015,
                dev_ap=aps,
                features=[(int(i), COLUMNS[i]) for i in cols],
                sklearn=sklearn.__version__,
                numpy=np.__version__,
            ),
        )
        print("LOCK", selected, len(cols), flush=True)
        cdir = out / "confirmation"
        names = list(
            dict.fromkeys(
                [
                    selected,
                    "dist-hgb",
                    "dist-et",
                    "dist-gbdt",
                    "dist-linear",
                    "behavior-hgb",
                ]
            )
        )
        run(z, confirm, names, cdir)
        yy = z["y"][confirm]
        ff = z["fold"][confirm]
        rng = np.random.default_rng(2026)
        yp = yy.copy()
        for f in np.unique(ff):
            yp[ff == f] = rng.permutation(yy[ff == f])
        run(z, confirm, ["dist-hgb"], out / "permuted", yp)
        ei = np.arange(len(dev))
        run(e, ei, ["dist-hgb"], out / "development-eof")
        eof = scores(out / "development-eof/dist-hgb/oof.csv", e, ei)
        pf = scores(cdir / "dist-hgb/oof.csv", z, confirm)
        ps = scores(cdir / selected / "oof.csv", z, confirm)
        keys = rowkeys(z["dist"][confirm])
        devkeys = set(rowkeys(z["dist"][dev])) - {None}
        dup = duplicates(keys, ff, devkeys)
        ok = np.isfinite(pf) & ~dup
        save(
            out / "comparisons.json",
            dict(
                compact_minus_full=paired(yy, ff, ps, pf),
                eof_minus_prefix=paired(z["y"][dev], z["fold"][dev], eof, ref),
                duplication=dict(
                    n_scored=int(np.isfinite(pf).sum()),
                    cross_fold_rows=sum(int(v) for v in duplicates(keys, ff, set())),
                    dev_overlap_rows=sum(k in devkeys for k in keys),
                    excluded_union=int(dup.sum()),
                    n_remaining=int(ok.sum()),
                    ap_full_on_remaining=ap(yy[ok], pf[ok]),
                    ap_selected_on_remaining=ap(yy[ok], ps[ok]),
                ),
            ),
        )


if __name__ == "__main__":
    main()
