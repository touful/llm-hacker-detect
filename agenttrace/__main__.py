"""Two-stage CLI: extract immutable session features, then compare models offline."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter
import numpy as np
from .models import NAMES


def extract(args):
    from .capture import capture
    from .views import features

    labels = args.labels.resolve()
    root = labels.parent
    with labels.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    files = [r["file"] for r in rows]
    y = np.array([int(r["label"]) for r in rows])
    folds = np.array([int(r["fold"]) for r in rows])
    idx = np.arange(len(rows))
    if (
        len(set(files)) != len(rows)
        or not np.isin(y, [0, 1]).all()
        or len(set(folds)) != 5
    ):
        raise ValueError(
            "expected a unique session per row, two binary labels and five folds"
        )
    if args.fraction < 1:
        rng = np.random.default_rng(args.seed)
        idx = np.sort(
            np.concatenate(
                [
                    rng.choice(g, max(1, round(len(g) * args.fraction)), replace=False)
                    for f in np.unique(folds)
                    for c in [0, 1]
                    if len(g := idx[(folds == f) & (y == c)])
                ]
            )
        )
    arrays = {k: [] for k in ("conv", "dist", "style", "bins")}
    counts = []
    start = perf_counter()
    for step, i in enumerate(idx, 1):
        path = (root / files[i]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("capture outside dataset")
        try:
            events = capture(path, args.max_packets)
            view = features(*events)
        except Exception as exc:
            raise RuntimeError(f"extraction failed: {files[i]}") from exc
        for k in arrays:
            arrays[k].append(view[k])
        counts.append(tuple(map(len, events)))
        del events
        if step % 100 == 0:
            print(f"{step}/{len(idx)}", flush=True)
    meta = dict(
        schema="passive-http-v1",
        max_packets=args.max_packets,
        fraction=args.fraction,
        seed=args.seed,
        labels_sha256=hashlib.sha256(labels.read_bytes()).hexdigest(),
        seconds=perf_counter() - start,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("xb") as f:
        np.savez_compressed(
            f,
            **arrays,
            count=counts,
            y=y[idx],
            fold=folds[idx],
            files=np.array(files)[idx],
            meta=json.dumps(meta),
        )
    print(meta, flush=True)


def main():
    p = argparse.ArgumentParser(__doc__)
    sub = p.add_subparsers(dest="op", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--labels", type=Path, required=True)
    e.add_argument("--out", type=Path, required=True)
    e.add_argument("--max-packets", type=int)
    e.add_argument("--fraction", type=float, default=1.0)
    e.add_argument("--seed", type=int, default=42)
    v = sub.add_parser("evaluate")
    v.add_argument("cache", type=Path)
    v.add_argument("--out", required=True, type=Path)
    v.add_argument("--models", nargs="+", choices=NAMES, default=NAMES)
    v.add_argument("--epochs", type=int, default=12)
    v.add_argument("--batch-size", type=int, default=32)
    v.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if args.out.exists():
        p.error("output exists; choose a new path")
    if args.op == "extract":
        if (
            not 0 < args.fraction <= 1
            or args.max_packets is not None
            and args.max_packets < 1
        ):
            p.error("invalid extraction budget")
        extract(args)
        return
    from .evaluate import evaluate
    from threadpoolctl import threadpool_limits

    with np.load(args.cache, allow_pickle=False) as z, threadpool_limits(limits=4):
        meta = json.loads(str(z["meta"]))
        X = {k: z[k] for k in ("conv", "dist", "bins", "style")}
        report = evaluate(
            z["y"],
            z["fold"],
            z["files"],
            X,
            args.models,
            args.out,
            args.seed,
            args.epochs,
            args.batch_size,
        )
    (args.out / "provenance.json").write_text(
        json.dumps(dict(cache=meta, command=sys.argv), indent=2), encoding="utf-8"
    )
    for name, r in report["models"].items():
        print(name, r["metrics"]["support"]["scored"], r["metrics"]["ap"], flush=True)


if __name__ == "__main__":
    main()
