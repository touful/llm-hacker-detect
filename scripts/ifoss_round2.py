"""PowerShell from attribution/: & $py -B -m scripts.ifoss_round2 --out runs/ifoss-round2"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import sklearn
from threadpoolctl import threadpool_limits

from agenttrace import study
from .r2_dedup import dedup
from .r2_fusion import fusion_export
from .r2_slr import fit_cal, slr_export
from .r2_report import render


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    out = parser.parse_args().out
    if out.exists():
        raise FileExistsError(out)
    z = study.load(Path("runs/cohort2048.npz"))
    e = study.load(Path("runs/full-session-pilot.npz"))
    dev = np.flatnonzero(np.isin(z["files"], e["files"]))
    ix = np.flatnonzero(~np.isin(z["files"], e["files"]))
    assert len(np.unique(z["files"])) == 5328 and len(ix) == 4795 and len(dev) == 533
    assert all(np.array_equal(z[key][dev], e[key]) for key in ("files", "fold", "y"))
    assert (
        z["meta"]["max_packets"] == 2048
        and e["meta"]["max_packets"] is None
        and z["meta"]["labels_sha256"] == e["meta"]["labels_sha256"]
    )
    files, fold, y = [z[k][ix] for k in ("files", "fold", "y")]
    assert np.array_equal(np.unique(fold), np.arange(5)) and y.sum() == 1536
    out.mkdir(parents=True, exist_ok=False)

    def inputs(idx):
        return {k: z[k][ix[idx]] for k in ("dist", "conv")}

    models = ["dist-hgb", "flags", "behavior-hgb", "joint-hgb"]
    reference = {
        n: study.scores(Path("runs/paper-study/confirmation") / n / "oof.csv", z, ix)
        for n in models[:3]
    }
    values = {"dist": {}, "flags": {}}
    byfold, params = [], {}
    with (
        threadpool_limits(limits=4),
        (out / "validation_scores.csv").open("x", encoding="utf8", newline="") as f,
    ):
        w = csv.writer(f)
        w.writerow(["file", "test_fold", "validation_fold", "y", *models])
        for k in range(5):
            print("fit fold", k, flush=True)
            vf = (k + 1) % 5
            va, te, tr = [
                np.flatnonzero(mask)
                for mask in [fold == vf, fold == k, (fold != k) & (fold != vf)]
            ]
            batch = dict(k=k, val=va, test=te, v={}, t={})
            params[k] = {}
            for n in models:
                model = study.factory(n, seed=42 + k, epochs=12, batch_size=32).fit(
                    inputs(tr), y[tr]
                )
                v, t = model.predict(inputs(va)), model.predict(inputs(te))
                batch["v"][n], batch["t"][n] = v, t
                if n not in reference:
                    continue
                np.testing.assert_allclose(t, reference[n][te], rtol=0, atol=1e-14)
                cal = fit_cal(v, y[va])
                params[k][n] = cal.params
                calibrated = cal(t)
                prefix = {
                    "dist-hgb": "dist",
                    "flags": "flags",
                    "behavior-hgb": "behavior",
                }[n]
                batch["v"][prefix + "_platt"] = cal(v)["platt_post"]
                batch["t"][prefix + "_platt"] = calibrated["platt_post"]
                if prefix in values:
                    for key, val in {"raw": t, **calibrated}.items():
                        values[prefix].setdefault(key, np.full(len(y), np.nan))[te] = (
                            val
                        )
            for j, pos in enumerate(va):
                w.writerow(
                    [
                        files[pos],
                        k,
                        vf,
                        y[pos],
                        *(
                            v if np.isfinite(v) else None
                            for v in [batch["v"][n][j] for n in models]
                        ),
                    ]
                )
            byfold.append(batch)
    study.save(out / "calibrator_parameters.json", params)
    study.save(
        out / "environment.json",
        dict(
            python=sys.version,
            executable=sys.executable,
            numpy=np.__version__,
            sklearn=sklearn.__version__,
            labels_sha256=z["meta"]["labels_sha256"],
            base_seed=42,
        ),
    )
    slr_export(out / "slr-exact", files, fold, y, values)
    fu = out / "fusion-fallback"
    res = fusion_export(fu, files, fold, y, byfold)
    for n, entry in res["models"].items():
        if n != "dist-hgb":
            p = study.scores(fu / n / "oof.csv", z, ix)
            entry["paired_common_ap_minus_dist"] = study.paired(
                y, fold, p, reference["dist-hgb"]
            )
    study.save(fu / "summary.json", res)
    print("dedup retraining", flush=True)
    dedup(z, dev, ix, out / "dedup-retrain")
    with (out / "README_RESULTS.md").open("x", encoding="utf8") as f:
        f.write(render(out))
    print("done", flush=True)


if __name__ == "__main__":
    main()
