"""Fixed, label-independent v1 feature schema. Undefined ratios remain NaN."""

from collections import defaultdict
import numpy as np
from .features import _header_observations, _ordinal_bins

NAN = float("nan")


def avg(x):
    return float(np.mean(x)) if len(x) else NAN


def stats(x):
    """Mean, population SD, median and maximum, in that order."""
    return [avg(x), np.std(x), np.median(x), np.max(x)] if len(x) else [NAN] * 4


def stage(t, lo, hi):
    return np.clip(((np.asarray(t) - lo) * 3 / max(hi - lo, 1e-9)).astype(int), 0, 2)


def style7(r):
    h = r[5]
    return [
        k in h
        for k in (
            "user-agent",
            "accept",
            "accept-language",
            "accept-encoding",
            "cookie",
            "referer",
        )
    ] + [bool(h and h[0] == "host")]


def http_views(req, rsp, lo, hi):
    bins = np.full((16, 103), np.nan)
    style = np.full(24, np.nan)
    conv = np.full(74, np.nan)
    dist = np.full(2190, np.nan)
    if not req:
        return bins, style, dist, conv
    ordinal = _ordinal_bins(len(req))
    weights = np.bincount(ordinal, minlength=16)
    stages = stage([r[0] for r in req], lo, hi)
    s7 = np.array([style7(r) for r in req])
    names = [r[5] for r in req]
    for j in np.flatnonzero(weights):
        # Ordinal bins are contiguous; avoid materializing an N x 103 observation matrix.
        lo_i, hi_i = np.searchsorted(ordinal, [j, j + 1])
        chunk = names[lo_i:hi_i]
        counts = np.zeros(101)
        for h in chunk:
            presence, order = _header_observations(h)
            counts[list(presence)] += 1
            counts[[11 + i for i in order]] += 1
        bins[j, :101] = counts / len(chunk)
        bins[j, 101] = avg([a != b for a, b in zip(chunk, chunk[1:])])
        bins[j, 102] = len(set(chunk)) / len(chunk)
    for j in range(3):
        if np.any(stages == j):
            style[j * 7 : j * 7 + 7] = s7[stages == j].mean(0)
    for k, (a, b) in enumerate(((0, 1), (1, 2), (0, 2))):
        style[21 + k] = np.abs(
            style[a * 7 : a * 7 + 7] - style[b * 7 : b * 7 + 7]
        ).mean()
    mask = np.isfinite(bins)
    denom = (weights[:, None] * mask).sum(0)
    mean = np.divide(
        np.nansum(bins * weights[:, None], axis=0),
        denom,
        out=np.full(103, np.nan),
        where=denom > 0,
    )
    var = np.divide(
        np.nansum((bins - mean) ** 2 * weights[:, None], axis=0),
        denom,
        out=np.full(103, np.nan),
        where=denom > 0,
    )
    covered = mask.any(0)
    low = np.full(103, np.nan)
    high = low.copy()
    low[covered] = np.nanmin(bins[:, covered], axis=0)
    high[covered] = np.nanmax(bins[:, covered], axis=0)
    dev = [
        np.nanmean(abs(row - mean)) if np.isfinite(row).any() else NAN for row in bins
    ]
    dist = np.r_[
        bins.ravel(),
        low,
        high,
        mean,
        np.sqrt(var),
        (np.nanmax(bins[:, 11:101], axis=0) > 0),
        dev,
        style,
    ]
    # Per status family: compare next request to immediately preceding request on the same flow.
    byflow = defaultdict(list)
    rt = {}
    changes = [[] for _ in range(5)]
    for r in req:
        byflow[r[1]].append(r)
    for fid, rs in byflow.items():
        rt[fid] = np.array([r[0] for r in rs])
    for t, fid, status in rsp:
        if fid not in rt or not 100 <= status < 600:
            continue
        i = np.searchsorted(rt[fid], t, side="right")
        rs = byflow[fid]
        if not 0 < i < len(rs):
            continue
        a, b = rs[i - 1 : i + 1]
        changes[status // 100 - 1].append(
            [
                a[2] != b[2],
                b[2] == "GET",
                b[2] == "POST",
                a[4] != b[4],
                b[4] > a[4],
                a[3] != b[3],
                b[3] > a[3],
                a[5] != b[5],
                set(a[5]) != set(b[5]),
                a[6] != b[6],
            ]
        )
    conv = np.r_[
        style[:21],
        *[np.mean(z, axis=0) if z else np.full(10, np.nan) for z in changes],
        style[21:],
    ]
    return bins, style, dist, conv


def packet_views(p, req, rsp):
    if not p:
        return [NAN] * 71
    t, size, fid, d = map(np.asarray, zip(*p))
    gaps = np.diff(t)
    start = np.r_[0, np.flatnonzero(gaps > 1.0) + 1]
    stop = np.r_[start[1:], len(p)]
    lens = stop - start
    duration = t[stop - 1] - t[start]
    idle = gaps[gaps > 1.0]
    activity = [
        t[-1] - t[0],
        len(start),
        len(t),
        sum(size),
        *stats(lens),
        *stats(duration),
        *stats(idle),
        *stats(abs(np.diff(lens))),
        *stats(abs(np.diff(duration))),
    ]
    n = min(len(p), 128)
    prefix = [r for r in req if r[0] <= t[n - 1]]
    methods = [r[2] for r in prefix]
    h = [r[5] for r in prefix]
    pref = [
        avg(d[:n] > 0),
        avg(np.diff(d[:n]) != 0),
        *stats(size[:n]),
        len(prefix),
        sum(r[0] <= t[n - 1] for r in rsp),
        *[avg([m == k for m in methods]) for k in ("GET", "POST", "HEAD")],
        *stats([r[3] for r in prefix])[:2],
        avg([r[4] > 0 for r in prefix]),
        avg([len(v) for v in h]),
        avg([len(set(v)) for v in h]),
        avg([a != b for a, b in zip(h, h[1:])]),
        avg([a != b for a, b in zip(methods, methods[1:])]),
        avg([a[3] != b[3] for a, b in zip(prefix, prefix[1:])]),
    ]
    ids, inv = np.unique(fid, return_inverse=True)
    inv = inv.ravel()
    count = np.bincount(inv)
    volume = np.bincount(inv, weights=size)
    plus = np.bincount(inv, weights=d > 0)
    first = np.full(len(ids), np.inf)
    last = np.full(len(ids), -np.inf)
    np.minimum.at(first, inv, t)
    np.maximum.at(last, inv, t)
    phase = stage(t, t[0], t[-1])
    concurrent = np.max(
        np.searchsorted(np.sort(first), first, side="right")
        - np.searchsorted(np.sort(last), first, side="left")
    )
    flow = [
        len(ids),
        avg((plus > 0) & (plus < count)),
        avg(count <= 4),
        *stats(last - first),
        *stats(count),
        max(volume) / sum(volume),
        max(count) / len(p),
        avg(d > 0),
        avg(np.diff(d) != 0),
        concurrent,
        *[np.sum(phase == j) / len(t) for j in range(3)],
        *[np.sum(size[phase == j]) / sum(volume) for j in range(3)],
        *[avg(d[phase == j] > 0) for j in range(3)],
        avg(size),
        *[avg([r[6] == k for r in req]) for k in ("browser", "script")],
    ]
    assert (len(activity), len(pref), len(flow)) == (24, 19, 28)
    return activity + pref + flow


def features(packets, requests, responses):
    lo, hi = (packets[0][0], packets[-1][0]) if packets else (0.0, 0.0)
    bins, style, dist, http = http_views(requests, responses, lo, hi)
    return {
        k: np.asarray(v, dtype=np.float32)
        for k, v in dict(
            conv=np.r_[packet_views(packets, requests, responses), http],
            dist=dist,
            bins=bins,
            style=style,
        ).items()
    }
