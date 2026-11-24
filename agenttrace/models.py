"""Binary session classifiers. Torch is only needed to fit sequence models."""

from operator import index

import numpy as np


from .classical import SUITE

DEFAULT_NAMES = (
    "conv-hgb",
    "dist-hgb",
    "dist-et",
    "header-mil",
    "header-tcn",
    "dist-linear",
    "joint-hgb",
    "header-gru",
    "behavior-hgb",
    "bag-hgb",
    "view-fusion",
)
NAMES = tuple(dict.fromkeys((*DEFAULT_NAMES, *SUITE)))
_SHAPES = {"conv": (145,), "dist": (2190,), "bins": (16, 103), "style": (24,)}


def _view(X, key):
    a = np.asarray(X[key], dtype=np.float32)
    if a.ndim != 1 + len(_SHAPES[key]) or a.shape[1:] != _SHAPES[key]:
        raise ValueError(f"{key}: expected (N, {_SHAPES[key]}), got {a.shape}")
    return a


def _tabular(name, seed, counts):
    if name.endswith("-hgb"):
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(
            max_iter=80,
            max_leaf_nodes=15,
            min_samples_leaf=20,
            learning_rate=0.05,
            l2_regularization=1.0,
            random_state=seed,
            early_stopping=False,
        )
    if name.endswith("-et"):
        from sklearn.ensemble import ExtraTreesClassifier

        return ExtraTreesClassifier(
            n_estimators=96,
            min_samples_leaf=5,
            max_features=0.8,
            random_state=seed,
            n_jobs=4,
        )
    if not name.endswith("-linear"):
        from .classical import additional_estimator

        return additional_estimator(name.split("-")[-1], seed, counts)
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, solver="liblinear", max_iter=500, random_state=seed),
    )


def _network(name):
    from .networks import HeaderNet

    return HeaderNet(name)


class _Model:
    def __init__(self, name, seed, epochs, batch_size):
        self.name = name
        self.seed = index(seed)
        self.epochs = index(epochs)
        self.batch_size = index(batch_size)
        if self.epochs < 1 or not 1 <= self.batch_size <= 32:
            raise ValueError("epochs must be positive and batch_size in [1, 32]")
        self._sequence = name.startswith("header-")
        self._fitted = False

    def _inputs(self, X):
        if self._sequence:
            bins, style = _view(X, "bins"), _view(X, "style")
            if len(bins) != len(style):
                raise ValueError("bins and style must have equal row counts")
            keep = np.isfinite(bins).any(axis=(1, 2))
            return (bins, style), keep
        if self.name == "joint-hgb":
            a = np.concatenate((_view(X, "conv"), _view(X, "dist")), axis=1)
        elif self.name == "bag-hgb":
            # Discard bin identity and temporal drift; keep symmetric summaries.
            a = _view(X, "dist")[:, 1648:2150]
        elif self.name == "behavior-hgb":
            v = _view(X, "conv")
            a = np.c_[v[:, :30], v[:, 43:69]]
        else:
            a = _view(X, self.name.split("-")[0])
        a = np.where(np.isfinite(a), a, np.nan)
        return (a,), np.isfinite(a).any(axis=1)

    def fit(self, X, y):
        """Fit on supported rows only; a fold without both classes abstains."""
        arrays, keep = self._inputs(X)
        y = np.asarray(y)
        if y.shape != keep.shape or not np.isin(y, (0, 1)).all():
            raise ValueError("y must be a length-N vector of binary labels")
        self._fitted = False
        self.estimator_ = None
        if not self._sequence:
            a = arrays[0]
            self.cols_ = np.isfinite(a).any(0)
            a = a[:, self.cols_]
            self.median_ = (
                None if self.name.endswith("hgb") else np.nanmedian(a, axis=0)
            )
            arrays = (self._transform(a),)
        ids = np.flatnonzero(keep)
        if np.unique(y[ids]).size == 2:
            if self._sequence:
                self.estimator_ = self._fit_network(arrays, ids, y)
            else:
                counts = np.bincount(y[ids].astype(int), minlength=2)
                if self.name.endswith("-rbf") and counts.min() < 2:
                    self._fitted = True
                    return self
                self.estimator_ = _tabular(self.name, self.seed, counts).fit(
                    arrays[0][ids], y[ids]
                )
        self._fitted = True
        return self

    def _transform(self, a):
        return a if self.median_ is None else np.where(np.isfinite(a), a, self.median_)

    def _fit_network(self, arrays, ids, y):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("header models require cuda:0")
        rng = np.random.default_rng(self.seed)
        # Construct on CPU with a scoped seed; forward passes use no random layers.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(self.seed)
            net = _network(self.name).to("cuda:0")
        optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3)
        criterion = torch.nn.BCEWithLogitsLoss()
        for _ in range(self.epochs):
            shuffled = rng.permutation(ids)
            for i in range(0, len(ids), self.batch_size):
                ix = shuffled[i : i + self.batch_size]
                inp = [torch.as_tensor(a[ix], device="cuda:0") for a in arrays]
                target = torch.as_tensor(y[ix], device="cuda:0", dtype=torch.float32)
                optimizer.zero_grad(set_to_none=True)
                criterion(net(*inp), target).backward()
                optimizer.step()
        return net.eval()

    def predict(self, X):
        """Return P(y=1) in input order, preserving unsupported rows as NaN."""
        if not self._fitted:
            raise RuntimeError("fit must be called before predict")
        arrays, keep = self._inputs(X)
        out = np.full(len(keep), np.nan)
        if self.estimator_ is None or not keep.any():
            return out
        if not self._sequence:
            a = arrays[0][:, self.cols_]
            ids = np.flatnonzero(np.isfinite(a).any(1))
            if len(ids):
                out[ids] = self.estimator_.predict_proba(self._transform(a[ids]))[:, 1]
            return out
        import torch

        ids = np.flatnonzero(keep)
        with torch.inference_mode():
            for i in range(0, len(ids), self.batch_size):
                ix = ids[i : i + self.batch_size]
                inp = [torch.as_tensor(a[ix], device="cuda:0") for a in arrays]
                out[ix] = torch.sigmoid(self.estimator_(*inp)).cpu().numpy()
        return out


class _ViewFusion:
    """Late fusion: pre-specified equal weights; a missing modality abstains."""

    def __init__(self, seed, epochs, batch_size):
        self.members = [
            _Model(name, seed, epochs, batch_size)
            for name in ("behavior-hgb", "dist-hgb")
        ]

    def fit(self, X, y):
        for model in self.members:
            model.fit(X, y)
        return self

    def predict(self, X):
        p = np.stack([model.predict(X) for model in self.members])
        n = np.isfinite(p).sum(0)
        return np.divide(
            np.nansum(p, axis=0), n, out=np.full(len(n), np.nan), where=n > 0
        )


def create(name, seed=42, epochs=12, batch_size=32):
    """Create an unfitted classifier; see docs/models.md for feature support."""
    if name not in NAMES:
        raise ValueError(f"unknown model {name!r}; expected one of {NAMES}")
    if name == "view-fusion":
        return _ViewFusion(seed, epochs, batch_size)
    return _Model(name, seed, epochs, batch_size)
