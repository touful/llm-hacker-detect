"""Paired classical baselines. Only the fixed-width input view varies within each pair."""

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    AdaBoostClassifier,
    RandomForestClassifier,
    GradientBoostingClassifier,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

FAMILIES = ("linear", "gnb", "knn", "rbf", "cart", "rf", "et", "ada", "gbdt", "hgb")
SUITE = tuple(f"{view}-{family}" for family in FAMILIES for view in ("conv", "dist"))


def additional_estimator(family, seed, class_counts):
    """Training counts adapt k and internal calibration folds on small inputs."""
    if family == "cart":
        return DecisionTreeClassifier(
            max_depth=6, min_samples_leaf=5, random_state=seed
        )
    if family == "rf":
        return RandomForestClassifier(
            n_estimators=160,
            min_samples_leaf=3,
            max_features="sqrt",
            random_state=seed,
            n_jobs=4,
        )
    if family == "ada":
        stump = DecisionTreeClassifier(
            max_depth=1, min_samples_leaf=5, random_state=seed
        )
        return AdaBoostClassifier(
            estimator=stump, n_estimators=80, learning_rate=0.5, random_state=seed
        )
    if family == "gbdt":
        return GradientBoostingClassifier(
            n_estimators=80,
            learning_rate=0.05,
            max_depth=2,
            min_samples_leaf=8,
            random_state=seed,
        )
    if family == "gnb":
        return make_pipeline(StandardScaler(), GaussianNB(var_smoothing=1e-6))
    if family == "knn":
        return make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(
                n_neighbors=min(7, int(sum(class_counts))),
                weights="distance",
                algorithm="brute",
            ),
        )
    if family == "rbf":
        # 外部验证折仅用于低误报阈值，概率校准只使用训练样本。
        splits = min(3, int(min(class_counts)))
        if splits < 2:
            raise ValueError(
                "RBF calibration requires at least two training rows per class"
            )
        svc = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0, gamma="scale"))
        return CalibratedClassifierCV(svc, method="sigmoid", cv=splits, ensemble=False)
    raise ValueError(f"unknown classifier family: {family}")
