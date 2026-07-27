"""SVM construction."""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def build_model() -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("svm", SVC(kernel="rbf", C=10.0, gamma="scale", class_weight="balanced", probability=True, random_state=42)),
        ]
    )
