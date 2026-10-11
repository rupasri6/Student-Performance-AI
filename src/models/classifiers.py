"""Classifier adapters used by serialized model pipelines."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone


class EncodedLabelClassifier(ClassifierMixin, BaseEstimator):
    """Adapt estimators that require integer labels while preserving public labels."""

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        self.classes_ = np.unique(np.asarray(y, dtype=str))
        self._label_to_index = {label: index for index, label in enumerate(self.classes_)}
        encoded = np.asarray([self._label_to_index[str(label)] for label in y], dtype=int)
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, encoded)
        return self

    def predict(self, X):
        encoded = np.asarray(self.estimator_.predict(X), dtype=int)
        return self.classes_[encoded]

    def predict_proba(self, X):
        probabilities = np.asarray(self.estimator_.predict_proba(X))
        estimator_codes = np.asarray(self.estimator_.classes_, dtype=int)
        order = np.argsort(estimator_codes)
        return probabilities[:, order]
