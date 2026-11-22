"""The two model families used throughout.

Both take a matrix whose first three columns are age, male and years since
2007; any further columns are extra features, such as climate variables.
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

AGE_GROUPS = [1, 5, 15, 50, 65]  # lower bounds of 1-4, 5-14, 15-49, 50-64 and 65+; 0 is infants
REGIME_YEAR = 11.0  # 2018-01-01, in years since 2007-01-01


def boosting(seed=0):
    """Gradient-boosted trees, kept shallow and heavily regularised for 3 000 records."""
    return HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300, max_depth=3,
                                          min_samples_leaf=80, l2_regularization=10.0,
                                          random_state=seed)


class AgePeriodBasis(TransformerMixin, BaseEstimator):
    """Smooth age and time effects, a time trend per age group, sex per age
    group, and a step at 2018 per age group, followed by any extra columns."""

    def fit(self, X, y=None):
        age, years = X[:, [0]], X[:, [2]]
        group = np.digitize(age, AGE_GROUPS)
        self.age_spline_ = SplineTransformer(n_knots=8, degree=3).fit(np.log1p(age))
        self.time_spline_ = SplineTransformer(n_knots=8, degree=3).fit(years)
        self.groups_ = OneHotEncoder(sparse_output=False).fit(group)
        self.extra_ = StandardScaler().fit(X[:, 3:]) if X.shape[1] > 3 else None
        return self

    def transform(self, X):
        age, male, years = X[:, [0]], X[:, [1]], X[:, [2]]
        group = self.groups_.transform(np.digitize(age, AGE_GROUPS))
        time = self.time_spline_.transform(years)
        late = (years >= REGIME_YEAR).astype(float)
        parts = [self.age_spline_.transform(np.log1p(age)), age == 0,
                 (group[:, :, None] * time[:, None, :]).reshape(len(X), -1),
                 group * male, group * late]
        if self.extra_ is not None:
            parts.append(self.extra_.transform(X[:, 3:]))
        return np.hstack(parts).astype(float)


def age_period(seed=0, C=0.1):
    """A penalised logistic regression on the age-period basis."""
    return make_pipeline(AgePeriodBasis(), LogisticRegression(C=C, max_iter=5000))
