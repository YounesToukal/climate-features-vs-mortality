import numpy as np
import pandas as pd
import pytest

from climate_health.analysis import mixed_label_groups, residual_test, share_between_dates, site_heterogeneity
from climate_health.models import AgePeriodBasis, age_period, boosting
from climate_health.validation import compare, cross_val_predict, score


@pytest.fixture
def records():
    rng = np.random.default_rng(0)
    n = 600
    age = rng.choice([0, 2, 30, 70], n).astype(float)
    years = rng.uniform(1, 15, n)
    male = rng.integers(0, 2, n).astype(float)
    logit = 2.0 - 0.04 * age - 1.5 * (age == 0) * (years > 11)
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int)
    sites = rng.integers(0, 12, n)
    return np.column_stack([age, male, years]), y, sites


def test_score_of_perfect_predictions_is_one():
    y = np.array([0, 1, 1, 0, 1])
    assert score(y, y.astype(float)) == pytest.approx(1.0)


@pytest.mark.parametrize("make_model", [boosting, age_period])
def test_cross_validation_fills_every_record(records, make_model):
    X, y, sites = records
    predictions = cross_val_predict(make_model, X, y, sites, repeats=2)
    assert predictions.shape == (2, len(y))
    assert ((predictions > 0) & (predictions < 1)).all()


def test_comparison_of_a_model_with_itself_is_zero(records):
    X, y, sites = records
    predictions = cross_val_predict(age_period, X, y, sites, repeats=2)
    result = compare(predictions, predictions, y, sites, n_boot=50)
    assert result["difference"] == 0 and result["low"] == 0 and result["high"] == 0


def test_basis_appends_extra_columns(records):
    X, _, _ = records
    with_extra = np.column_stack([X, np.arange(len(X), dtype=float)])
    assert AgePeriodBasis().fit_transform(with_extra).shape[1] == AgePeriodBasis().fit_transform(X).shape[1] + 1


def test_residual_test_finds_a_real_signal_and_ignores_noise(records):
    X, y, _ = records
    rng = np.random.default_rng(1)
    base = np.full(len(y), y.mean())
    assert residual_test(y, base, X[:, [0]]) < 1e-6
    assert residual_test(y, base, rng.normal(size=(len(y), 1))) > 0.001


def test_variable_fixed_by_date_is_fully_explained_by_date():
    frame = pd.DataFrame({"deathdate": ["a", "a", "b", "b"], "rain": [1.0, 1.0, 3.0, 3.0],
                          "ndvi": [0.1, 0.3, 0.2, 0.4]})
    assert share_between_dates(frame, "rain") == pytest.approx(1.0)
    assert share_between_dates(frame, "ndvi") < 1


def test_mixed_label_groups():
    frame = pd.DataFrame({"day": [1, 1, 2, 2, 3], "y": [0, 1, 1, 1, 0]})
    assert mixed_label_groups(frame, "day", "y") == {"groups": 2, "mixed groups": 1, "records in mixed groups": 2}


def test_no_site_effect_when_predictions_match_rates(records):
    _, y, sites = records
    result = site_heterogeneity(y, np.full(len(y), y.mean()), sites)
    assert result["sites"] == 12 and 0 <= result["p"] <= 1
