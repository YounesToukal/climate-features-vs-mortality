"""Statistics used to explain why the climate variables carry no extra signal."""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


def share_between_dates(frame, column):
    """Fraction of a variable's variance explained by the date of death alone."""
    within = frame.groupby("deathdate")[column].transform(lambda v: v - v.mean())
    return 1 - within.var() / frame[column].var()


def mixed_label_groups(frame, keys, target):
    """Groups of two or more deaths sharing `keys`, and how many hold both labels."""
    groups = frame.groupby(keys)[target].agg(["size", "mean"])
    groups = groups[groups["size"] > 1]
    mixed = groups[(groups["mean"] > 0) & (groups["mean"] < 1)]
    return {"groups": len(groups), "mixed groups": len(mixed), "records in mixed groups": int(mixed["size"].sum())}


def residual_test(y, base_probability, z):
    """Likelihood-ratio test of extra columns `z` on top of a fixed model.

    The base model enters as an offset, so the test asks whether `z` explains
    anything the base model has not already explained.
    """
    offset = np.log(base_probability / (1 - base_probability))
    intercept = np.ones((len(y), 1))
    null = sm.GLM(y, intercept, family=sm.families.Binomial(), offset=offset).fit()
    full = sm.GLM(y, np.hstack([intercept, z]), family=sm.families.Binomial(), offset=offset).fit()
    return stats.chi2.sf(2 * (full.llf - null.llf), z.shape[1])


def site_heterogeneity(y, probability, sites):
    """Chi-squared test of observed against expected positives, site by site."""
    frame = pd.DataFrame({"site": sites, "observed": y, "expected": probability,
                          "variance": probability * (1 - probability)})
    totals = frame.groupby("site").sum()
    z = (totals["observed"] - totals["expected"]) / np.sqrt(totals["variance"])
    return {"sites": len(z), "chi2": (z**2).sum(), "p": stats.chi2.sf((z**2).sum(), len(z)), "sd of z": z.std()}
