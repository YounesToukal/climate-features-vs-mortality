"""Loading the challenge files and the columns derived from them."""

from pathlib import Path

import pandas as pd

TARGET = "is_climate_sensitive"
START = pd.Timestamp("2007-01-01")

# The four columns that came with each record, and the features the organisers
# downloaded from CHIRPS, ERA5-Land, MODIS and SRTM. hot_days_30d is left out:
# it is 0 for every record.
WEATHER = ["avg_temperature", "max_temperature", "min_temperature", "precipitation"]
CLIMATE = ["rain_sum_7d", "rain_sum_30d", "rain_sum_90d", "rain_days_30d", "max_daily_rain_30d",
           "tavg_7d", "tavg_30d", "tavg_90d", "tmax_30d", "tmin_30d", "temp_range_mean_30d",
           "ndvi_30d", "ndvi_90d"]
TERRAIN = ["elevation", "slope"]
LOCATION = ["latitude", "longitude"]
DEMOGRAPHIC = ["age", "male", "years"]


def _prepare(records, climate):
    frame = records.merge(climate.drop(columns="deathdate"), on="ID", how="left", validate="1:1")
    frame["deathdate"] = pd.to_datetime(frame["deathdate"])
    frame["male"] = (frame["gender"] == "Male").astype(int)
    frame["rural"] = (frame["zone"] == "Rural").astype(int)
    frame["years"] = (frame["deathdate"] - START).dt.days / 365.25
    frame["year"] = frame["deathdate"].dt.year
    # A site is one pair of coordinates; the test sites never appear in Train.
    frame["site"] = frame["latitude"].round(6).astype(str) + "," + frame["longitude"].round(6).astype(str)
    return frame


def load(data_dir="data"):
    """Returns Train and Test, each joined to the downloaded climate features."""
    data_dir = Path(data_dir)
    climate = pd.read_csv(data_dir / "climate_features.csv")
    train = _prepare(pd.read_csv(data_dir / "Train.csv"), climate)
    test = _prepare(pd.read_csv(data_dir / "Test.csv"), climate)
    return train, test
