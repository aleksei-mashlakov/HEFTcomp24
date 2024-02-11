import datetime
import warnings

import pandas as pd
import xarray as xr

# Convert nwp data frame to xarray
def weather_df_to_xr(weather_data) -> xr.Dataset:
    weather_data["ref_datetime"] = pd.to_datetime(
        weather_data["ref_datetime"], utc=True
    )
    weather_data["valid_datetime"] = pd.to_datetime(
        weather_data["valid_datetime"], utc=True
    )

    if "point" in weather_data.columns:
        weather_data = weather_data.set_index(
            ["ref_datetime", "valid_datetime", "point"]
        )
    else:
        weather_data = pd.melt(weather_data, id_vars=["ref_datetime", "valid_datetime"])

        weather_data = pd.concat(
            [weather_data, weather_data["variable"].str.split("_", expand=True)], axis=1
        ).drop(["variable", 1, 3], axis=1)

        weather_data.rename(
            columns={0: "variable", 2: "latitude", 4: "longitude"}, inplace=True
        )

        weather_data = weather_data.set_index(
            ["ref_datetime", "valid_datetime", "longitude", "latitude"]
        )
        weather_data = weather_data.pivot(columns="variable", values="value")

    weather_data = weather_data.to_xarray()

    weather_data["ref_datetime"] = pd.DatetimeIndex(
        weather_data["ref_datetime"].values, tz="UTC"
    )
    weather_data["valid_datetime"] = pd.DatetimeIndex(
        weather_data["valid_datetime"].values, tz="UTC"
    )

    return weather_data


def day_ahead_market_times(today_date=pd.to_datetime("today")):
    tomorrow_date = today_date + pd.Timedelta(1, unit="day")
    DA_Market = [
        pd.Timestamp(
            datetime.datetime(
                today_date.year, today_date.month, today_date.day, 23, 0, 0
            ),
            tz="Europe/London",
        ),
        pd.Timestamp(
            datetime.datetime(
                tomorrow_date.year, tomorrow_date.month, tomorrow_date.day, 22, 30, 0
            ),
            tz="Europe/London",
        ),
    ]

    DA_Market = pd.date_range(
        start=DA_Market[0], end=DA_Market[1], freq=pd.Timedelta(30, unit="minute")
    )

    return DA_Market


def prep_submission_in_json_format(
    submission_data, market_day=pd.to_datetime("today") + pd.Timedelta(1, unit="day")
):
    submission = []

    if any(submission_data["market_bid"] < 0):
        submission_data.loc[submission_data["market_bid"] < 0, "market_bid"] = 0
        warnings.warn(
            "Warning...Some market bids were less than 0 and have been set to 0"
        )

    if any(submission_data["market_bid"] > 1800):
        submission_data.loc[submission_data["market_bid"] > 1800, "market_bid"] = 1800
        warnings.warn(
            "Warning...Some market bids were greater than 1800 and have been set to 1800"
        )

    for i in range(len(submission_data.index)):
        submission.append(
            {
                "timestamp": submission_data["datetime"][i].isoformat(),
                "market_bid": submission_data["market_bid"][i],
                "probabilistic_forecast": {
                    10: submission_data["q10"][i],
                    20: submission_data["q20"][i],
                    30: submission_data["q30"][i],
                    40: submission_data["q40"][i],
                    50: submission_data["q50"][i],
                    60: submission_data["q60"][i],
                    70: submission_data["q70"][i],
                    80: submission_data["q80"][i],
                    90: submission_data["q90"][i],
                },
            }
        )

    data = {"market_day": market_day.strftime("%Y-%m-%d"), "submission": submission}

    return data


def preprocess_with_coord_averaging(
    dataset: xr.Dataset,
    features: list[str],
    dims: str | list[str] = ["latitude", "longitude"],
) -> pd.DataFrame:
    return dataset[features].mean(dim=dims).to_dataframe().reset_index()
