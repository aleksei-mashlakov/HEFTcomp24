import datetime
import warnings

import pandas as pd
import polars as pl
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


def preprocess_with_coord_averaging_pipeline(
    dataset: xr.Dataset,
    features: list[str],
    dims: str | list[str] = ["latitude", "longitude"],
) -> pd.DataFrame:
    return dataset[features].mean(dim=dims).to_dataframe().reset_index()


def preprocess_with_coord_averaging(
    dataset: xr.Dataset,
    features: list[str],
    dims: str | list[str] = ["latitude", "longitude"],
) -> pd.DataFrame:
    dataset_features = dataset[features].mean(dim=dims).to_dataframe().reset_index()
    # fix naming in the dwd_icon_eu_hornsea_1_20231027_20240108.nc file
    dataset_features.rename(
        columns={"valid_time": "valid_datetime", "reference_time": "ref_datetime"},
        inplace=True,
    )
    dataset_features["ref_datetime"] = dataset_features["ref_datetime"].dt.tz_localize(
        "UTC"
    )
    dataset_features["valid_datetime"] = dataset_features[
        "ref_datetime"
    ] + pd.TimedeltaIndex(dataset_features["valid_datetime"], unit="h")
    return dataset_features


def convert_xr_to_pl(dataarray: xr.DataArray):
    df = (
        dataarray.to_dataframe()
        .reset_index()
        .rename(
            columns={"valid_time": "valid_datetime", "reference_time": "ref_datetime"}
        )
    )
    # df["ref_datetime"] = df["ref_datetime"].dt.tz_localize("UTC")
    # df["valid_datetime"] = df["ref_datetime"] + pd.TimedeltaIndex(
    #     df["valid_datetime"], unit="h"
    # )
    return pl.DataFrame(df).with_columns(
        pl.exclude(["valid_datetime", "ref_datetime"]).cast(pl.Float32)
    )


def convert_grid_data_to_ts(
    df: pl.DataFrame, features: list[str], partition_by: list[str]
) -> pl.DataFrame:
    grid_df_list = df.partition_by(partition_by, maintain_order=True)
    grid_df = []
    for i, df in enumerate(grid_df_list):
        grid_df.append(
            df.drop(partition_by)
            .select(
                [
                    pl.col(["ref_datetime", "valid_datetime"]),
                    pl.col(features).name.suffix(f"_{i}"),
                ]
            )
            .sort(by="ref_datetime")
        )

    return pl.concat(grid_df, how="align").sort("ref_datetime")


def upsample_frame(df: pl.DataFrame, ref_hour: int = 0) -> pl.DataFrame:
    df = (
        df.with_columns(
            (pl.col("valid_datetime") - pl.col("ref_datetime")).alias("delta_datetime"),
        )
        .filter(
            (pl.col("ref_datetime").dt.hour() == ref_hour)
            & (pl.col("delta_datetime") >= datetime.timedelta(hours=23))
            & (pl.col("delta_datetime") < datetime.timedelta(hours=48))
        )
        .sort(by=["ref_datetime", "valid_datetime"])
        .set_sorted(["ref_datetime", "valid_datetime"])
        .upsample("valid_datetime", every="30m", maintain_order=True)
        .with_columns(
            [
                pl.col("ref_datetime").forward_fill(),
                pl.exclude(["valid_datetime", "ref_datetime"]).interpolate(),
            ]
        )
        .drop("delta_datetime")
        .unique(subset=["valid_datetime", "ref_datetime"], keep="last")
        .sort(by=["ref_datetime", "valid_datetime"])
    )
    return df


def preprocess_grid_ts_data(
    dfs: list[xr.Dataset],
    features: list[str],
    names_to_replace: dict[str, str],
    partition_by: list[str] = ["latitude", "longitude"],
) -> pl.DataFrame:
    dfs = [convert_xr_to_pl(df).rename(names_to_replace) for df in dfs]
    df_grid_ts_features = pl.concat(dfs)
    df_grid_ts_features = convert_grid_data_to_ts(
        df_grid_ts_features, features=features, partition_by=partition_by
    )
    return upsample_frame(df_grid_ts_features)


def preprocess_grid_ts_data_separate(
    dfs: list[xr.Dataset],
    features: list[str],
    names_to_replace: dict[str, str],
    partition_by: list[str] = ["latitude", "longitude"],
) -> pl.DataFrame:
    dfs = [convert_xr_to_pl(df).rename(names_to_replace) for df in dfs]
    dfs_with_same_columns = [df[dfs[0].columns] for df in dfs]
    dfs = [
        convert_grid_data_to_ts(df, features=features, partition_by=partition_by)
        for df in dfs_with_same_columns
    ]
    dfs = [upsample_frame(df) for df in dfs]
    return pl.concat(dfs, how="vertical_relaxed").sort(
        by=["valid_datetime", "ref_datetime"]
    )


def preprocess_average_ts_data(
    dfs: list[xr.Dataset],
    features: list[str],
    names_to_replace: dict[str, str],
    dims: list[str] = ["latitude", "longitude"],
) -> pl.DataFrame:
    df_average_features = (
        pd.concat(
            [
                preprocess_with_coord_averaging(df, features=features, dims=dims)
                for df in dfs
            ],
            axis=0,
        )
        .drop_duplicates()
        .reset_index(drop=True)
        .rename(columns=names_to_replace)
    )
    return upsample_frame(pl.DataFrame(df_average_features))
