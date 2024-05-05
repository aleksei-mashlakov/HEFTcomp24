import abc
import dataclasses
from typing import Any, TypeVar

import pandas as pd
import xarray as xr

import src.utils as utils
from src.data.format import InputData
from src.utils import preprocess_with_coord_averaging_pipeline

SOLAR_FEATURES = ["SolarDownwardRadiation", "CloudCover", "Temperature"]
WIND_FEATURES = ["WindSpeed:100", "WindDirection:100", "RelativeHumidity", "Temperature", "WindDirection", "WindSpeed"]

REF_HOUR = 0


@dataclasses.dataclass
class BaseDataProcessor(abc.ABC):
    @abc.abstractmethod
    def store(self) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def transform(self, data: InputData) -> pd.DataFrame: ...

    def submit_transform(self, forecast: pd.DataFrame, market_bid: pd.DataFrame) -> dict[str, Any]:
        trading_data = pd.merge(forecast, market_bid, how="inner", on="valid_datetime")
        submission_data = pd.DataFrame({"datetime": utils.day_ahead_market_times()})
        submission_data = submission_data.merge(
            trading_data,
            how="left",
            left_on="datetime",
            right_on="valid_datetime",
        )

        # print(f"{submission_data=}")
        submission_data = utils.prep_submission_in_json_format(submission_data)
        return submission_data


DataProcessorT = TypeVar("DataProcessorT", bound=BaseDataProcessor)


@dataclasses.dataclass
class MeanWeatherDataProcessor(BaseDataProcessor):
    def _transform_solar_data(self, data) -> xr.Dataset:
        latest_dwd_solar = utils.weather_df_to_xr(data)

        # datetime = pd.to_datetime("today").strftime("%Y-%m-%d")
        # latest_dwd_solar.to_netcdf(f"./data/raw/online/dwd_solar_{datetime}.nc")

        latest_solar_features = preprocess_with_coord_averaging_pipeline(
            dataset=latest_dwd_solar, dims="point", features=SOLAR_FEATURES
        )
        latest_solar_features.rename(columns={"Temperature": "SolarTemperature"}, inplace=True)
        return latest_solar_features

    def _transform_wind_data(self, data: pd.DataFrame) -> xr.Dataset:
        latest_wind_data = utils.weather_df_to_xr(data)
        # datetime = pd.to_datetime("today").strftime("%Y-%m-%d")
        # latest_wind_data.to_netcdf(f"./data/raw/online/dwd_wind_{datetime}.nc")
        latest_wind_data_features = preprocess_with_coord_averaging_pipeline(
            dataset=latest_wind_data,
            dims=["latitude", "longitude"],
            features=WIND_FEATURES,
        )

        latest_wind_data_features.rename(
            columns={
                "RelativeHumidity": "WindHumidity",
                "Temperature": "WindTemperature",
            },
            inplace=True,
        )
        return latest_wind_data_features

    def transform(self, data: InputData) -> pd.DataFrame:
        wind_data = self._transform_wind_data(data.wind_data)
        solar_data = self._transform_solar_data(data.solar_data)

        latest_forecast_table = wind_data.merge(
            solar_data,
            how="outer",
            on=["ref_datetime", "valid_datetime"],
        )
        latest_forecast_table = (
            latest_forecast_table.set_index("valid_datetime")
            .resample("30min")
            # .interpolate("linear", limit=5)
            .ffill()
            .bfill()
            .reset_index()
        )
        datetime = pd.to_datetime("today").strftime("%Y-%m-%d")
        latest_forecast_table.to_csv(f"./data/raw/online/latest_forecast_table_{datetime}.csv", index=False)
        return latest_forecast_table

    def store(self) -> None: ...


@dataclasses.dataclass
class MultivariateWeatherDataProcessor(BaseDataProcessor):
    def _transform_solar_data(self, data) -> xr.Dataset:
        names_to_replace = {
            "Temperature": "SolarTemperature",
            "CloudCover": "SolarCloudCover",
        }
        features = ["SolarDownwardRadiation", "SolarCloudCover", "SolarTemperature"]

        latest_dwd_solar = utils.weather_df_to_xr(data)
        latest_dwd_solar = utils.convert_xr_to_pl(latest_dwd_solar).rename(names_to_replace)
        latest_solar_features = utils.convert_grid_data_to_ts(
            latest_dwd_solar, features=features, partition_by=["point"]
        )

        latest_solar_features = utils.upsample_frame(latest_solar_features, ref_hour=REF_HOUR)
        # datetime = pd.to_datetime("today").strftime("%Y-%m-%d")
        # latest_dwd_solar.to_netcdf(f"./data/raw/online/dwd_solar_{datetime}.nc")

        return latest_solar_features

    def _transform_wind_data(self, data: pd.DataFrame) -> xr.Dataset:
        features = [
            "WindSpeed_100",
            "WindDirection_100",
            "WindHumidity",
            "WindTemperature",
            "WindDirection",
            "WindSpeed",
        ]

        names_to_replace = {
            "RelativeHumidity": "WindHumidity",
            "Temperature": "WindTemperature",
            "WindSpeed:100": "WindSpeed_100",
            "WindDirection:100": "WindDirection_100",
        }

        latest_wind_data = utils.weather_df_to_xr(data)
        latest_wind_data = utils.convert_xr_to_pl(latest_wind_data).rename(names_to_replace)
        latest_wind_data_features = utils.convert_grid_data_to_ts(
            latest_wind_data, features=features, partition_by=["latitude", "longitude"]
        )
        print(latest_wind_data_features)
        latest_wind_data_features = utils.upsample_frame(latest_wind_data_features, ref_hour=REF_HOUR)
        print(latest_wind_data_features)
        # datetime = pd.to_datetime("today").strftime("%Y-%m-%d")
        # latest_wind_data.to_netcdf(f"./data/raw/online/dwd_wind_{datetime}.nc")
        return latest_wind_data_features

    def transform(self, data: InputData) -> pd.DataFrame:
        wind_data = self._transform_wind_data(data.wind_data).to_pandas()
        solar_data = self._transform_solar_data(data.solar_data).to_pandas()

        latest_forecast_table = wind_data.merge(
            solar_data,
            how="inner",
            on=["ref_datetime", "valid_datetime"],
        )
        latest_forecast_table = (
            latest_forecast_table.set_index("valid_datetime")
            .resample("30min")
            # .interpolate("linear", limit=5)
            .ffill()
            .bfill()
            .reset_index()
        )
        datetime = pd.to_datetime("today").strftime("%Y-%m-%d")
        latest_forecast_table.to_csv(f"./data/raw/online/latest_forecast_table_{datetime}.csv", index=False)
        return latest_forecast_table

    def store(self) -> None: ...
