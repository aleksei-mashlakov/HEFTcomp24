import dataclasses

import pandas as pd
import xarray as xr

import src.utils as utils
from src.data.abc import BaseDataProcessor, InputData

SOLAR_FEATURES: list[str] = ["SolarDownwardRadiation", "CloudCover", "Temperature"]
WIND_FEATURES: list[str] = [
    "WindSpeed:100",
    "WindDirection:100",
    "RelativeHumidity",
    "Temperature",
    "WindDirection",
    "WindSpeed",
]
REF_HOUR: int = 0


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
        print(f"{latest_wind_data_features}")
        latest_wind_data_features = utils.upsample_frame(latest_wind_data_features, ref_hour=REF_HOUR)
        print(f"{latest_wind_data_features}")
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
        print(latest_forecast_table)
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

    def store(self) -> None:
        ...
