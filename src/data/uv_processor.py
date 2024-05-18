import dataclasses

import pandas as pd
import xarray as xr

import src.utils as utils
from src.data.abc import BaseDataProcessor, InputData
from src.utils import preprocess_with_coord_averaging_pipeline

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

    def store(self) -> None:
        ...
