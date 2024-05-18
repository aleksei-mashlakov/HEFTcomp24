import dataclasses
import pickle
from pathlib import Path

import pandas as pd
from mlforecast import MLForecast
from sklearn.preprocessing import MinMaxScaler

from src.forecast.abc import BaseForecaster


@dataclasses.dataclass
class LightGBMForecaster(BaseForecaster):
    model_name: str = "lgbm"
    horizon: int = 48
    target: str = "total_generation_MWh"

    def load_model(self, model_path: Path):
        return MLForecast.load(path=model_path)

    def load_scalers(self, model_path: Path) -> tuple[MinMaxScaler, MinMaxScaler]:
        with open(model_path / "target_scaler.pickle", "rb") as handle:
            target_scaler = pickle.load(handle)

        with open(model_path / "input_scaler.pickle", "rb") as handle:
            input_scaler = pickle.load(handle)
        return input_scaler, target_scaler

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        # Produce quantile forecasts
        model_path = Path(f"./models/{self.model_name}")
        input_scaler, target_scaler = self.load_scalers(model_path)
        predictor = self.load_model(model_path)

        transformed_data = features.drop("ref_datetime", axis=1).set_index("valid_datetime")
        transformed_data = transformed_data[: transformed_data.index.max().replace(hour=22, minute=30, second=0)]
        transformed_data[transformed_data < 0] = 0.0
        scaled_dataset = pd.DataFrame(
            data=input_scaler.transform(transformed_data.reset_index(drop=True)),
            index=transformed_data.index,
            columns=transformed_data.columns,
        )
        scaled_dataset[scaled_dataset < 0] = 0.0
        scaled_dataset.index = scaled_dataset.index.tz_localize(None)
        scaled_dataset.reset_index(names="ds", inplace=True)
        scaled_dataset["unique_id"] = self.target

        old_index = scaled_dataset.ds.copy()
        scaled_dataset.ds = predictor.make_future_dataframe(self.horizon).ds

        quantile_forecasts = predictor.predict(self.horizon, X_df=scaled_dataset)
        quantile_forecasts.set_index("ds", inplace=True)
        quantile_forecasts = quantile_forecasts.drop("unique_id", axis=1)
        for column in quantile_forecasts.columns:
            quantile_forecasts.loc[:, column] = target_scaler.inverse_transform(quantile_forecasts[[column]].values)
        quantile_forecasts.index = old_index[: self.horizon]
        quantile_forecasts.reset_index(names="valid_datetime", inplace=True)
        quantile_forecasts.valid_datetime = pd.to_datetime(quantile_forecasts.valid_datetime, utc=True)
        # market_time = utils.day_ahead_market_times().tz_convert("UTC")
        # quantile_forecasts = quantile_forecasts[quantile_forecasts.index <= market_time.max()]
        return quantile_forecasts
