import dataclasses
import datetime
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import split
from gluonts.model.forecast import SampleForecast
from gluonts.model.predictor import Predictor
from gluonts.torch.model.forecast import DistributionForecast
from sklearn.preprocessing import MinMaxScaler

import src.utils as utils
from src.forecast.abc import BaseForecaster


@dataclasses.dataclass
class GluonTSForecaster(BaseForecaster):
    model_name: str = "DeepARwithMixture"

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        model_path = Path(f"./models/{self.model_name}")

        input_scaler, target_scaler = self._load_scalers(model_path)
        predictor = self.load_model(model_path)

        transformed_data = features.drop("ref_datetime", axis=1).set_index("valid_datetime")
        transformed_data["valid_data"] = 1.0

        transformed_data = reindex_dataset(transformed_data)
        transformed_data = transformed_data[: transformed_data.index.max().replace(hour=22, minute=30, second=0)]

        scaled_dataset = pd.DataFrame(
            data=input_scaler.transform(transformed_data.reset_index(drop=True)),
            index=transformed_data.index,
            columns=transformed_data.columns,
        )
        target = "total_generation_MWh"
        market_time = utils.day_ahead_market_times().tz_convert("UTC")
        scaled_dataset = scaled_dataset.interpolate().ffill().bfill()
        scaled_dataset = scaled_dataset[scaled_dataset.index <= market_time.max()]
        scaled_dataset[scaled_dataset < 0] = 0.0
        scaled_dataset[target] = np.nan
        scaled_dataset.index = scaled_dataset.index.tz_localize(None)
        scaled_dataset = scaled_dataset.resample("30min").mean()

        # Convert to the gluonts format and predict the results
        ds_test = PandasDataset(
            scaled_dataset,
            target=target,
            feat_dynamic_real=[
                "WindSpeed:100",
                "WindDirection:100",
                "WindHumidity",
                "WindTemperature",
                "WindDirection",
                "WindSpeed",
                "SolarDownwardRadiation",
                "CloudCover",
                "SolarTemperature",
                "valid_data",
            ],
            past_feat_dynamic_real=[],
            freq="30min",
            assume_sorted=True,
        )
        _, test_template = split(
            ds_test,
            date=pd.Period(market_time.min() - datetime.timedelta(minutes=30), freq="30min"),
        )

        online_test_ds = test_template.generate_instances(
            prediction_length=self.horizon,
            windows=1,
            distance=self.horizon,
        )

        test_forecast = list(predictor.predict(online_test_ds.input))
        quantile_seq = list(np.linspace(0.1, 0.9, 9).round(2))
        quantile_forecasts = pd.concat(
            [quantify(entry, quantile_seq).assign(entry=i) for i, entry in enumerate(test_forecast)]
        ).drop("entry", axis=1)
        new_index = pd.date_range(
            quantile_forecasts.index.min(),
            periods=quantile_forecasts.shape[0],
            freq="30min",
            tz="UTC",
        )
        quantile_forecasts.reset_index(drop=True, inplace=True)
        quantile_forecasts.index = new_index
        quantile_forecasts = quantile_forecasts.astype(float)
        for column in quantile_forecasts.columns:
            quantile_forecasts.loc[:, column] = target_scaler.inverse_transform(quantile_forecasts[[column]].values)
        quantile_forecasts.reset_index(names="valid_datetime", inplace=True)
        return quantile_forecasts

    def load_model(self, model_path: Path):
        return Predictor.deserialize(model_path)

    def save_model(self, predictor, model_path: Path) -> None:
        # model_path = Path(predictor_path)
        if not os.path.isdir(model_path):
            os.mkdir(model_path)
        predictor.serialize(model_path)

    def _load_scalers(self, model_path: Path) -> tuple[MinMaxScaler, MinMaxScaler]:
        with open(model_path / "target_scaler.pickle", "rb") as handle:
            target_scaler = pickle.load(handle)

        with open(model_path / "input_scaler.pickle", "rb") as handle:
            input_scaler = pickle.load(handle)
        return input_scaler, target_scaler


@dataclasses.dataclass
class GluonTsSplitForecaster(GluonTSForecaster):
    solar_model_name: str = "SolarDeepARwithIQN"  # "SolarDeepARwithMixture"
    wind_model_name: str = "WindDeepARwithMixture"

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        dyn_features = [
            "SolarDownwardRadiation",
            "CloudCover",
            "SolarTemperature",
            "valid_data",
        ]
        solar_forecast = self._predict(self.solar_model_name, features, dyn_features, 900)
        dyn_features = [
            "WindSpeed:100",
            "WindDirection:100",
            "WindHumidity",
            "WindTemperature",
            "WindDirection",
            "WindSpeed",
            "valid_data",
        ]

        wind_forecast = self._predict(self.wind_model_name, features, dyn_features, 150)
        quantile_forecasts = solar_forecast + wind_forecast
        quantile_forecasts.reset_index(names="valid_datetime", inplace=True)
        return quantile_forecasts

    def _predict(
        self,
        model_name: str,
        features: pd.DataFrame,
        dyn_features: list[str],
        upper_clip: float,
    ) -> pd.DataFrame:
        # model_name = "SolarDeepARwithMixture"
        model_path = Path(f"./models/{model_name}")
        input_scaler, target_scaler = self._load_scalers(model_path)
        predictor = self.load_model(model_path)

        transformed_data = features.drop("ref_datetime", axis=1).set_index("valid_datetime")
        transformed_data["valid_data"] = 1.0

        transformed_data = reindex_dataset(transformed_data)
        transformed_data = transformed_data[: transformed_data.index.max().replace(hour=22, minute=30, second=0)]

        transformed_data = transformed_data.loc[:, dyn_features]

        scaled_dataset = pd.DataFrame(
            data=input_scaler.transform(transformed_data.reset_index(drop=True)),
            index=transformed_data.index,
            columns=transformed_data.columns,
        )
        target = "total_generation_MWh"
        market_time = utils.day_ahead_market_times().tz_convert("UTC")
        scaled_dataset = scaled_dataset.interpolate().ffill().bfill()
        scaled_dataset = scaled_dataset[scaled_dataset.index <= market_time.max()]
        scaled_dataset[scaled_dataset < 0] = 0.0
        scaled_dataset[target] = np.nan
        scaled_dataset.index = scaled_dataset.index.tz_localize(None)
        scaled_dataset = scaled_dataset.resample("30min").mean()

        # Convert to the gluonts format and predict the results
        ds_test = PandasDataset(
            scaled_dataset,
            target=target,
            feat_dynamic_real=dyn_features,
            past_feat_dynamic_real=[],
            freq="30min",
            assume_sorted=True,
        )
        _, test_template = split(
            ds_test,
            date=pd.Period(market_time.min() - datetime.timedelta(minutes=30), freq="30min"),
        )

        online_test_ds = test_template.generate_instances(
            prediction_length=self.horizon,
            windows=1,
            distance=self.horizon,
        )

        test_forecast = list(predictor.predict(online_test_ds.input))
        quantile_seq = list(np.linspace(0.1, 0.9, 9).round(2))
        quantile_forecasts = pd.concat(
            [quantify(entry, quantile_seq).assign(entry=i) for i, entry in enumerate(test_forecast)]
        ).drop("entry", axis=1)
        new_index = pd.date_range(
            quantile_forecasts.index.min(),
            periods=quantile_forecasts.shape[0],
            freq="30min",
            tz="UTC",
        )
        quantile_forecasts.reset_index(drop=True, inplace=True)
        quantile_forecasts.index = new_index
        quantile_forecasts = quantile_forecasts.astype(float)
        for column in quantile_forecasts.columns:
            quantile_forecasts.loc[:, column] = target_scaler.inverse_transform(quantile_forecasts[[column]].values)
        # quantile_forecasts = quantile_forecasts.clip(upper=upper_clip, lower=0)
        # quantile_forecasts.reset_index(names="valid_datetime", inplace=True)
        return quantile_forecasts


def reindex_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    new_index = pd.date_range(dataset.index.min(), end=dataset.index.max(), freq="30min")
    reindexed_dataset = dataset.reindex(new_index).interpolate().ffill().bfill()
    return reindexed_dataset


def quantify(forecast: SampleForecast | DistributionForecast, quantiles: list[str]):
    quantile_array = forecast.to_quantile_forecast(quantiles=[str(q) for q in quantiles]).forecast_array
    _, h = quantile_array.shape
    dates = pd.date_range(forecast.start_date.to_timestamp(), freq=forecast.freq, periods=h)
    quantile_seq = [f"q{int(q*100):2d}" for q in quantiles]
    return pd.DataFrame(quantile_array.T, index=dates, columns=quantile_seq)


def sample(forecast: SampleForecast | DistributionForecast, num_samples: int = 100):
    if type(forecast) == DistributionForecast:
        sample_array = forecast.to_sample_forecast(num_samples).samples
    elif type(forecast) == SampleForecast:
        sample_array = forecast.samples
    _, h = sample_array.shape
    dates = pd.date_range(forecast.start_date.to_timestamp(), freq=forecast.freq, periods=h)
    return pd.DataFrame(sample_array.T, index=dates)


def sample_iqn(forecast: DistributionForecast):
    sample_array = forecast.distribution.sample()
    _, h = sample_array.shape
    dates = pd.date_range(forecast.start_date.to_timestamp(), freq=forecast.freq, periods=h)
    return pd.DataFrame(sample_array.T, index=dates)
