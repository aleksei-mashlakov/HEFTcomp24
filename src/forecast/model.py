import abc
import dataclasses
import datetime
import os
import pickle
from pathlib import Path
from typing import Generic, TypeVar

import numpy as np
import pandas as pd
from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import split
from gluonts.model.predictor import Predictor
from mlforecast import MLForecast
from sklearn.preprocessing import MinMaxScaler
from statsmodels.iolib.smpickle import load_pickle

import src.utils as utils
from src.forecast.gluonts import quantify


def reindex_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    new_index = pd.date_range(
        dataset.index.min(), end=dataset.index.max(), freq="30min"
    )
    reindexed_dataset = dataset.reindex(new_index).interpolate().ffill().bfill()
    return reindexed_dataset


@dataclasses.dataclass
class BaseForecaster(abc.ABC):
    @abc.abstractmethod
    def predict(self) -> None:
        raise NotImplementedError


ForecasterT = TypeVar("ForecasterT", bound=BaseForecaster)
ForecastInputT = TypeVar("ForecastInputT", pd.DataFrame, pd.Series)


class ForecastModel(Generic[ForecastInputT]):
    def predict(self, ForecastInputT) -> ForecastInputT:
        raise NotImplementedError


@dataclasses.dataclass
class LightGBMForecaster(BaseForecaster):
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
        model_name = "lgb3"
        model_path = Path(f"./models/{model_name}")
        horizon = 48
        target = "total_generation_MWh"
        input_scaler, target_scaler = self.load_scalers(model_path)
        predictor = self.load_model(Path("./models/lgb2/"))

        transformed_data = features.drop("ref_datetime", axis=1).set_index(
            "valid_datetime"
        )
        transformed_data = transformed_data[
            : transformed_data.index.max().replace(hour=22, minute=30, second=0)
        ]
        transformed_data[transformed_data < 0] = 0.0
        scaled_dataset = pd.DataFrame(
            data=input_scaler.transform(transformed_data.reset_index(drop=True)),
            index=transformed_data.index,
            columns=transformed_data.columns,
        )
        market_time = utils.day_ahead_market_times().tz_convert("UTC")
        scaled_dataset[scaled_dataset < 0] = 0.0
        scaled_dataset = scaled_dataset[scaled_dataset.index <= market_time.max()]
        scaled_dataset.index = scaled_dataset.index.tz_localize(None)
        scaled_dataset.reset_index(names="ds", inplace=True)

        scaled_dataset["unique_id"] = target

        old_index = scaled_dataset.ds.copy()
        scaled_dataset.ds = predictor.make_future_dataframe(horizon).ds

        quantile_forecasts = predictor.predict(horizon, X_df=scaled_dataset)
        quantile_forecasts.set_index("ds", inplace=True)
        quantile_forecasts = quantile_forecasts.drop("unique_id", axis=1)
        for column in quantile_forecasts.columns:
            quantile_forecasts.loc[:, column] = target_scaler.inverse_transform(
                quantile_forecasts[[column]].values
            )
        quantile_forecasts.index = old_index
        quantile_forecasts.reset_index(names="valid_datetime", inplace=True)
        quantile_forecasts.valid_datetime = pd.to_datetime(
            quantile_forecasts.valid_datetime, utc=True
        )
        return quantile_forecasts


@dataclasses.dataclass
class BaselineForecaster(BaseForecaster):
    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        # Produce quantile forecasts
        print(features)
        for quantile in range(10, 100, 10):
            loaded_model = self.load_model(f"models/model_q{quantile}.pickle")
            features[f"q{quantile}"] = loaded_model.predict(
                features.loc[
                    :,
                    [
                        "valid_datetime",
                        "ref_datetime",
                        "SolarDownwardRadiation",
                        "WindSpeed:100",
                    ],
                ].rename(columns={"WindSpeed:100": "WindSpeed"})
            )
        return features

    def load_model(self, predictor_path: str) -> ForecastModel:
        return load_pickle(predictor_path)


@dataclasses.dataclass
class GluonTSForecaster(BaseForecaster):
    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        model_name = "DeepARwithMixture"
        model_path = Path(f"./models/{model_name}")

        input_scaler, target_scaler = self._load_scalers(model_path)
        predictor = self.load_model(model_path)

        transformed_data = features.drop("ref_datetime", axis=1).set_index(
            "valid_datetime"
        )
        transformed_data["valid_data"] = 1.0

        transformed_data = reindex_dataset(transformed_data)
        transformed_data = transformed_data[
            : transformed_data.index.max().replace(hour=22, minute=30, second=0)
        ]

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
            date=pd.Period(
                market_time.min() - datetime.timedelta(minutes=30), freq="30min"
            ),
        )

        online_test_ds = test_template.generate_instances(
            prediction_length=48,
            windows=1,
            distance=48,
        )

        test_forecast = list(predictor.predict(online_test_ds.input))
        quantile_seq = list(np.linspace(0.1, 0.9, 9).round(2))
        quantile_forecasts = pd.concat(
            [
                quantify(entry, quantile_seq).assign(entry=i)
                for i, entry in enumerate(test_forecast)
            ]
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
            quantile_forecasts.loc[:, column] = target_scaler.inverse_transform(
                quantile_forecasts[[column]].values
            )
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
    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        # model_name = "SolarDeepARwithMixture"
        model_name = "SolaeDeepARwithIQN"
        dyn_features = [
            "SolarDownwardRadiation",
            "CloudCover",
            "SolarTemperature",
            "valid_data",
        ]
        solar_forecast = self._predict(model_name, features, dyn_features, 900)
        # print(solar_forecast)
        dyn_features = [
            "WindSpeed:100",
            "WindDirection:100",
            "WindHumidity",
            "WindTemperature",
            "WindDirection",
            "WindSpeed",
            "valid_data",
        ]
        model_name = "WindDeepARwithMixture"
        # wind_forecast = self._predict(model_name, features, dyn_features, 150)
        wind_forecast = solar_forecast.copy()
        # print(wind_forecast)
        wind_forecast.loc[:, :] = 425 / 2
        # print(wind_forecast)
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
        print(model_path)
        input_scaler, target_scaler = self._load_scalers(model_path)
        predictor = self.load_model(model_path)

        transformed_data = features.drop("ref_datetime", axis=1).set_index(
            "valid_datetime"
        )
        transformed_data["valid_data"] = 1.0

        transformed_data = reindex_dataset(transformed_data)
        transformed_data = transformed_data[
            : transformed_data.index.max().replace(hour=22, minute=30, second=0)
        ]

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
            date=pd.Period(
                market_time.min() - datetime.timedelta(minutes=30), freq="30min"
            ),
        )

        online_test_ds = test_template.generate_instances(
            prediction_length=48,
            windows=1,
            distance=48,
        )

        test_forecast = list(predictor.predict(online_test_ds.input))
        quantile_seq = list(np.linspace(0.1, 0.9, 9).round(2))
        quantile_forecasts = pd.concat(
            [
                quantify(entry, quantile_seq).assign(entry=i)
                for i, entry in enumerate(test_forecast)
            ]
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
            quantile_forecasts.loc[:, column] = target_scaler.inverse_transform(
                quantile_forecasts[[column]].values
            )
        # quantile_forecasts = quantile_forecasts.clip(upper=upper_clip, lower=0)
        # quantile_forecasts.reset_index(names="valid_datetime", inplace=True)
        return quantile_forecasts
