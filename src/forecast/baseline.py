import dataclasses

import pandas as pd
from statsmodels.iolib.smpickle import load_pickle

from src.forecast.abc import BaseForecaster, ForecastModel


@dataclasses.dataclass
class BaselineForecaster(BaseForecaster):
    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        # Produce quantile forecasts
        print(features)
        for quantile in range(10, 100, 10):
            loaded_model = self.load_model(f"models/model_q{quantile}.pickle")
            features[f"q{quantile}"] = loaded_model.predict(
                features.loc[:, ["valid_datetime", "ref_datetime", "SolarDownwardRadiation", "WindSpeed:100"]].rename(
                    columns={"WindSpeed:100": "WindSpeed"}
                )
            )
        return features

    def load_model(self, predictor_path: str) -> ForecastModel:
        return load_pickle(predictor_path)
