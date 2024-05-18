import abc
import dataclasses

import pandas as pd


@dataclasses.dataclass
class TradeInput:
    production_forecast: pd.DataFrame
    weather_forecast: pd.DataFrame | None = None
    da_price_forecast: pd.DataFrame | None = None
    imbalance_price_forecast: pd.DataFrame | None = None


@dataclasses.dataclass
class BaseTradeStrategy(abc.ABC):
    @abc.abstractmethod
    def compute_volume(self, input_data: TradeInput) -> pd.DataFrame:
        raise NotImplementedError
