import abc
import dataclasses
from typing import TypeVar

import pandas as pd


@dataclasses.dataclass
class TradeInput:
    production_forecast: pd.DataFrame
    weather_forecast: pd.DataFrame | None = None
    price_forecast: pd.DataFrame | None = None


@dataclasses.dataclass
class BaseTradeStrategy(abc.ABC):
    @abc.abstractmethod
    def compute_volume(self, input_data: TradeInput) -> pd.DataFrame:
        raise NotImplementedError


TraderT = TypeVar("TraderT", bound=BaseTradeStrategy)


@dataclasses.dataclass
class MeanForecastTradeStrategy(BaseTradeStrategy):
    def compute_volume(self, input_data: TradeInput) -> pd.DataFrame:
        market_bid = input_data.production_forecast.loc[
            :, ["valid_datetime", "q50"]
        ].copy()
        return market_bid.rename(columns={"q50": "market_bid"})
