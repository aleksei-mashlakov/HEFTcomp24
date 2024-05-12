import dataclasses

import pandas as pd

from src.trading.abc import BaseTradeStrategy, TradeInput


@dataclasses.dataclass
class MeanForecastTradeStrategy(BaseTradeStrategy):
    def compute_volume(self, input_data: TradeInput) -> pd.DataFrame:
        market_bid = input_data.production_forecast.loc[:, ["valid_datetime", "q50"]].copy()
        return market_bid.rename(columns={"q50": "market_bid"})


@dataclasses.dataclass
class DeterministicRuleBasedTradeStrategy(BaseTradeStrategy):
    def compute_volume(self, input_data: TradeInput) -> pd.DataFrame:
        """
        In a one-price system, under the assumption of deterministic market prices,
        the optimal offer for a risk-neutral stochastic power producer is price-inelastic
        and equal to zero volume if the balancing price is higher than the day-ahead
        price, while it is equal to the nominal capacity E if the balancing price is lower
        than the day-ahead price; if such prices are equal, any offer is optimal.

        If DA price and BAL price are correlated:
        1. if (DA price - BAL price) < 0, the optimal bid is 0.
        2. if (DA price - BAL price) > 0, the optimal bid is full capacity.
        3. if (DA price - BAL price) == 0, the power producer is indifferent since any decision on
        ED would yield the same profit in expectation.

        If DA price and BAL price are NOT correlated:

        As the price difference is positive in expectation, the profit is maximized by bidding
        0 at the day-ahead market and placing all the production at the balancing market.

        If DA price and BAL price are negative, then the logic is reversed?

        """
        market_bid = input_data.production_forecast.loc[:, ["valid_datetime", "q50"]].copy()

        return market_bid.rename(columns={"q50": "market_bid"})
