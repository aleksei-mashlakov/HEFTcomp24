import dataclasses

from typing_extensions import Self

from src.apis.handler import MultivariateApiHandler
from src.data.mv_processor import MultivariateWeatherDataProcessor
from src.forecast.lgbm import LightGBMForecaster
from src.pipeline.abc import BasePipeline
from src.trading.strategy import MeanForecastTradeStrategy, TradeInput


@dataclasses.dataclass
class LGBPipeline(
    BasePipeline[
        MultivariateApiHandler,
        LightGBMForecaster,
        MultivariateWeatherDataProcessor,
        MeanForecastTradeStrategy,
    ]
):
    def run(self) -> None:
        # Get latest weather forecasts
        input_data = self.api_handler.collect()
        # Process weather data
        transformed_input_data = self.data_processor.transform(data=input_data)
        # Produce quantile forecasts
        production_forecast = self.forecaster.predict(features=transformed_input_data)
        # Make trading decision
        trader_input = TradeInput(production_forecast=production_forecast)
        bid = self.trader.compute_volume(trader_input)
        # Transform results
        # print(production_forecast)
        submission_data = self.data_processor.submit_transform(forecast=production_forecast, market_bid=bid)
        # Make submission
        self.api_handler.submit(submission_data)

    @classmethod
    def default(cls) -> Self:
        return cls(
            data_processor=MultivariateWeatherDataProcessor(),
            api_handler=MultivariateApiHandler(),
            forecaster=LightGBMForecaster(),
            trader=MeanForecastTradeStrategy(),
        )
