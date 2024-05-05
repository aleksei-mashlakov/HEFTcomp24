import abc
import dataclasses
from typing import Generic

from typing_extensions import Self

from src.apis.handler import ApiHandlerT, BaselineApiHandler, MultivariateApiHandler
from src.data.processor import (
    DataProcessorT,
    MeanWeatherDataProcessor,
    MultivariateWeatherDataProcessor,
)
from src.forecast.model import (
    BaselineForecaster,
    ForecasterT,
    GluonTsSplitForecaster,
    LightGBMForecaster,
)
from src.trading.strategy import MeanForecastTradeStrategy, TradeInput, TraderT


@dataclasses.dataclass
class BasePipeline(Generic[ApiHandlerT, ForecasterT, DataProcessorT, TraderT], abc.ABC):
    data_processor: DataProcessorT
    api_handler: ApiHandlerT
    forecaster: ForecasterT
    trader: TraderT

    @abc.abstractmethod
    def run(self) -> None:
        raise NotImplementedError


@dataclasses.dataclass
class BaselinePipeline(
    BasePipeline[
        BaselineApiHandler,
        BaselineForecaster,
        MeanWeatherDataProcessor,
        MeanForecastTradeStrategy,
    ]
):
    def run(self) -> None:
        # Get latest weather forecasts
        input_data = self.api_handler.collect()
        print(f"{input_data}")
        # Process weather data
        transformed_input_data = self.data_processor.transform(data=input_data)
        # Produce quantile forecasts
        production_forecast = self.forecaster.predict(features=transformed_input_data)
        # Make trading decision
        trader_input = TradeInput(production_forecast=production_forecast)
        bid = self.trader.compute_volume(trader_input)
        # Transform results
        print(production_forecast)
        submission_data = self.data_processor.submit_transform(
            forecast=production_forecast, market_bid=bid
        )
        # Make submission
        self.api_handler.submit(submission_data)

    @classmethod
    def default(cls) -> Self:
        return cls(
            data_processor=MeanWeatherDataProcessor(),
            api_handler=BaselineApiHandler(),
            forecaster=BaselineForecaster(),
            trader=MeanForecastTradeStrategy(),
        )


@dataclasses.dataclass
class GluonTsPipeline(
    BasePipeline[
        BaselineApiHandler,
        GluonTsSplitForecaster,  # GluonTSForecaster,
        MeanWeatherDataProcessor,
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
        print(production_forecast)
        submission_data = self.data_processor.submit_transform(
            forecast=production_forecast, market_bid=bid
        )
        # Make submission
        self.api_handler.submit(submission_data)

    @classmethod
    def default(cls) -> Self:
        return cls(
            data_processor=MeanWeatherDataProcessor(),
            api_handler=BaselineApiHandler(),
            forecaster=GluonTsSplitForecaster(),  # GluonTSForecaster(),
            trader=MeanForecastTradeStrategy(),
        )


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
        print(production_forecast)
        submission_data = self.data_processor.submit_transform(
            forecast=production_forecast, market_bid=bid
        )
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
