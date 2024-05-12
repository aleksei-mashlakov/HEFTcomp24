import abc
import dataclasses
from typing import Generic, TypeVar

from src.apis.handler import BaseApiHandler
from src.data.abc import BaseDataProcessor
from src.forecast.abc import BaseForecaster
from src.trading.abc import BaseTradeStrategy

ApiHandlerT = TypeVar("ApiHandlerT", bound=BaseApiHandler)
ForecasterT = TypeVar("ForecasterT", bound=BaseForecaster)
DataProcessorT = TypeVar("DataProcessorT", bound=BaseDataProcessor)
TraderT = TypeVar("TraderT", bound=BaseTradeStrategy)


@dataclasses.dataclass
class BasePipeline(Generic[ApiHandlerT, ForecasterT, DataProcessorT, TraderT], abc.ABC):
    data_processor: DataProcessorT
    api_handler: ApiHandlerT
    forecaster: ForecasterT
    trader: TraderT

    @abc.abstractmethod
    def run(self) -> None:
        raise NotImplementedError
