import abc
import dataclasses
from typing import Generic, TypeVar

import pandas as pd


@dataclasses.dataclass
class BaseForecaster(abc.ABC):
    horizon: int = 48

    @abc.abstractmethod
    def predict(self) -> None:
        raise NotImplementedError


ForecastInputT = TypeVar("ForecastInputT", pd.DataFrame, pd.Series)


class ForecastModel(Generic[ForecastInputT]):
    @abc.abstractmethod
    def predict(self, input: ForecastInputT) -> ForecastInputT:
        raise NotImplementedError
