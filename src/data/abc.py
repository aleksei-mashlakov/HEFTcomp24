import abc
import dataclasses
from enum import Enum
from typing import Any

import pandas as pd

import src.utils as utils


@dataclasses.dataclass
class InputData:
    wind_data: pd.DataFrame
    solar_data: pd.DataFrame


class WeatherModel(Enum):
    DWD_ICON_EU = "DWD_ICON-EU"
    NCEP_GFS = "NCEP_GFS"


@dataclasses.dataclass
class BaseDataProcessor(abc.ABC):
    @abc.abstractmethod
    def store(self) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def transform(self, data: InputData) -> pd.DataFrame:
        ...

    def submit_transform(self, forecast: pd.DataFrame, market_bid: pd.DataFrame) -> dict[str, Any]:
        trading_data = pd.merge(forecast, market_bid, how="inner", on="valid_datetime")
        submission_data = pd.DataFrame({"datetime": utils.day_ahead_market_times()})
        submission_data = submission_data.merge(
            trading_data,
            how="left",
            left_on="datetime",
            right_on="valid_datetime",
        )
        print(submission_data)
        submission_data = utils.prep_submission_in_json_format(submission_data)
        return submission_data
