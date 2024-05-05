import abc
import dataclasses
from typing import Any, TypeVar

from src.apis.rebase import RebaseAPI
from src.data.format import InputData


@dataclasses.dataclass
class BaseApiHandler(abc.ABC):
    @abc.abstractmethod
    def collect() -> InputData:
        raise NotImplementedError

    @abc.abstractmethod
    def submit() -> None:
        raise NotImplementedError


ApiHandlerT = TypeVar("ApiHandlerT", bound=BaseApiHandler)


@dataclasses.dataclass
class BaselineApiHandler(BaseApiHandler):
    comp_api: RebaseAPI = RebaseAPI()

    def collect(self) -> InputData:
        wind_data = self.comp_api.get_hornsea_dwd()
        solar_data = self.comp_api.get_pes10_nwp("DWD_ICON-EU")
        return InputData(wind_data=wind_data, solar_data=solar_data)

    def submit(self, data: dict[str, Any]) -> None:
        self.comp_api.submit(data)


@dataclasses.dataclass
class MultivariateApiHandler(BaseApiHandler):
    comp_api: RebaseAPI = RebaseAPI()

    def collect(self) -> InputData:
        wind_data = self.comp_api.get_hornsea_gfs()
        solar_data = self.comp_api.get_pes10_nwp("DWD_ICON-EU")
        return InputData(wind_data=wind_data, solar_data=solar_data)

    def submit(self, data: dict[str, Any]) -> None:
        self.comp_api.submit(data)
