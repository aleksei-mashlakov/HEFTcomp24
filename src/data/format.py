import dataclasses

import pandas as pd


@dataclasses.dataclass
class InputData:
    wind_data: pd.DataFrame
    solar_data: pd.DataFrame
