import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def scale_dataset(dataset: pd.DataFrame) -> tuple[MinMaxScaler, pd.DataFrame]:
    scaler = MinMaxScaler()
    scaled_dataset = pd.DataFrame(
        data=scaler.fit_transform(dataset.reset_index(drop=True)),
        index=dataset.index,
        columns=dataset.columns,
    )
    return scaler, scaled_dataset
