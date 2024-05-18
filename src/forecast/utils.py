import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class HeftForecastScorer:
    @classmethod
    def pinball(cls, y: pd.Series, q: pd.Series, alpha: float):
        return (y - q) * alpha * (y >= q) + (q - y) * (1 - alpha) * (y < q)

    @classmethod
    def pinball_score(cls, df: pd.DataFrame):
        score = list()
        for qu in range(10, 100, 10):
            score.append(cls.pinball(y=df["target"], q=df[f"q{qu}"], alpha=qu / 100).mean())
        return sum(score) / len(score)


def scale_dataset(dataset: pd.DataFrame) -> tuple[MinMaxScaler, pd.DataFrame]:
    scaler = MinMaxScaler()
    scaled_dataset = pd.DataFrame(
        data=scaler.fit_transform(dataset.reset_index(drop=True)),
        index=dataset.index,
        columns=dataset.columns,
    )
    return scaler, scaled_dataset


def load_parquet_dataset(filename: str) -> pd.DataFrame:
    dataset = pd.read_parquet(filename)
    dataset = dataset.set_index("dtm")
    dataset.index = pd.to_datetime(dataset.index)
    return dataset


def filter_features(dataset: pd.DataFrame, features: list) -> pd.DataFrame:
    return dataset.loc[:, features]


def reindex_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    new_index = pd.date_range(dataset.index[0], end=dataset.index[-1], freq="30min")
    reindexed_dataset = dataset.reindex(new_index).interpolate()
    return reindexed_dataset


def load_dataset() -> pd.DataFrame:
    dataset = pd.read_csv("./data/final/ModellingDataset_2020-09-20_2024-01-14.csv", index_col=0)
    dataset.index = pd.to_datetime(dataset.index)
    return dataset
