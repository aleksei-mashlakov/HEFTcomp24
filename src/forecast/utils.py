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