import pandas as pd


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
