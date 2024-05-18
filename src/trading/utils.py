import pandas as pd


def revenue(
    bid: pd.Series,
    y: pd.Series,
    day_ahead_price: pd.Series,
    single_system_price: pd.Series,
    imbalance_impact: float = 0.07,
) -> pd.Series:
    """Calculates the trading revenue based on the `day_ahead_price` and the `single_system_price`

    Args:
        bid (pd.Series): _description_
        y (pd.Series): _description_
        day_ahead_price (pd.Series): _description_
        single_system_price (pd.Series): _description_
        imbalance_impact (float): _description_

    Returns:
        _type_: _description_
    """
    return bid * day_ahead_price + (y - bid) * (single_system_price - imbalance_impact * (y - bid))
