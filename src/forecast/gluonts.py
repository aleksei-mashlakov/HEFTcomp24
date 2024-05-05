import pandas as pd
from gluonts.model.forecast import SampleForecast
from gluonts.torch.model.forecast import DistributionForecast

# from gluonts.mx.model.forecast import DistributionForecast


def quantify(forecast: SampleForecast | DistributionForecast, quantiles: list[str]):
    quantile_array = forecast.to_quantile_forecast(
        quantiles=[str(q) for q in quantiles]
    ).forecast_array
    _, h = quantile_array.shape
    dates = pd.date_range(
        forecast.start_date.to_timestamp(), freq=forecast.freq, periods=h
    )
    quantile_seq = [f"q{int(q*100):2d}" for q in quantiles]
    return pd.DataFrame(quantile_array.T, index=dates, columns=quantile_seq)


def sample(forecast: SampleForecast | DistributionForecast, num_samples: int = 100):
    if type(forecast) == DistributionForecast:
        sample_array = forecast.to_sample_forecast(num_samples).samples
    elif type(forecast) == SampleForecast:
        sample_array = forecast.samples
    _, h = sample_array.shape
    dates = pd.date_range(
        forecast.start_date.to_timestamp(), freq=forecast.freq, periods=h
    )
    return pd.DataFrame(sample_array.T, index=dates)


def sample_iqn(forecast: DistributionForecast):
    sample_array = forecast.distribution.sample()
    _, h = sample_array.shape
    dates = pd.date_range(
        forecast.start_date.to_timestamp(), freq=forecast.freq, periods=h
    )
    return pd.DataFrame(sample_array.T, index=dates)
