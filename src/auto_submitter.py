import pandas as pd
from prefect import flow
from statsmodels.iolib.smpickle import load_pickle

import src.utils as utils
from src.apis.rebase import RebaseAPI


@flow(log_prints=True)
def main() -> None:
    rebase_api = RebaseAPI()

    # Get latest weather forecasts
    hornsea1_weather_data = rebase_api.get_hornsea_dwd()
    latest_dwd_Hornsea1 = utils.weather_df_to_xr(hornsea1_weather_data)
    latest_dwd_Hornsea1_features = (
        latest_dwd_Hornsea1["WindSpeed:100"]
        .mean(dim=["latitude", "longitude"])
        .to_dataframe()
        .reset_index()
    )

    solar_weather_data = rebase_api.get_pes10_nwp("DWD_ICON-EU")
    latest_dwd_solar = utils.weather_df_to_xr(solar_weather_data)
    latest_dwd_solar_features = (
        latest_dwd_solar["SolarDownwardRadiation"]
        .mean(dim="point")
        .to_dataframe()
        .reset_index()
    )

    latest_forecast_table = latest_dwd_Hornsea1_features.merge(
        latest_dwd_solar_features, how="outer", on=["ref_datetime", "valid_datetime"]
    )
    latest_forecast_table = (
        latest_forecast_table.set_index("valid_datetime")
        .resample("30min")
        .interpolate("linear", limit=5)
        .reset_index()
    )
    latest_forecast_table.rename(columns={"WindSpeed:100": "WindSpeed"}, inplace=True)

    # Produce quantile forecasts
    for quantile in range(10, 100, 10):
        loaded_model = load_pickle(f"models/model_q{quantile}.pickle")
        latest_forecast_table[f"q{quantile}"] = loaded_model.predict(
            latest_forecast_table
        )

    # Make submission
    submission_data = pd.DataFrame({"datetime": utils.day_ahead_market_times()})
    submission_data = submission_data.merge(
        latest_forecast_table, how="left", left_on="datetime", right_on="valid_datetime"
    )
    submission_data["market_bid"] = submission_data["q50"]

    submission_data = utils.prep_submission_in_json_format(submission_data)
    print(submission_data)

    rebase_api.submit(submission_data)


if __name__ == "__main__":
    main.serve(name="submission_pipeline", cron="0 9 * jan-may *")
