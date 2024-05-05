import datetime

import pandas as pd
import requests


class RebaseAPI:
    challenge_id: str = "heftcom2024"
    base_url: str = "https://api.rebase.energy"

    def __init__(self, api_key=open("team_key.txt").read()):
        self.api_key = api_key
        self.headers = {"Authorization": f"Bearer {api_key}"}
        self.session = requests.Session()
        self.session.headers = self.headers

    def get_submission(self, day: str) -> pd.DataFrame | None:
        url = "https://api.rebase.energy/challenges/heftcom2024/submissions"
        resp = self.session.get(url, params={"market_day": day})
        if resp.status_code != 200:
            print(f"Response status code for variable data is {resp.status_code}")
            return None
        data = resp.json()
        if not data["items"]:
            print(f"No items for {day}")
            return None
        submission = data["items"][-1]["solution"]["submission"]
        df = pd.json_normalize(submission)
        df.columns = df.columns.to_series().replace(
            {"^probabilistic_forecast.": "q", "^timestamp*": "valid_datetime"},
            regex=True,
        )
        return df

    def get_variable(self, day: str, variable: str) -> pd.DataFrame | None:
        if variable not in [
            "market_index",
            "day_ahead_price",
            "imbalance_price",
            "wind_total_production",
            "solar_total_production",
            "solar_and_wind_forecast",
        ]:
            raise Exception(f"No such variable {variable} in API!")
        url = f"{self.base_url}/challenges/data/{variable}"
        resp = self.session.get(url, params={"day": day})
        if resp.status_code != 200:
            print(f"Response status code for variable data is {resp.status_code}")
            return
        data = resp.json()
        df = pd.DataFrame(data)
        if df.empty:
            print(f"No data for {day}")
            return
        return df

    # Solar and wind forecast
    def get_solar_wind_forecast(self, day):
        url = f"{self.base_url}/challenges/data/solar_and_wind_forecast"
        params = {"day": day}
        resp = self.session.get(url, params=params)
        data = resp.json()
        df = pd.DataFrame(data)
        return df

    # Day ahead demand forecast
    def get_day_ahead_demand_forecast(self):
        url = f"{self.base_url}/challenges/data/day_ahead_demand"
        resp = self.session.get(url)
        print(resp)
        return resp.json()

    # Margin forecast
    def get_margin_forecast(self):
        url = f"{self.base_url}/challenges/data/margin_forecast"
        resp = self.session.get(url)
        print(resp)
        return resp.json()

    def query_weather_latest(
        self,
        model,
        lats: list[float],
        lons: list[float],
        variables: str,
        query_type,
    ):
        url = f"{self.base_url}/weather/v2/query"

        body = {
            "model": model,
            "latitude": lats,
            "longitude": lons,
            "variables": variables,
            "type": query_type,
            "output-format": "json",
            "forecast-horizon": "latest",
        }

        resp = requests.post(url, json=body, headers={"Authorization": self.api_key})
        print(f"Response status code for weather data is {resp.status_code}")
        return resp.json()

    def query_weather_latest_points(self, model, lats: list[float], lons: list[float], variables: str) -> pd.DataFrame:
        # Data here is returned a list
        data = self.query_weather_latest(model, lats, lons, variables, "points")

        df = pd.DataFrame()
        for point in range(len(data)):
            new_df = pd.DataFrame(data[point])
            new_df["point"] = point
            new_df["latitude"] = lats[point]
            new_df["longitude"] = lons[point]
            df = pd.concat([df, new_df])

        return df

    def query_weather_latest_grid(self, model, lats, lons, variables) -> pd.DataFrame:
        # Data here is returned as a flattened
        data = self.query_weather_latest(model, lats, lons, variables, "grid")
        df = pd.DataFrame(data)
        return df

    # To query Hornsea project 1 DWD_ICON-EU grid
    def get_hornsea_dwd(self) -> pd.DataFrame:
        # As a 6x6 grid
        lats = [53.77, 53.84, 53.9, 53.97, 54.03, 54.1]
        lons = [1.702, 1.767, 1.832, 1.897, 1.962, 2.027]

        variables = "WindSpeed, WindSpeed:100, WindDirection, WindDirection:100, Temperature, RelativeHumidity"
        return self.query_weather_latest_grid("DWD_ICON-EU", lats, lons, variables)

    # To query Hornsea project 1 GFS grid
    def get_hornsea_gfs(self) -> pd.DataFrame:
        # As a 3x3 grid
        lats = [53.59, 53.84, 54.09]
        lons = [1.522, 1.772, 2.022]

        variables = "WindSpeed, WindSpeed:100, WindDirection, WindDirection:100, Temperature, RelativeHumidity"
        return self.query_weather_latest_grid("NCEP_GFS", lats, lons, variables)

    def get_pes10_nwp(self, model):
        # As a list of points
        lats = [
            52.4872562,
            52.8776682,
            52.1354277,
            52.4880497,
            51.9563696,
            52.2499177,
            52.6416477,
            52.2700912,
            52.1960768,
            52.7082618,
            52.4043468,
            52.0679429,
            52.024023,
            52.7681276,
            51.8750506,
            52.5582373,
            52.4478922,
            52.5214863,
            52.8776682,
            52.0780721,
        ]
        lons = [
            0.4012455,
            0.7906532,
            -0.2640343,
            -0.1267052,
            0.6588173,
            1.3894081,
            1.3509559,
            0.7082557,
            0.1534462,
            0.7302284,
            1.0762977,
            1.1751747,
            0.2962684,
            0.1699257,
            0.9115028,
            0.7137489,
            0.1204872,
            1.5706825,
            1.1916542,
            -0.0113488,
        ]

        variables = "SolarDownwardRadiation, CloudCover, Temperature"
        return self.query_weather_latest_points(model, lats, lons, variables)

    def get_demand_nwp(self, model) -> pd.DataFrame:
        # As list of points
        lats = [51.479, 51.453, 52.449, 53.175, 55.86, 53.875, 54.297]
        lons = [-0.451, -2.6, -1.926, -2.986, -4.264, -0.442, -1.533]

        variables = "Temperature, WindSpeed, WindDirection, TotalPrecipitation, RelativeHumidity"
        return self.query_weather_latest_points(model, lats, lons, variables)

    def submit(self, data) -> None:
        url = f"{self.base_url}/challenges/{self.challenge_id}/submit"

        resp = self.session.post(url, headers=self.headers, json=data)

        print(f"Received response {resp}")

        match resp.status_code:
            case 200:
                print("The submission was successful")
            case _:
                print("The submission failed")

        print(resp.text)

        # Write log file
        text_file = open(f"logs/sub_{pd.Timestamp('today').strftime('%Y%m%d-%H%M%S')}.txt", "w")
        text_file.write(resp.text)
        text_file.close()

    def fetch_submission_data(
        self,
        start: datetime.datetime = datetime.datetime(2024, 2, 1, tzinfo=datetime.timezone.utc),
        end: datetime.datetime = pd.to_datetime("today", utc=True),
    ) -> pd.DataFrame:
        market_days = pd.date_range(
            start=start,
            end=end + pd.Timedelta(1, unit="day"),
            freq=pd.Timedelta(1, unit="day"),
        )
        batch_submission_data: list[pd.DataFrame] = [
            self.get_submission(day=day.strftime("%Y-%m-%d")) for day in market_days
        ]  # type: ignore

        submission_data = pd.concat(batch_submission_data, axis=0).reset_index(drop=True)
        return submission_data
