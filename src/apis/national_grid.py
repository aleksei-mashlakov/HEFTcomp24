import logging
from datetime import datetime
from typing import Any
from urllib import parse

import requests

logger = logging.getLogger()


class SqlFetcher:
    @classmethod
    def query_data(cls, base_url: str, query: str) -> dict[str, list[Any]]:
        try:
            response = requests.get(
                base_url,
                params=parse.urlencode({"sql": query}),
            )
            data = response.json()["result"]
            logger.info(f"Data: {data}")  # Printing data
        except requests.exceptions.RequestException as e:
            logger.warning(f"{e.response.text}")
            data = {"records": []}
        return data


class NationalGridAPI:
    base_url: str = "https://api.nationalgrideso.com/api/3/action/datastore_search_sql"

    @classmethod
    def carbon_intensity_forecast(
        cls, start_date: datetime, end_date: datetime
    ) -> dict[str, list[Any]]:
        """Forecast national carbon intensity, predicted using machine learning models. Forecast values are given in gCO2/kWh."""
        # gb_carbon_intensity_forecast = f"""SELECT * FROM  "0e5fde43-2de7-4fb4-833d-c7bca3b658b0" WHERE datetime BETWEEN '{start_date}' AND '{end_date}'"""
        gb_carbon_intensity_forecast = f"""SELECT * FROM  "0e5fde43-2de7-4fb4-833d-c7bca3b658b0" WHERE datetime >= '{start_date}' AND datetime < '{end_date}'"""
        return SqlFetcher.query_data(cls.base_url, gb_carbon_intensity_forecast)


class LowCarbonContractAPI:
    base_url: str = "https://dp.lowcarboncontracts.uk/api/3/action/datastore_search_sql"

    @classmethod
    def imrp_price_actuals(
        cls, start_date: datetime, end_date: datetime
    ) -> dict[str, list[Any]]:
        """The actual Intermittent Market Reference Price (IMRP) by date and hourly period
        From: https://dp.lowcarboncontracts.uk/dataset/imrp-actuals/resource/eb894276-ce08-44f9-b485-fd817fd14481?inner_span=True
        """

        imrp_price_query = f"""
        SELECT * from 'eb894276-ce08-44f9-b485-fd817fd14481' WHERE IMRP_Date >= '{start_date}' AND IMRP_Date < '{end_date}'
        """
        return SqlFetcher.query_data(cls.base_url, imrp_price_query)

    @classmethod
    def imrp_price_forecast(
        cls, start_date: datetime, end_date: datetime
    ) -> dict[str, list[Any]]:
        """The forecasted Intermittent Market Reference Price (IMRP) by date and hourly period
        From: https://dp.lowcarboncontracts.uk/dataset/forecast-imrp/resource/0c2a7261-7935-478a-a09f-3d5da177482c
        """
        imrp_price_query = f"""
        SELECT * from '0c2a7261-7935-478a-a09f-3d5da177482c' WHERE Period_Start >= '{start_date}' AND Period_End < '{end_date}'
        """
        return SqlFetcher.query_data(cls.base_url, imrp_price_query)
