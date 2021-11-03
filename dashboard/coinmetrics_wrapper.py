import requests
from datetime import datetime
import urllib.parse
import pandas as pd
import numpy as np

class CoinMetricsBase:

    URL_ASSET = "/assets"
    URL_METRICS = "/metrics"
    URL_METRIC_DATA = "/metricdata"
    URL_METRICS_INFO = "/metric_info"
    URL_ASSETS_INFO = "/asset_info"

    @staticmethod
    def raise_exception(r: requests.models.Response):
        j = r.json()

        raise CoinMetricsException(status_code=r.status_code, data=j.get("error", None))


    @staticmethod
    def format_datetime(dt: datetime) -> int:
        dt_ = datetime(dt.year, dt.month, dt.day, 16, 0)
        return dt_.strftime("%Y-%m-%dT%H:%M:%S.000Z")

class CoinMetricsException(Exception):

    def __init__(self, status_code, data=None):

        self.status_code = status_code

        if data:

            self.code = data["code"]
            self.description = data["description"]
            message = f"Status code: {status_code} - Code: {self.code} - Description: {self.description}"

        else:

            self.code = None
            self.status = None
            self.description = None
            message = f"Status code: {status_code}"

        super().__init__(message)

class CoinMetricsWrapper(CoinMetricsBase):

    def __init__(self):

        self.api = "https://community-api.coinmetrics.io/v2"

        self.headers = {}


    def get_available_assets(self)->dict:

        request_url = f"{self.api}{self.URL_ASSET}"

        r = requests.get(request_url, headers=[])

        if r.status_code == 200:

            return r.json()

        else:

            self.raise_exception(r)


    def get_asset_info(self, subset:list=[])->dict:

        if subset:
            metrics = ",".join(subset) if len(subset)>1 else subset[0]

            options = {
                "subset": metrics,
            }

        else:

            options = {}

        encoded_options = encoded_options = urllib.parse.urlencode(options)

        request_url = f"{self.api}{self.URL_ASSETS_INFO}?{encoded_options}"

        r = requests.get(request_url, headers={})

        if r.status_code == 200:

            return r.json()

        else:

            self.raise_exception(r)

    def get_available_metrics(self)->dict:

        request_url = f"{self.api}{self.URL_METRICS}"

        r = requests.get(request_url, headers=[])

        if r.status_code == 200:

            return r.json()

        else:

            self.raise_exception(r)


    def get_metrics_info(self, subset:list=[])->dict:


        if subset:
            metrics = ",".join(subset) if len(subset)>1 else subset[0]

            options = {
                "subset": metrics,
            }

        else:

            options = {}

        encoded_options = encoded_options = urllib.parse.urlencode(options)

        request_url = f"{self.api}{self.URL_METRICS_INFO}?{encoded_options}"

        r = requests.get(request_url, headers={})

        if r.status_code == 200:

            return r.json()

        else:

            self.raise_exception(r)


    def get_asset_metrics_data(
        self, asset:str, metrics:list, start:datetime, end:datetime
    )->pd.DataFrame:


        endpoint = f"{self.URL_ASSET}/{asset}/{self.URL_METRIC_DATA}"

        metrics = ",".join(metrics) if len(metrics)>1 else metrics[0]

        options = {
            "metrics": metrics,
            "start": self.format_datetime(start),
            "end": self.format_datetime(end),
            "time_agg": "day"
        }

        encoded_options = urllib.parse.urlencode(options)

        request_url = f"{self.api}{endpoint}?{encoded_options}"

        r = requests.get(request_url, headers={})

        if r.status_code == 200:
            data = r.json()["metricData"]
            metrics_ = data["metrics"]
            series = data["series"]

            df = pd.DataFrame([
                {"date": serie["time"], **{m: v for m,v in zip(metrics_, serie["values"])}}
                for serie in series
            ])

            df.loc[:, "date"] = pd.to_datetime(df.loc[:, "date"])

            df = df.replace({None: np.nan}).astype({col: float for col in df.columns if col != "date"}).astype({col: float for col in df.columns if col != "date"})

            return df
        else:
            self.raise_exception(r)


