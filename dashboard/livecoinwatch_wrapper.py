"""
Wrapper para Live Coin Watch API
"""

from datetime import datetime
from urllib.parse import urljoin
import requests
import pandas as pd
import json

class LiveCoinWatchBase:

    URL_STATUS = "/status"
    URL_CREDITS = "/credits"
    URL_OVERVIEW = "/overview"
    URL_HISTORICAL_OVERVIEW = URL_OVERVIEW + "/history"
    URL_COINS = "/coins"
    URL_COINS_SINGLE = URL_COINS + "/single"
    URL_COINS_SINGLE_HISTORY = URL_COINS_SINGLE + "/history"
    URL_COINS_LIST = URL_COINS + "/list"
    URL_FIATS = "/fiats"
    URL_ALL_FIATS = URL_FIATS + "/all"
    URL_EXCHANGES = "/exchanges"
    URL_EXCHANGES_SINGLE = URL_EXCHANGES + "/single"
    URL_EXCHANGES_LIST = URL_EXCHANGES + "/list"

    @staticmethod
    def raise_exception(r: requests.models.Response):
        j = r.json()

        raise LiveCoinWatchException(status_code=r.status_code, data=j.get("error", None))


    @staticmethod
    def format_datetime(dt: datetime) -> int:
        dt_ = datetime(dt.year, dt.month, dt.day, 16, 0)
        return int(dt_.timestamp()*1000)

class LiveCoinWatchException(Exception):

    def __init__(self, status_code, data=None):

        self.status_code = status_code

        if data:

            self.code = data["code"]
            self.status = data["status"]
            self.description = data["description"]

            message = f"Status code: {status_code} - Code: {self.code} - Status: {self.status} - Description: {self.description}"

        else:
            self.code = None
            self.status = None
            self.description = None
            message = f"Status code: {status_code}"

        super().__init__(message)

class LiveCoinWatchWrapper(LiveCoinWatchBase):

    def __init__(self, api_key:str):

        self.api = "https://api.livecoinwatch.com"

        self.headers = {
            "content-type": "application/json", "x-api-key": api_key
        }

    def test_api(self,):
        """
        Tests the API status.
        """

        url = urljoin(self.api, self.URL_STATUS)

        r = requests.post(url, headers={"content-type": self.headers["content-type"]})

        if r.status_code == 200:
            print("The API is working correctly")
        else:
            self.raise_exception(r)

    def api_key_info(self):
        """
        Find out your API key related information.
        """
        url = urljoin(self.api, self.URL_CREDITS)

        r = requests.post(url, headers=self.headers)

        if r.status_code == 200:

            return r.json()

        else:

            self.raise_exception(r)

    def current_market_overview(self, currency:str)->dict:
        """
        Get current aggregated data for all coins.

        Parameters
        ----------
        currency: str
            Any valid coin or fiat code.

        Returns
        -------
        dict
            Aggregated data for all coins, including:
                - Market cap of all coins (cap)
                - Volume of all coins
                - 2% liquidity of all coins (liquidity)
                - Percentage of BTC market cap in total market cap (btcDominance)
        """

        url = urljoin(self.api, self.URL_OVERVIEW)

        r = requests.post(url, headers=self.headers, data=json.dumps({"currency": currency}))

        if r.status_code == 200:

            return r.json()

        else:

            self.raise_exception(r)

    def historical_market_overview(
        self, currency:str, start:datetime, end:datetime
    )->pd.DataFrame:

        """
        Get historical aggregated data of entire market.

        Parameters
        ----------
        currency: str
            Any valid coin or fiat code
        start: datetime
            Datetime of time interval start.
        end: datetime
            Datetime of time interval end.

        Returns
        -------
        df: pd.DataFrame
            DataFrame containing:
                - date of the datapoint
                - Market cap of all coins (cap)
                - Volume of all coins
                - 2% liquidity of all coins (liquidity)
                - Percentage of BTC market cap in total market cap (btcDominance)
        """

        url = urljoin(self.api, self.URL_HISTORICAL_OVERVIEW)

        data = {
            "currency": currency,
            "start": self.format_datetime(start),
            "end": self.format_datetime(end)
        }

        r = requests.post(url, headers=self.headers, data=json.dumps(data))

        if r.status_code == 200:

            df = pd.DataFrame(r.json())

            df.loc[:, "date"] = pd.to_datetime(
                df.loc[:, "date"], origin="unix", unit="ms"
            )

            return df

        else:

            self.raise_exception(r)

    def current_coin_info(
        self, currency:str, code:str, meta:bool=False
    )->dict:
        """
        All information about a single coin at latest moment in time.

        Parameters
        ----------
        currency: str
            Any valid coin or fiat code.
        code: str
            Coin code.
        meta: bool
            To include full coin information or not.

        Returns
        -------
        dict
            Dictionary containing:
                - Price of coin in requested currency (rate)
                - Reported trading volume of the coin in last 24 hours in requested currency (volume)
                - Coin's market cap in requested currency (cap)
            If meta==True, the dictionary also includes:
                - Coin's name (name)
                - Coin's symbol (symbol)
                - Hexadecimal color code (color)
                - 32 pixel png image of coin icon (png32)
                - 64 pixel png image of coin icon (png64)
                - 32 pixel webp image of coin icon (webp32)
                - 64 pixel webp image of coin icon (webp64)
                - Number of exchange coin is present at (exchanges)
                - Number of markets coin is present at (markets)
                - Number of unique markets coin is present at (pairs)
                - All time high in USD (allTimeHighUSD)
                - Number of coins minted, but not locked (circulatingSupply)
                - Number of coins minted, including locked (totalSupply)
                - Maximum number of coins that can be minted (maxSupply)
                - Coin's hypothetical total capitalization at the moment (totalCap)
        """
        url = urljoin(self.api, self.URL_COINS_SINGLE)

        data = {
            "currency": currency,
            "code": code,
            "meta": meta,
        }

        r = requests.post(url, headers=self.headers, data=json.dumps(data))

        if r.status_code == 200:

            return r.json()

        else:

            self.raise_exception(r)

    def historical_coin_info(
        self, currency:str, code:str, start:datetime,
        end:datetime, meta:bool=False
    )->dict:
        """
        Histocial values for coin.

        Parameters
        ----------
        currency: str
            Any valid coin or fiat code.
        code: str
            Coin code.
        start: datetime
            Datetime of time interval start.
        end: datetime
            Datetime of time interval end.
        meta: bool
            To include full coin information or not.

        Returns
        -------
        dict
            Dictionary containing:
                - DataFrame with historical price (rate), volume (volume) and
                market cap (cap) for the requested interval (history).
            If meta==True, the dictionary also includes:
                - Coin's name (name)
                - Coin's symbol (symbol)
                - Hexadecimal color code (color)
                - 32 pixel png image of coin icon (png32)
                - 64 pixel png image of coin icon (png64)
                - 32 pixel webp image of coin icon (webp32)
                - 64 pixel webp image of coin icon (webp64)
                - Number of exchange coin is present at (exchanges)
                - Number of markets coin is present at (markets)
                - Number of unique markets coin is present at (pairs)
                - All time high in USD (allTimeHighUSD)
                - Number of coins minted, but not locked (circulatingSupply)
                - Number of coins minted, including locked (totalSupply)
                - Maximum number of coins that can be minted (maxSupply)
                - Coin's hypothetical total capitalization at the moment (totalCap)
        """

        url = urljoin(self.api, self.URL_COINS_SINGLE_HISTORY)

        data = {
            "currency": currency,
            "code": code,
            "start": self.format_datetime(start),
            "end": self.format_datetime(end),
            "meta": meta,
        }

        r = requests.post(url, headers=self.headers, data=json.dumps(data))

        if r.status_code == 200:

            d = r.json()
            df = pd.DataFrame(d["history"])
            df.loc[:, "date"] = pd.to_datetime(
                df.loc[:, "date"], origin="unix", unit="ms"
            )
            d["history"] = df

            return d

        else:

            self.raise_exception(r)

    def current_coins_info(
        self, currency:str, sort:str, order:str, offset:int=0, limit:int=10,
        meta:bool=False
    )->pd.DataFrame:

        """
        Assorted information for a list of coins.

        Parameters
        ----------
        currency: str
            Any valid coin or fiat code.
        sort: str
            Sorting parameter. Possible values: rank, price, volume, code, name
        order: str
            Sorting order. Possible values: ascending or descending
        offset: int
            Offset of the list, default 0.
        limit: int
            Limit of the list, default 10, maximum 100
        meta: bool
            To include full coin information or not.

        Returns
        -------
        pd.DataFrame
            DataFrame with the same information as "current_coin_info" for all
            the coins in the list.
        """

        url = urljoin(self.api, self.URL_COINS_LIST)

        data = {
            "currency": currency,
            "sort": sort,
            "order": order,
            "offset": offset,
            "limit": limit,
            "meta": meta,
        }

        r = requests.post(url, headers=self.headers, data=json.dumps(data))

        if r.status_code == 200:

            df = pd.DataFrame(r.json())

            return df

        else:

            self.raise_exception(r)

    def list_of_fiats(self,)->pd.DataFrame:
        """
        List of all the fiats.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the following information for each fiat:
            - Fiat ISO code (code)
            - ISO country code list (country)
            - ISO country code of the flag (flag)
            - Fiat name (name)
            - Fiat symbol (symbol)
        """
        url = urljoin(self.api, self.URL_ALL_FIATS)

        r = requests.post(url, headers=self.headers)

        if r.status_code == 200:

            return pd.DataFrame(r.json())

        else:

            self.raise_exception(r)

    def exchange_info(
        self, currency:str, code:str, meta:bool=False
    )->dict:
        """
        Assorted exchange information.

        Parameters
        ----------
        currency: str
            Any valid coin or fiat code.
        code: str
            Exchange code
        meta: bool
            To include full exchange information or not

        Returns
        -------
        dict
            Dictionary containing:
                - Exchange name (name)
                - 64-pixel png image of exchange icon (png64)
                - 128-pixel png image of exchange icon (png128)
                - 64-pixel webp image of exchange icon (webp64)
                - 128-pixel webp image of exchange icon (webp128)
                - Is the exchange centralized or decentralized (centralized)
                - Is the exchange compliant in the USA (usCompliant)
                - Exchange code (code)
                - Count of currently active markets on the exchange (markets)
                - 24 hour volume in specified currency (volume)
                - 2% order book value bids (bidTotal)
                - 2% order book value asks (askTotal)
                - Number of daily visitors, estimate (visitors)
                - Daily volume per daily visitor (volumePerVisitor)
        """
        url = urljoin(self.api, self.URL_EXCHANGES_SINGLE)

        data = {
            "currency": currency,
            "code": code,
            "meta": meta,
        }

        r = requests.post(url, headers=self.headers, data=json.dumps(data))

        if r.status_code == 200:

            return r.json()

        else:

            self.raise_exception(r)

    def exchanges_info(
        self, currency:str, sort:str, order:str, offset:str=0,
        limit:int=50, meta:bool=False
    ):
        """
        Assorted information on list of exchanges.

        Parameters
        ----------
        currency: str
            Any valid coin or fiat code.
        sort: str
            Sorting parameter. Possible values: volume, liquidity, code, name
        order: str
            Sorting order. Possible values: ascending or descending
        offset: int
            Offset of the list, default 0.
        limit: int
            Limit of the list, default 10, maximum 100
        meta: bool
            To include full coin information or not, default False.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the following data for each exchange:
                - Exchange name (name)
                - 64-pixel png image of exchange icon (png64)
                - 128-pixel png image of exchange icon (png128)
                - 64-pixel webp image of exchange icon (webp64)
                - 128-pixel webp image of exchange icon (webp128)
                - Is the exchange centralized or decentralized (centralized)
                - Is the exchange compliant in the USA (usCompliant)
                - Exchange code (code)
                - Count of currently active markets on the exchange (markets)
                - 24 hour volume in specified currency (volume)
                - 2% order book value bids (bidTotal)
                - 2% order book value asks (askTotal)
                - Number of daily visitors, estimate (visitors)
                - Daily volume per daily visitor (volumePerVisitor)
        """
        url = urljoin(self.api, self.URL_EXCHANGES_LIST)

        data = {
            "currency": currency,
            "sort": sort,
            "offset": offset,
            "limit": limit,
            "meta": meta,
        }

        r = requests.post(url, headers=self.headers, data=json.dumps(data))

        if r.status_code == 200:

            return pd.DataFrame(r.json())

        else:

            self.raise_exception(r)


if __name__ == "__main__":

    lc = LiveCoinWatchWrapper("46dcd129-194e-4f7e-8dfd-9ba5c47cfdb8")

    lc.current_coin_info("USD", "BTC")