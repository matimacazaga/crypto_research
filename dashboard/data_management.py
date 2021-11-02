# from typing import Tuple
# from .binance_wrapper import BinanceWrapper
# from pycoingecko import CoinGeckoAPI
# import yfinance as yf
import os
from .config import BASE_PATH

if not os.path.isdir(BASE_PATH):
    os.mkdir(BASE_PATH)

from joblib import Parallel, delayed
from pycoingecko import CoinGeckoAPI
from .livecoinwatch_wrapper import LiveCoinWatchWrapper
from dateutil.relativedelta import relativedelta
import pickle
import pandas as pd
import ccxt
import json
from datetime import datetime
import numpy as np
import yfinance as yf
import requests

class CoinNotFound(Exception):

    def __init__(self, message:str):

        self.message = message

        super().__init__(self.message)

class DataManager:

    def __init__(self):

        self.exchanges = ['gateio', 'hitbtc', 'mexc', 'binance', 'kucoin', 'yobit']

        self.supported_tickers = self._get_supported_tickers()

        self.livecoinwatch = LiveCoinWatchWrapper(
            # "46dcd129-194e-4f7e-8dfd-9ba5c47cfdb8"
            "703afe15-7cd2-4cc5-8b52-bb33d925f6a4"
        )

        self.coingecko = CoinGeckoAPI()

        self.coins_dict = json.load(open("./crypto_classification/coins1.json"))

        self.messari_api = "https://data.messari.io/api/v2"


    @staticmethod
    def _df_agg(df:pd.DataFrame, col:str)->pd.DataFrame:
        """
        Aggregates symbols information by col, and calculates the total
        mkt cap, count and symbols by group.

        Parameters
        ----------
        df: pd.DataFrame
            DataFrame with symbols information (mkt_cap, symbol, sector,
            category)
        col: str
            Col to group by.
        """
        df_agg = df.groupby(col).agg(
            {
                "mkt_cap": [("mkt_cap", "sum")],
                "symbol": [("symbols", lambda x: x.tolist()), ("count", "count")]
            }
        ).droplevel(0, axis=1).reset_index()

        df_agg.loc[:, "mkt_cap_pct"] = df_agg.loc[:, "mkt_cap"] / df_agg.loc[:, "mkt_cap"].sum()
        df_agg.sort_values("count", inplace=True, ascending=False)

        return df_agg


    def get_coins_information(self,):

        r = requests.get(self.messari_api + "/assets?limit=100")

        d = r.json()

        symbols_information = {
            l["symbol"]: {
                "sector": l["profile"]["general"]["overview"]["sector"],
                "category":  l["profile"]["general"]["overview"]["category"],
                "mkt_cap": l["metrics"]["marketcap"]["current_marketcap_usd"]
            } for l in d["data"]
        }

        symbols_information = pd.DataFrame.from_dict(
            symbols_information, orient="index"
        ).reset_index().rename({"index": "symbol"}, axis=1)

        sector_agg = self._df_agg(symbols_information, "sector")

        category_agg = self._df_agg(symbols_information, "category")

        return symbols_information, sector_agg, category_agg


    def _get_exchange_markets(self, exchange:str):
        """
        Get supported markets by the exchange.

        Parameters
        ----------
        exchange: str
            Exchange name.
        Returns
        -------
        tuple
            Exchange name and list of markets.
        """
        client = getattr(ccxt, exchange)()
        markets = client.load_markets()
        return exchange, list(markets.keys())


    def _get_supported_tickers(self):
        """
        Get supported tickers by all the available exchanges.

        Returns
        -------
        tickers: dict
            Dictionary with exchange name as key and supported tickers as
            values.
        """
        tickers = Parallel(n_jobs=6, backend="threading")(delayed(self._get_exchange_markets)(exchange) for exchange in self.exchanges)

        return dict(tickers)


    def _convert_datetime(self, date:datetime)->int:
        """
        Convert datetime to timestamp (ms).

        Parameters
        ----------
        datetime: datetime
            Date to convert

        Returns
        -------
        int
            Date converted into timestamp (unit=ms)
        """

        return int(date.timestamp()*1000)


    @staticmethod
    def _standardize_datetime(date:datetime):

        return datetime(date.year, date.month, date.day, 16, 0, 0)


    def _fetch_ohlcv(self, client, symbol:str, since:datetime, limit:int)->pd.DataFrame:
        since_ = self._convert_datetime(since)
        df = client.fetch_ohlcv(symbol, timeframe="1d", since=since_, limit=limit)
        df = pd.DataFrame(df, columns=["date", "open", "high", "low", "close", "volume"])
        df.loc[:, "date"] = pd.to_datetime(df.loc[:, "date"], origin="unix", unit="ms")
        return df


    def _get_coin_history_ccxt(self, symbol:str, from_date:datetime,
        to_date:datetime, log_returns:bool=True,):

        for exchange, tickers in self.supported_tickers.items():
            if symbol + "/USDT" in tickers:
                client = getattr(ccxt, exchange)()
                symbol_ = symbol + "/USDT"
                exchange_ = exchange
                break
            elif symbol + "/USD" in tickers:
                client = getattr(ccxt, exchange)()
                symbol_ = symbol + "/USD"
                exchange_ = exchange
                break
        else:
            raise CoinNotFound(f"Coin not found in any of the following exchanges: {self.exchanges}")

        days = (to_date - from_date).days
        if days > 1000:
            limits = days // 999
            limits = [999 for _ in range(limits)]
            if modulo:=days%999:
                limits += [modulo]
            dfs = []
            for l, lim in enumerate(limits):
                dfs.append(
                    self._fetch_ohlcv(
                        client,
                        symbol_,
                        since=from_date if l==0 else dfs[-1].iloc[-1]["date"].to_pydatetime(),
                        limit=lim
                    )
            )

            df = pd.concat(dfs)

            df.drop_duplicates(subset="date", keep="first", inplace=True)

        else:

            df = self._fetch_ohlcv(client, symbol_, since=from_date, limit=days)

        df.rename({"close": "price"}, axis=1, inplace=True)

        returns = df.loc[:, "price"].pct_change()

        df.loc[:, "return"] = np.log1p(returns) if log_returns else returns

        return df.loc[:, ["date", "price", "return", "volume"]], exchange_


    def _historical_coin_info_livewatch(self, symbol:str, from_date:datetime,
        to_date:datetime):

        try:
            df = self.livecoinwatch.historical_coin_info(
                "USD", symbol, from_date, to_date
            )

        except:

            df = {"history": pd.DataFrame(columns=["date", "mkt_cap", "price", "volume", "return"])}

        return df


    def _get_coin_history_livecoinwatch(self, symbol:str, from_date:datetime,
        to_date:datetime, log_returns:bool=True):

        days = (to_date-from_date).days
        if days>100:
            dates = [from_date]
            for d in range(days//100):
                if d == 0:
                    dates.append(from_date + relativedelta(days=100))
                else:
                    dates.append(dates[-1] + relativedelta(days=100))

            dates.append(to_date)

            df = Parallel(
                n_jobs=len(dates)-1, backend="threading"
            )(delayed(
                self._historical_coin_info_livewatch
            )(symbol, dates[i], dates[i+1]) for i in range(len(dates)-1))

            df = pd.concat([d["history"] for d in df])

        else:

            df = self.livecoinwatch.historical_coin_info(
                "USD", symbol, from_date, to_date
            )["history"]

        df.set_index("date", inplace=True)

        df = df.resample("1d").agg(
            {
                "cap": [("mkt_cap", "last")],
                "rate": [("price", "last")],
                "volume": [("volume", "last")]
            }
        ).droplevel(0, axis=1).reset_index()

        returns = df.loc[:, "price"].pct_change()

        df.loc[:, "return"] = np.log1p(returns) if log_returns else returns

        return df, "livecoinwatch"


    def get_coin_history(
        self, symbol:str, from_date:datetime, to_date:datetime,
        log_returns:bool=True, include_mkt_cap:bool=False
    )->tuple:
        """
        Get historical daily prices for a coin.

        Parameters
        ----------
        symbol: str
            Symbol of the coin. Eg.: BTC
        from_date: datetime
            Start date.
        to_date: datetime
            End date.

        Returns
        -------
        coin_data: pd.DataFrame
            DataFrame with historical data. Columns: date, open, high, low,
            close, volume.
        exchange: str
            Source of the data.
        """
        to_date = self._standardize_datetime(to_date)
        from_date = self._standardize_datetime(from_date)
        if include_mkt_cap:
            coin_data, exchange = self._get_coin_history_livecoinwatch(
                symbol, from_date, to_date, log_returns
            )
        else:
            coin_data, exchange = self._get_coin_history_ccxt(
                symbol, from_date, to_date,log_returns
            )

        return coin_data.dropna(), exchange


    def _get_mkt_cap_coingecko(self, symbol:str, from_date:datetime,
        to_date:datetime)->pd.DataFrame:

        coin_data = self.coingecko.get_coin_market_chart_range_by_id(
                id=self.coins_dict[symbol.lower()]["coingecko_id"],
                vs_currency='usd',
                from_timestamp=self._convert_datetime(from_date)/1000,
                to_timestamp=self._convert_datetime(to_date)/1000
        )

        df = {}
        df["date"] = [l[0] for l in coin_data["prices"]]
        df["mkt_cap"] = [l[1] for l in coin_data["market_caps"]]

        df = pd.DataFrame(df)

        df.loc[:, "date"] = pd.to_datetime(df.loc[:, "date"], origin="unix", unit="ms")

        return df


    def get_coin_market_cap_history(self, symbol:str, from_date:datetime,
        to_date:datetime, use_coingecko:bool=False):
        from_date = self._standardize_datetime(from_date)
        to_date = self._standardize_datetime(to_date)

        if use_coingecko:

            return self._get_mkt_cap_coingecko(symbol, from_date, to_date)

        else:

            return self._get_coin_history_livecoinwatch(symbol, from_date, to_date)


    def get_top_coins_by_mkt_cap(self, limit:int=100, offset:int=0):

        df = self.livecoinwatch.current_coins_info(
            currency="USD", sort="rank", order="ascending",
            offset=offset, limit=limit, meta=True
        )

        if "symbol" in df.columns:

            df.drop("symbol", axis=1, inplace=True)

        df.rename({"code": "symbol", "cap": "mkt_cap", "rate": "price"}, axis=1, inplace=True)

        df.loc[:, "symbol"] = df.loc[:, "symbol"].str.strip("_")

        return df.loc[:, ["symbol", "price", "mkt_cap", "volume", "allTimeHighUSD", "circulatingSupply", "maxSupply"]]

    @staticmethod
    def load_coins_on_chain_metrics()->pd.DataFrame:

        df = pickle.load(open(f"{BASE_PATH}/coins_metrics", "rb"))

        cols = ["symbol"] + [col for col in df.columns if col != "symbol"]

        df.dropna(axis=0, how="all", subset=[col for col in cols if col!= "coin"], inplace=True)

        return df.loc[:, cols]

    @staticmethod
    def load_cluster_df()->pd.DataFrame:

        df = pickle.load(open(f"{BASE_PATH}/df_clustering", "rb"))

        return df


    def get_spy_price(self, from_date:datetime, to_date:datetime,
        log_returns:bool=True):

        from_date = self._standardize_datetime(from_date)
        to_date = self._standardize_datetime(to_date)

        df = yf.download(
            "SPY", start=from_date, end=to_date
        )

        returns = df.loc[:, "Adj Close"].pct_change()

        df.loc[:, "return"] = np.log1p(returns) if log_returns else returns

        df = df.loc[:, ["Adj Close", "return"]].reset_index()

        df.rename({"Date": "date", "Adj Close": "price"}, axis=1, inplace=True)

        return df.dropna()

