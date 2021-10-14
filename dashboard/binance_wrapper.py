"""
Wrapper para Binance
"""
from datetime import datetime
from urllib.parse import urljoin

import requests
import pandas as pd
import numpy as np


class BinanceBase:

    URL_TEST_CONNECTION = '/api/v3/ping'
    URL_KLINES = '/api/v3/klines'
    URL_PRICE_CHANGE_STATISTICS_24HR = '/api/v3/ticker/24hr'
    URL_ORDER_BOOK = '/api/v3/depth'
    URL_AVG_PRICE = '/api/v3/avgPrice'
    URL_LAST_PRICE = '/api/v3/ticker/price'
    URL_BOOK_TICKER = '/api/v3/ticker/bookTicker'
    URL_RECENT_TRADES = '/api/v3/trades'

    ORDER_BOOK_COLUMNS = [
        'bid_price',
        'bid_qty',
        'ask_price',
        'ask_qty',
    ]
    KLINES_COLUMNS = [
        'open_time',
        'open',
        'high',
        'low',
        'close',
        'volume',
        'close_time',
        'quote_asset_vol',
        'number_of_trades',
        'taker_buy_base_asset_vol',
        'taker_buy_quote_asset_vol',
        'ignore',
    ]

    RECENT_TRADES_TYPES = {
        'id': int,
        'price': float,
        'qty': float,
        'quoteQty': float,
        'time': np.int64,
        'isBuyerMaker': bool,
        'isBestMatch': bool,
    }
    AVG_PRICE_TYPES = {
        'mins': int,
        'price': float,
    }
    KLINES_TYPES = {
        'open_time': np.int64,
        'open': float,
        'high': float,
        'low': float,
        'close': float,
        'volume': float,
        'close_time': np.int64,
        'quote_asset_vol': float,
        'number_of_trades': int,
        'taker_buy_base_asset_vol': float,
        'taker_buy_quote_asset_vol': float,
    }
    PRICE_CHANGE_STATISTICS_TYPES = {
        'symbol': str,
        'priceChange': float,
        'priceChangePercent': float,
        'weightedAvgPrice': float,
        'prevClosePrice': float,
        'lastPrice': float,
        'lastQty': float,
        'bidPrice': float,
        'bidQty': float,
        'askPrice': float,
        'askQty': float,
        'openPrice': float,
        'highPrice': float,
        'lowPrice': float,
        'volume': float,
        'quoteVolume': float,
        'openTime': np.int64,
        'closeTime': np.int64,
        'firstId': int,
        'lastId': int,
        'count': int,
    }
    LAST_PRICE_TYPES = {
        'symbol': str,
        'price': float,
    }
    BOOK_TICKER_TYPES = {
        'symbol': str,
        'bidPrice': float,
        'bidQty': float,
        'askPrice': float,
        'askQty': float,
    }

    @staticmethod
    def raise_exception(r: requests.models.Response):
        j = r.json()
        if 'msg' in j:
            raise BinanceException(status_code=r.status_code, data=j)
        else:
            raise BinanceException(status_code=r.status_code)

    @staticmethod
    def format_datetime(dt: datetime) -> int:
        if not dt:
            return dt
        return int(dt.timestamp() * 1000)


class BinanceException(Exception):

    def __init__(self, status_code, data=None):

        self.status_code = status_code

        if data:
            self.code = data['code']

            self.msg = data['msg']

            message = f"{status_code} [{self.code}] {self.msg}"

        else:

            self.code = None

            self.msg = None

            message = f"status_code={status_code}"

        super().__init__(message)


class BinanceWrapper(BinanceBase):

    def __init__(self,):

        self.api = 'https://api.binance.com'

    def test_connection(self,):
        """
        Testea la conexión a la API.
        """
        url = urljoin(self.api, self.URL_TEST_CONNECTION)

        r = requests.get(url)

        if r.status_code == 200:
            print('Conexión exitosa!')
        else:
            self.raise_exception(r)

    def get_order_book(self, symbol: str, limit: int = 100,) -> pd.DataFrame:
        """
        Obtiene el order book (bids-asks).

        Parameters
        ----------
        symbol: str
            Símbolo del par de criptomonedas requerido. Por ej:
            BTCUSDT.
        limit: int
            Límite de observaciones requeridas. El valor por defecto
            es 100. Los posibles valores son: [5, 10, 20, 50,
            100, 500, 1000, 5000].

        Returns
        -------
        df: pd.DataFrame
            DataFrame con precios y cantidades de los bids y asks.
        """

        params = {
            'symbol': symbol,
            'limit': limit
        }

        url = urljoin(self.api, self.URL_ORDER_BOOK)

        r = requests.get(url, params=params)

        if r.status_code == 200:
            j = r.json()

            df = pd.DataFrame(
                np.concatenate([j['bids'], j['asks']], axis=1),
                columns=self.ORDER_BOOK_COLUMNS
            ).astype(float)

            return df

        else:
            self.raise_exception(r)

    def get_recent_trades(self, symbol: str, limit: int = 10) -> pd.DataFrame:
        """
        Obtiene los trades recientes.

        Parameters
        ----------
        symbol: str
            Símbolo del par de criptomonedas requerido. Por ej:
            BTCUSDT.
        limit: int
            Límite de observaciones requeridas. El valor por defecto
            es 10. El valor máximo es 1000.

        Returns
        -------
        df: pd.DataFrame
            DataFrame con información de los trades recientes.
        """

        params = {
            'symbol': symbol,
            'limit': limit,
        }

        url = urljoin(self.api, self.URL_RECENT_TRADES)

        r = requests.get(url, params=params)

        if r.status_code == 200:

            df = pd.DataFrame(r.json()).astype(self.RECENT_TRADES_TYPES)

            return df

        else:

            self.raise_exception(r)

    def get_klines(self, symbol: str, interval: str, limit: int = 100, start_time: datetime = None, end_time: datetime = None) -> pd.DataFrame:
        """
        Barras kline/candlestick para un dado símbolo. Las klines están identificadas
        de forma única por su tiempo de apertura (open_time).

        NOTA: el campo "ignore" es un "legacy field" (campo heredado) que no contiene
        información relevante. Por esta razón, es eliminado del DataFrame final.

        Parameters
        ----------
        symbol: str
            Símbolo del par de criptomonedas requerido. Por ej:
            BTCUSDT.
        interval: str
            Intervalo entre una observación y otra. Los intervalos disponibles son:
            ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d'
             '3d', '1w', '1M'].
        limit: int
            Límite de observaciones requeridas. El valor por defecto
            es 100. El valor máximo es 1000.
        start_time: datetime
            Fecha inicial. Por ej: datetime(2020, 3, 31, 11, 55, 0)
        end_time: datetime
            Fecha final. Por ej: datetime(2020, 3, 31, 12, 55, 0)

        Returns
        -------
        df: pd.DataFrame
            DataFrame con la información para cada kline.
        """
        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': self.format_datetime(start_time),
            'endTime': self.format_datetime(end_time),
            'limit': limit,
        }

        url = urljoin(self.api, self.URL_KLINES)

        r = requests.get(url, params=params)

        if r.status_code == 200:

            df = pd.DataFrame(r.json(), columns=self.KLINES_COLUMNS).drop(
                'ignore', axis=1).astype(self.KLINES_TYPES)

            df.loc[:, ['open_time', 'close_time']] = (
                df.loc[:, ['open_time', 'close_time']]/1000).applymap(datetime.fromtimestamp)

            return df

        else:

            self.raise_exception(r)

    def get_current_avg_price(self, symbol: str) -> pd.DataFrame:
        """
        Obtiene el precio promedio de los últimos 5 minutos.

        Parameters
        ----------
        symbol: str
            Símbolo del par de criptomonedas requerido. Por ej:
            BTCUSDT.

        Returns
        -------
        df: pd.DataFrame
            DataFrame con minutos considerados en el calculo y el precio
            promedio.
        """

        params = {
            'symbol': symbol,
        }

        url = urljoin(self.api, self.URL_AVG_PRICE)

        r = requests.get(url, params=params)

        if r.status_code == 200:

            j = r.json()

            df = pd.DataFrame(j, index=[0]).astype(self.AVG_PRICE_TYPES)

            df.loc[0, 'symbol'] = symbol

            return df

        else:

            self.raise_exception(r)

    def get_price_change_statistics(self, symbol: str = None) -> pd.DataFrame:
        """
        Estadísticas del cambio de precio en una ventana móvil de 24 hs.
        Cuando no se pasa un símbolo, se devuelven las estadísticas para
        todos los símbolos disponibles.

        Parameters
        ----------
        symbol: str
            Símbolo del par de criptomonedas requerido. Por ej:
            BTCUSDT.

        Returns
        -------
        df: pd.DataFrame
            DataFrame con estadísticas de las últimas 24hs para el/los
            símbolo/s.
        """

        params = {
            'symbol': symbol,
        }

        url = urljoin(self.api, self.URL_PRICE_CHANGE_STATISTICS_24HR)

        r = requests.get(url, params=params)

        if r.status_code == 200:

            j = r.json()

            df = pd.DataFrame(j, index=[0] if symbol else None).astype(
                self.PRICE_CHANGE_STATISTICS_TYPES)

            df.loc[:, ['openTime', 'closeTime']] = (
                df.loc[:, ['openTime', 'closeTime']]/1000).applymap(datetime.fromtimestamp)

            return df

        else:

            self.raise_exception(r)

    def get_last_price(self, symbol: str = None) -> pd.DataFrame:
        """
        Último precio para el/los símbolo/s.

        Parameters
        ----------
        symbol: str
            Símbolo del par de criptomonedas requerido. Por ej:
            BTCUSDT.

        Returns
        -------
        df: pd.DataFrame
            DataFrame con el último precio para el/los símbolo/s.
        """

        params = {
            'symbol': symbol,
        }

        url = urljoin(self.api, self.URL_LAST_PRICE)

        r = requests.get(url, params=params)

        if r.status_code == 200:

            j = r.json()

            df = pd.DataFrame(j, index=[0] if symbol else None).astype(
                self.LAST_PRICE_TYPES)

            return df

        else:

            self.raise_exception(r)

    def get_book_ticker(self, symbol: str = None):
        """
        Obtiene el mejor precio/cantidad en el order book para un símbolo
        o símbolos. Si el parámetro "symbol" es None, se devuelve la
        información para todos los símbolos.

        Parameters
        ----------
        symbol: str
            Símbolo del par de criptomonedas requerido. Por ej:
            BTCUSDT.

        Returns
        -------
        df: pd.DataFrame
            DataFrame con la información para el/los símbolo/s.
        """

        params = {
            'symbol': symbol,
        }

        url = urljoin(self.api, self.URL_BOOK_TICKER)

        r = requests.get(url, params=params)

        if r.status_code == 200:

            j = r.json()

            df = pd.DataFrame(j, index=[0] if symbol else None).astype(
                self.BOOK_TICKER_TYPES)

            return df

        else:

            self.raise_exception(r)

    def get_crypto_list(self) -> list:
        """
        Devuelve una lista de todas las crypto disponibles.

        Returns
        -------
        list
            Lista de criptomonedas disponibles al dia de la fecha.
        """
        df = self.get_last_price()

        return df.loc[:, 'symbol'].tolist()


if __name__ == '__main__':
    binance = BinanceWrapper()
    binance.test_connection()

    data = binance.get_klines(
        symbol='BTCUSDT',
        interval='1d',
        start_time=datetime(2020, 3, 5),
        end_time=datetime(2021, 3, 5),
        limit=1000
    )
    data.set_index('close_time', inplace=True)

    print(data.head())
