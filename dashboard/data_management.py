from typing import Tuple
from .binance_wrapper import BinanceWrapper
from pycoingecko import CoinGeckoAPI
import yfinance as yf
import os
from datetime import datetime
import pickle
import pandas as pd
import numpy as np

COINGECKO = CoinGeckoAPI()
BINANCE = BinanceWrapper()
BINANCE_COINS_LIST = BINANCE.get_crypto_list()

BASE_PATH = "./dashboard/data"

ret_log = True # True si queres retornos logaritmicos, False si queres normales

if not os.path.isdir(BASE_PATH):
    os.mkdir(BASE_PATH)

if not os.path.isdir(f"{BASE_PATH}/coins"):
        os.mkdir(f"{BASE_PATH}/coins")

if not os.path.isdir(f"{BASE_PATH}/spy"):
        os.mkdir(f"{BASE_PATH}/spy")

def get_spy_price(from_datetime, to_datetime):

    file_path = f"{BASE_PATH}/spy/spy_{from_datetime.strftime('%Y_%m_%d')}_{to_datetime.strftime('%Y_%m_%d')}"
    if os.path.exists(file_path):
        df = pickle.load(open(file_path, "rb"))
        return df

    df = yf.download(
        "SPY", start=from_datetime, end=to_datetime
    )

    if (ret_log): df.loc[:, "return"] = np.log1p(df.loc[:, "Adj Close"].pct_change())
    else: df.loc[:, "return"] = df.loc[:, "Adj Close"].pct_change()

    df.rename({"Adj Close": "price"}, axis=1, inplace=True)

    df = df.loc[:, ["price", "return"]]

    pickle.dump(df, open(file_path, "wb"))

    return df

def get_coin_price_binance(binance_symbol:str, from_datetime:datetime,
    to_datetime:datetime)->Tuple[pd.DataFrame, str]:

    coin_data = BINANCE.get_klines(
            binance_symbol,
            "1d",
            1000,
            from_datetime,
            to_datetime,
    )

    df = coin_data.loc[:, ["close", "volume"]].copy()

    try:

        df.loc[:, "date"] = coin_data.loc[:, "close_time"].dt.date

        df.loc[:, "date"] = pd.to_datetime(df.loc[:, "date"])

    except:

        df = pd.DataFrame(columns=["date", "price", "total_volume"])

        return df

    df.rename({"close": "price", "volume":"total_volume"}, axis=1, inplace=True)

    source = "Binance"

    return df, source


def get_coin_price_coingecko(coin:list, from_datetime:datetime,
    to_datetime:datetime)->Tuple[pd.DataFrame, str]:

    coin_data = COINGECKO.get_coin_market_chart_range_by_id(
            id=coin[0], vs_currency='usd',from_timestamp=from_datetime.timestamp(),to_timestamp=to_datetime.timestamp()
    )

    dates = [l[0] for l in coin_data["prices"]]
    prices = [l[1] for l in coin_data["prices"]]
    volumes = [l[1] for l in coin_data["total_volumes"]]

    df = pd.DataFrame({
        "date": dates,
        "price": prices,
        "total_volume": volumes,
    })

    df.loc[:, "date"] = pd.to_datetime(df.loc[:, "date"], origin="unix", unit="ms")

    source = "Coingecko"

    return df, source

def get_coin_price(coin:list,
    from_datetime:datetime, to_datetime:datetime)->Tuple[pd.DataFrame, str]:

    # file_path = f"{BASE_PATH}/coins/{coin[0]}_{from_datetime.strftime('%Y_%m_%d')}_{to_datetime.strftime('%Y_%m_%d')}"

    # if os.path.exists(file_path):
    #     df, source = pickle.load(open(file_path, "rb"))
    #     return df, source

    if coin[1].upper() + "USDT" in BINANCE_COINS_LIST:

        df, source = get_coin_price_binance(coin[1].upper() + "USDT", from_datetime, to_datetime)

        if (ret_log): df.loc[:, "return"] = np.log1p(df.loc[:, "price"].pct_change())
        else: df.loc[:, "return"] = df.loc[:, "price"].pct_change()
        

        return df, source

    elif coin[1].upper() + "BTC" in BINANCE_COINS_LIST:

        df, source = get_coin_price_binance(coin[1].upper() + "BTC", from_datetime, to_datetime)

        if (ret_log): df.loc[:, "return"] = np.log1p(df.loc[:, "price"].pct_change())
        else: df.loc[:, "return"] = df.loc[:, "price"].pct_change()

        return df, source

    else:

        file_path = f"{BASE_PATH}/coins/{coin[0]}_{from_datetime.strftime('%Y_%m_%d')}_{to_datetime.strftime('%Y_%m_%d')}"

        try:
            df, source = get_coin_price_coingecko(coin, from_datetime, to_datetime)
            if (ret_log): df.loc[:, "return"] = np.log1p(df.loc[:, "price"].pct_change())
            else: df.loc[:, "return"] = df.loc[:, "price"].pct_change()
            pickle.dump((df, source), open(file_path, "wb"))

        except:
            if os.path.exists(file_path):
                df, source = pickle.load(open(file_path, "rb"))

        return df, source

def get_coins_market_caps_cg()->list:
    return COINGECKO.get_coins_markets(
        vs_currency='usd',
        include_market_cap="true"
    )

<<<<<<< HEAD	

#TODO: arreglar pedido de datos a binance cuando el rango excede los 100 dias.	
=======	
>>>>>>> c9dea976c8aa5e6c4182eb04d6b0fd00f27a5115
