from dashboard import app1
from dashboard import app2
from dashboard import app3
from dashboard import app4
import json
from datetime import datetime
import streamlit as st
from .config import BASE_PATH, lookback_period
from dashboard.finance import *
import pickle
from .config import BASE_PATH, lookback_period
from dateutil.relativedelta import relativedelta
from .data_management import DataManager
import os
from joblib import Parallel, delayed

#app.py
def run_app():

    TODAY = datetime.now()

    TODAY_STR = TODAY.strftime("%Y-%m-%d")

    INIT_DATE = TODAY - relativedelta(years=lookback_period)

    INIT_DATE_STR = INIT_DATE.strftime("%Y-%m-%d")

    with st.spinner("Descargando datos"):
        if os.path.isfile(f"{BASE_PATH}/last_update.json"):
            last_update = json.load(open(f"{BASE_PATH}/last_update.json", "r"))
        else:
            last_update = {}

        if last_update.get("date", None) != TODAY_STR:

            dm = DataManager()

            SPY = dm.get_spy_price(INIT_DATE, TODAY)

            coins_by_market_cap = dm.get_top_coins_by_mkt_cap()

            coins_history = Parallel(n_jobs=50, backend="threading")(
                delayed(
                    dm.get_coin_history
                )(
                    symbol, INIT_DATE, TODAY, include_mkt_cap=True
                ) for symbol in coins_by_market_cap.loc[:, "symbol"]
            )

            coins_history = {
                symbol: coin_history
                for symbol, coin_history in zip(
                    coins_by_market_cap.loc[:, "symbol"], coins_history
                ) if coin_history[0].shape[0]>10
            }

            BTC = coins_history["BTC"][0]

            symbols_information, sector_agg, category_agg = dm.get_coins_information()

            for symbol in symbols_information:
                if symbol not in coins_history:
                    try:
                        h = dm.get_coin_history(
                            symbol, INIT_DATE, TODAY, include_mkt_cap=True
                        )
                        coins_history[symbol] = h
                    except:
                        continue

            coins_stats = get_coins_stats(coins_history, SPY, BTC)

            portfolios_by_sector = get_portfolios_by_col(
                coins_history, sector_agg, "sector"
            )

            portfolios_by_category = get_portfolios_by_col(
                coins_history, category_agg, "category"
            )

            cache_dict = {
                "init_date": INIT_DATE,
                "end_date": TODAY,
                "init_date_str": INIT_DATE_STR,
                "end_date_str": TODAY_STR,
                "dm": dm,
                "BTC": BTC,
                "SPY": SPY,
                "coins_by_market_cap": coins_by_market_cap,
                "coins_history": coins_history,
                "coins_stats": coins_stats,
                "symbols_information": symbols_information,
                "sector_agg": sector_agg,
                "category_agg": category_agg,
                "portfolios_by_category": portfolios_by_category,
                "portfolios_by_sector": portfolios_by_sector
            }

            pickle.dump(cache_dict, open(f"{BASE_PATH}/cache", "wb"))

            last_update = {"date": TODAY_STR}

            json.dump(last_update, open(f"{BASE_PATH}/last_update.json", "w"))

        else:

            cache_dict = pickle.load(open(f"{BASE_PATH}/cache", "rb"))

    PAGES = {
        "General Analysis": app1,
        "Specific Crypto Analysis": app2,
        "On-Chain Metrics and Clustering": app3,
        "Sector Portfolios": app4
    }

    st.sidebar.title('Navigation')
    selection = st.sidebar.radio("Go to", list(PAGES.keys()))
    page = PAGES[selection]
    page.run_app(cache_dict)