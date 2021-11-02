
from datetime import datetime
from functools import cache
import pickle
import streamlit as st

from dashboard.livecoinwatch_wrapper import LiveCoinWatchException
from .finance import *
from .data_management import *
from .plotting import *
import altair as alt
import pandas as pd
from .config import BASE_PATH, lookback_period

alt.themes.enable("fivethirtyeight")
alt.renderers.set_embed_options(actions=False)


def run_app():

    # DATA DOWNLOADING

    TODAY = datetime.now()
    TODAY_STR = TODAY.strftime("%Y-%m-%d %H:%M:%S")

    INIT_DATE = TODAY - relativedelta(years=lookback_period)

    with st.spinner("Descargando datos"):
        if "last_update" not in st.session_state or (datetime.now()- st.session_state["last_update"]).seconds / 60 > 240:
            try:
                dm = DataManager()

                BTC, _ = dm.get_coin_history("BTC", INIT_DATE, TODAY, include_mkt_cap=True)
                SPY = dm.get_spy_price(INIT_DATE, TODAY)

                coins_by_market_cap = dm.get_top_coins_by_mkt_cap()

                coins_history = Parallel(n_jobs=50, backend="threading")(
                    delayed(
                        dm.get_coin_history
                    )(
                        symbol, INIT_DATE, TODAY, include_mkt_cap=True
                    ) for symbol in coins_by_market_cap.loc[:, "symbol"]
                )

                coins_history = {symbol: coin_history for symbol, coin_history in zip(coins_by_market_cap.loc[:, "symbol"], coins_history)}

                symbols_information, sector_agg, category_agg = dm.get_coins_information()

                for symbol in symbols_information:
                    if symbol not in coins_history:
                        try:
                            h = dm.get_coin_history(symbol, INIT_DATE, TODAY, include_mkt_cap=True)
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

                st.session_state["last_update"] = datetime.now()

            except LiveCoinWatchException as e:

                print(e)

                cache_dict = pickle.load(open(f"{BASE_PATH}/cache", "rb"))

                dm = cache_dict["dm"]

                BTC = cache_dict["BTC"]

                SPY = cache_dict["SPY"]

                coins_by_market_cap = cache_dict["coins_by_market_cap"]

                coins_history = cache_dict["coins_history"]

                coins_stats = cache_dict["coins_stats"]

                symbols_information = cache_dict["symbols_information"]

                sector_agg = cache_dict["sector_agg"]

                category_agg = cache_dict["category_agg"]

                portfolios_by_sector = cache_dict["portfolios_by_sector"]

                portfolios_by_category = cache_dict["portfolios_by_category"]

        else:

            cache_dict = pickle.load(open(f"{BASE_PATH}/cache", "rb"))

            dm = cache_dict["dm"]

            BTC = cache_dict["BTC"]

            SPY = cache_dict["SPY"]

            coins_by_market_cap = cache_dict["coins_by_market_cap"]

            coins_history = cache_dict["coins_history"]

            coins_stats = cache_dict["coins_stats"]

            symbols_information = cache_dict["symbols_information"]

            sector_agg = cache_dict["sector_agg"]

            category_agg = cache_dict["category_agg"]

            portfolios_by_sector = cache_dict["portfolios_by_sector"]

            portfolios_by_category = cache_dict["portfolios_by_category"]


    # MKT CAP TABLE
    coins_by_market_cap = coins_by_market_cap.merge(coins_stats, on="symbol")

    # ANALYSIS BY COIN
    st.subheader("Análisis por criptomoneda")

    # DROPDOWN
    symbols = coins_by_market_cap.loc[:, "symbol"].tolist()
    symbols.sort()

    selected_symbol = st.selectbox(
        'Seleccionar una criptomoneda',
        symbols
    )

    with st.spinner("Descargando datos"):
        df, source = dm.get_coin_history(selected_symbol, INIT_DATE, TODAY)

    st.markdown(f"**Source**: {source}")

    # DAILY RETURNS STATS

    st.markdown("### Descripción de la criptomoneda")

    coins_dict = json.load(open("./crypto_classification/coins2.json"))

    try:
        st.markdown(f"**Description**: {coins_dict[selected_symbol.lower()]['desc']}")
    except:
        st.markdown('Description not available')
    st.markdown("### Daily returns statistics")

    coin_stats = get_coin_returns_stats(df)

    st.dataframe(coin_stats.style.format(formatter="{:.3%}"), width=660)

    # COIN CHARTS
    price_chart, return_chart, return_dist, boxplot = make_coin_plots(df)

    st.altair_chart(
        price_chart,
        use_container_width=False
    )

    st.altair_chart(
        return_chart,
        use_container_width=False
    )

    st.altair_chart(
        return_dist,
        use_container_width=True
    )

    st.altair_chart(
        boxplot,
        use_container_width=True
    )
