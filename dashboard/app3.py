
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

METRICS_DESC = pickle.load(open(f"{BASE_PATH}/metrics_description", "rb"))

def describe_metric(metric_to_describe:str)->None:

    d = [d for d in METRICS_DESC if d["id"] == metric_to_describe][0]

    st.markdown(
        f"**Name**: {d['name']}"
    )

    st.markdown(
        f"**Category**: {d['category']}"
    )

    st.markdown(
        f"**Description**: {d['description']}"
    )

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

        #ON CHAIN ANALYSIS
    st.subheader("Métricas On-Chain")

    df = dm.load_coins_on_chain_metrics()

    st.dataframe(df.style.format(
        formatter={
            col: "{:.3f}" for col in df.columns.tolist() if col!= "symbol"
        }
    ))

    with st.expander("Descripción de las métricas"):

        metric_to_describe = st.selectbox(
            'Seleccionar una métrica',
            [d["id"] for d in METRICS_DESC],
        )

        describe_metric(metric_to_describe)

    df = dm.load_cluster_df()

    st.markdown("### Clustering")

    cols = df.drop(["symbol", "cluster", "PC1", "PC2"], axis=1).columns.tolist()

    st.markdown(f"Para realizar el *clustering* se utilizaron las siguientes variables: {', '.join(cols)}")

    cluster_chart = make_cluster_plot(df)

    st.altair_chart(cluster_chart, use_container_width=True)

    st.markdown("Estadísticas de portafolios creados a partir de los clusters, utilizando Mkt. Cap. como peso.")

    df = pickle.load(open("./dashboard/data/clusters_stats", "rb"))

    clusters_stats_chart = make_clusters_stats_plot(df)

    st.altair_chart(clusters_stats_chart, use_container_width=True)

