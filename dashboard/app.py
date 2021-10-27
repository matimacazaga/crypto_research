
from datetime import datetime
from functools import cache
import pickle
import streamlit as st
from utils import *
from data_management import *
import altair as alt
import pandas as pd
from config import BASE_PATH, lookback_period


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
        if "last_update" not in st.session_state or (datetime.now()- st.session_state["last_update"]).seconds / 60 > 30:
            dm = DataManager()

            BTC, _ = dm.get_coin_history("BTC", INIT_DATE, TODAY)
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

            coins_stats = get_coins_stats(coins_history, SPY, BTC)

            cache_dict = {
                "dm": dm,
                "BTC": BTC,
                "SPY": SPY,
                "coins_by_market_cap": coins_by_market_cap,
                "coins_history": coins_history,
                "coins_stats": coins_stats
            }

            pickle.dump(cache_dict, open(f"{BASE_PATH}/cache", "wb"))

            st.session_state["last_update"] = datetime.now()

        else:

            cache_dict = pickle.load(open(f"{BASE_PATH}/cache", "rb"))

            dm = cache_dict["dm"]

            BTC = cache_dict["BTC"]

            SPY = cache_dict["SPY"]

            coins_by_market_cap = cache_dict["coins_by_market_cap"]

            coins_history = cache_dict["coins_history"]

            coins_stats = cache_dict["coins_stats"]

    # MAIN TITLE
    st.title(f"Top 100 criptomonedas según Market Cap actualizado al {TODAY_STR}")

    # MKT CAP TABLE
    coins_by_market_cap = coins_by_market_cap.merge(coins_stats, on="symbol")

    st.markdown(
        f"Estadísticas obtenidas entre {INIT_DATE.strftime('%Y-%m-%d')} y {TODAY.strftime('%Y-%m-%d')}"
    )

    st.write(coins_by_market_cap)

    # VOLUME vs MKT CAP
    c = alt.Chart(coins_by_market_cap).mark_circle(size=100).encode(
        x=alt.X('mkt_cap:Q', title="Market Cap"),
        y=alt.Y('volume:Q', title="Volumen Total"),
        tooltip=["symbol:N"]
    ).interactive()

    st.altair_chart(c, use_container_width=True)

    # MEAN RETURN vs VOLATILITY
    c = alt.Chart(coins_stats).mark_circle(size=100).encode(
        x=alt.X('mean_return:Q', title="Retorno Promedio"),
        y=alt.Y('volatility:Q', title="Volatilidad"),
        color="sharpe_ratio:Q",
        tooltip=[
            alt.Tooltip("symbol:N", title="Coin"),
            alt.Tooltip('sharpe_ratio:Q', format='.3f', title="Sharpe Ratio")
        ]
    ).interactive()

    st.altair_chart(c, use_container_width=True)

    # CORRELATION WITH BTC AND S&P
    btc_corrs = coins_stats.sort_values(by="btc_corr", ascending=False)

    btc_corrs = pd.concat([btc_corrs.iloc[:10], btc_corrs.iloc[-10:]])

    c = alt.Chart(btc_corrs).mark_bar().encode(
        x=alt.X("symbol:N", title="Coin"),
        y=alt.Y("btc_corr:Q", title="Correlación con BTC"),
        color=alt.Color("spy_corr_2:Q", title="Correlación con SPY"),
        tooltip = [
            alt.Tooltip(
                "btc_corr:Q", format=".2f", title="Correlación con BTC"
            ),
            alt.Tooltip(
                "spy_corr_2:Q", format=".2f", title="Correlación con SPY"
            )
        ]
    )

    st.altair_chart(c, use_container_width=True)

    # BETA CAPM CRYPTO
    btc_beta = coins_stats.sort_values(by="beta_capm_crypto", ascending=False)

    btc_beta = pd.concat([btc_beta.iloc[:10], btc_beta.iloc[:-10]])

    c = alt.Chart(btc_corrs).mark_bar().encode(
        x=alt.X("symbol:N", title="Coin"),
        y=alt.Y("beta_capm_crypto:Q", title="Beta CAPM Crypto"),
        tooltip = [
            alt.Tooltip(
                "beta_capm_crypto:Q", format=".2f", title="Beta CAPM Crypto"
            )
        ]
    )

    st.altair_chart(c, use_container_width=True)

    #ON CHAIN ANALYSIS
    st.subheader("Métricas On-Chain")

    df = dm.load_coins_on_chain_metrics()

    st.dataframe(df.style.format(
        formatter={
            col: "{:.3f}" for col in df.columns.tolist() if col!= "symbol"
        }
    ))

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

    with st.expander("Descripción de las métricas"):

        metric_to_describe = st.selectbox(
            'Seleccionar una métrica',
            [d["id"] for d in METRICS_DESC],
        )

        describe_metric(metric_to_describe)

    # ANALYSIS BY COIN
    st.subheader("Análisis por criptomoneda")

    # DROPDOWN
    symbols = coins_by_market_cap.loc[:, "symbol"].tolist()
    symbols.sort()

    selected_symbol = st.selectbox(
        'Seleccionar una criptomoneda',
        symbols
    )

    # # DATE INPUT
    # MIN_VALUE = datetime(2017,1,1)
    # dates = st.date_input(
    #     "Seleccione el rango de fechas",
    #     [INIT_DATE, TODAY.date()],
    #     min_value=MIN_VALUE,
    #     max_value=datetime(TODAY.year, TODAY.month, TODAY.day)
    # )

    # DOWNLOADING PRICE DATA
    # coin = coins_by_market_cap.loc[
    #     coins_by_market_cap.loc[:, "symbol"] == coin,
    #     ["id", "symbol"]
    # ].values.ravel()

    # from_datetime = datetime(dates[0].year, dates[0].month, dates[0].day)

    # to_datetime = datetime(dates[1].year, dates[1].month, dates[1].day)

    with st.spinner("Descargando datos"):
        df, source = dm.get_coin_history(selected_symbol, INIT_DATE, TODAY)

    st.markdown(f"**Source**: {source}")

    # DAILY RETURNS STATS

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

if __name__ == "__main__":

    run_app()