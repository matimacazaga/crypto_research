
from datetime import datetime
import pickle
import streamlit as st
from .utils import *
from .data_management import *
import altair as alt
import pandas as pd
from .config import *


METRICS_DESC = pickle.load(open("./dashboard/data/metrics_description", "rb"))

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
    with st.spinner("Descargando datos"):
        BTC, _ = get_coin_price(["bitcoin", "btc"], INIT_DATE, datetime(TODAY.year, TODAY.month, TODAY.day))
        BTC.set_index("date", inplace=True)
        SPY = get_spy_price(INIT_DATE, datetime(TODAY.year, TODAY.month, TODAY.day))
        coins_markets = get_coins_market_caps_cg()
        coins_by_market_cap = top_100_by_mkt_cap(coins_markets, COLUMNS)
        coins = coins_by_market_cap.loc[:, ["id", "symbol"]].values
        coins_stats = get_stats(coins, INIT_DATE, datetime(TODAY.year, TODAY.month, TODAY.day), SPY, BTC)

    # MAIN TITLE
    st.title(f"Top 100 criptomonedas según Market Cap actualizado al {TODAY_STR}")

    # MKT CAP TABLE
    coins_by_market_cap = coins_by_market_cap.merge(
        coins_stats, left_on="symbol", right_on="name", suffixes=[None, "_y"]
    ).drop("name_y", axis=1)

    st.markdown(
        f"Estadísticas obtenidas entre {INIT_DATE.strftime('%Y-%m-%d')} y {TODAY.strftime('%Y-%m-%d')}"
    )

    st.write(coins_by_market_cap)

    # VOLUME vs MKT CAP
    c = alt.Chart(coins_by_market_cap).mark_circle(size=100).encode(
        x=alt.X('market_cap:Q', title="Market Cap"),
        y=alt.Y('total_volume:Q', title="Volumen Total"),
        tooltip=["name:N"]
    ).interactive()

    st.altair_chart(c, use_container_width=True)

    # MEAN RETURN vs VOLATILITY
    c = alt.Chart(coins_stats).mark_circle(size=100).encode(
        x=alt.X('mean_return:Q', title="Retorno Promedio"),
        y=alt.Y('volatility:Q', title="Volatilidad"),
        color="sharpe_ratio:Q",
        tooltip=[
            alt.Tooltip("name:N", title="Coin"),
            alt.Tooltip('sharpe_ratio:Q', format='.3f', title="Sharpe Ratio")
        ]
    ).interactive()

    st.altair_chart(c, use_container_width=True)

    # CORRELATION WITH BTC AND S&P
    btc_corrs = coins_stats.sort_values(by="btc_corr", ascending=False)

    btc_corrs = pd.concat([btc_corrs.iloc[:10], btc_corrs.iloc[-10:]])

    c = alt.Chart(btc_corrs).mark_bar().encode(
        x=alt.X("name:N", title="Coin"),
        y=alt.Y("btc_corr:Q", title="Correlación con BTC"),
        color="spy_corr:Q",
        tooltip = [
            alt.Tooltip(
                "btc_corr:Q", format=".2f", title="Correlación con BTC"
            ),
            alt.Tooltip(
                "spy_corr:Q", format=".2f", title="Correlación con SPY"
            )
        ]
    )

    st.altair_chart(c, use_container_width=True)

    # BETA CAPM CRYPTO
    btc_beta = coins_stats.sort_values(by="beta_capm_crypto", ascending=False)

    btc_beta = pd.concat([btc_beta.iloc[:10], btc_beta.iloc[:-10]])

    c = alt.Chart(btc_corrs).mark_bar().encode(
        x=alt.X("name:N", title="Coin"),
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

    df = load_coins_on_chain_metrics()

    st.dataframe(df.style.format(
        formatter={
            col: "{:.3f}" for col in df.columns.tolist() if col!= "symbol"
        }
    ))

    df = load_cluster_df()

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
    names = coins_by_market_cap.loc[:, "name"].tolist()
    names.sort()

    coin = st.selectbox(
            'Seleccionar una criptomoneda',
        names
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
    coin = coins_by_market_cap.loc[
        coins_by_market_cap.loc[:, "name"] == coin,
        ["id", "symbol"]
    ].values.ravel()

    # from_datetime = datetime(dates[0].year, dates[0].month, dates[0].day)

    # to_datetime = datetime(dates[1].year, dates[1].month, dates[1].day)

    with st.spinner("Descargando datos"):
        df, source = get_coin_price(coin, INIT_DATE, datetime(TODAY.year, TODAY.month, TODAY.day))

    st.markdown(f"**Source**: {source}")

    # DAILY RETURNS STATS

    st.markdown("### Daily returns statistics")

    coin_stats = get_coin_stats(df)

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