
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

    # MAIN TITLE
    st.title(f"Top 100 criptomonedas según Market Cap actualizado al {TODAY_STR}")

    # MKT CAP TABLE
    coins_by_market_cap = coins_by_market_cap.merge(coins_stats, on="symbol")

    st.markdown(
        f"Estadísticas obtenidas entre {INIT_DATE.strftime('%Y-%m-%d')} y {TODAY.strftime('%Y-%m-%d')}"
    )

    st.write(coins_by_market_cap)

    # VOLUME vs MKT CAP
    c = make_volume_vs_mkt_cap_plot(coins_by_market_cap)

    st.altair_chart(c, use_container_width=True)

    # MEAN RETURN vs VOLATILITY
    c = make_mean_ret_vs_std_plot(coins_stats)

    st.altair_chart(c, use_container_width=True)

    # CORRELATION WITH BTC AND S&P
    btc_corrs = coins_stats.sort_values(by="btc_corr", ascending=False)

    btc_corrs = pd.concat([btc_corrs.iloc[:10], btc_corrs.iloc[-10:]])

    c = make_btc_spy_corr_plot(btc_corrs)

    st.altair_chart(c, use_container_width=True)

    # BETA CAPM CRYPTO
    btc_beta = coins_stats.sort_values(by="beta_capm_crypto", ascending=False)

    btc_beta = pd.concat([btc_beta.iloc[:10], btc_beta.iloc[:-10]])

    c = make_beta_capm_crypto_plot(btc_corrs)

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

    # SECTOR ANALYSIS
    st.subheader("Portfolios por sectores y categorías")

    st.markdown("Source: [Messari](https://messari.io/)")

    sector_plot, category_plot = make_symbols_info_plots(sector_agg, category_agg)

    st.markdown("### Coins por sector")

    st.altair_chart(sector_plot, use_container_width=True)

    st.markdown("### Coins por cateogría")

    st.altair_chart(category_plot, use_container_width=True)

    st.markdown("### Análisis de portfolios por sector")

    sector_portfolios_stats = get_portfolios_stats(portfolios_by_sector, "sector")

    st.markdown("Estadísticas de los retornos de portfolios por sector")

    st.markdown("Nota: se utiliza Mkt. Cap. como peso")

    st.dataframe(sector_portfolios_stats.style.format({
        "Volatilidad (Anualiz.)": "{:.2%}",
        "Retorno Promedio (Anualiz.)": "{:.2%}",
        "Max Drawdown": "{:.2%}",
        "Historical VaR 5%": "{:.2%}",
        "Sortino Ratio (Anualiz.)": "{:.2f}",
        "Sharpe Ratio (Anualiz.)": "{:.2f}"
    }))

    selected_sector = st.selectbox(
        'Seleccionar un sector',
        list(portfolios_by_sector.keys())
    )

    portfolio = portfolios_by_sector[selected_sector]

    portfolio_composition, portfolio_cum_ret = make_portfolio_plots(portfolio)

    st.altair_chart(portfolio_composition, use_container_width=True)

    st.altair_chart(portfolio_cum_ret, use_container_width=True)

    st.markdown("### Análisis de portfolios por categoría")

    category_portfolios_stats = get_portfolios_stats(portfolios_by_sector, "category")

    st.markdown("Nota: se utiliza Mkt. Cap. como peso")

    st.markdown("Estadísticas de los retornos de portfolios por cateogría")

    st.dataframe(category_portfolios_stats.style.format({
        "Volatilidad (Anualiz.)": "{:.2%}",
        "Retorno Promedio (Anualiz.)": "{:.2%}",
        "Max Drawdown": "{:.2%}",
        "Historical VaR 5%": "{:.2%}",
        "Sortino Ratio (Anualiz.)": "{:.2f}",
        "Sharpe Ratio (Anualiz.)": "{:.2f}"
    }))

    selected_sector = st.selectbox(
        'Seleccionar una categoría',
        list(portfolios_by_category.keys())
    )

    portfolio = portfolios_by_category[selected_sector]

    portfolio_composition, portfolio_cum_ret = make_portfolio_plots(portfolio)

    st.altair_chart(portfolio_composition, use_container_width=True)

    st.altair_chart(portfolio_cum_ret, use_container_width=True)

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