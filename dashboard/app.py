
from datetime import datetime
from dateutil.relativedelta import relativedelta
import streamlit as st
from .utils import top_100_by_mkt_cap, get_stats, make_coin_plots
from .data_management import get_coin_prices, get_spy_prices, get_coins_markets_cg
import altair as alt
import pandas as pd

def run_app():
    COLUMNS = [
        "id",
        "symbol",
        "name",
        "market_cap",
        "market_cap_rank",
        "current_price",
        "total_volume",
        "circulating_supply",
        "max_supply"
    ]

    TODAY = datetime.now()
    TODAY_STR = TODAY.strftime("%Y-%m-%d %H:%M:%S")

    INIT_DATE = TODAY - relativedelta(years=2)

    INIT_DATE = datetime(INIT_DATE.year, INIT_DATE.month, INIT_DATE.day)

    BTC, _ = get_coin_prices(["bitcoin", "btc"], INIT_DATE, datetime(TODAY.year, TODAY.month, TODAY.day))
    BTC.set_index("date", inplace=True)

    SPY = get_spy_prices(INIT_DATE, datetime(TODAY.year, TODAY.month, TODAY.day))

    with st.spinner("Descargando datos"):
        coins_markets = get_coins_markets_cg()
        coins_by_market_cap = top_100_by_mkt_cap(coins_markets, COLUMNS)
        coins = coins_by_market_cap.loc[:, ["id", "symbol"]].values
        coins_stats = get_stats(coins, INIT_DATE, datetime(TODAY.year, TODAY.month, TODAY.day), SPY, BTC)


    st.title(f"Top 100 criptomonedas según Market Cap actualizado al {TODAY_STR}")

    coins_by_market_cap = coins_by_market_cap.merge(coins_stats, left_on="symbol", right_on="name", suffixes=[None, "_y"]).drop("name_y", axis=1)

    st.markdown(f"Estadísticas obtenidas entre {INIT_DATE.strftime('%Y-%m-%d')} y {TODAY.strftime('%Y-%m-%d')}")
    st.write(coins_by_market_cap)

    c = alt.Chart(coins_by_market_cap).mark_circle(size=100).encode(
        x=alt.X('market_cap:Q', title="Market Cap"), y=alt.Y('total_volume:Q', title="Volumen Total"), tooltip=["name:N"]
    ).interactive()

    st.altair_chart(c, use_container_width=True)

    c = alt.Chart(coins_stats).mark_circle(size=100).encode(
        x=alt.X('mean_return:Q', title="Retorno Promedio"), y=alt.Y('volatility:Q', title="Volatilidad"), color="sharpe_ratio:Q", tooltip=[alt.Tooltip("name:N", title="Coin"), alt.Tooltip('sharpe_ratio:Q', format='.3f', title="Sharpe Ratio")]
    ).interactive()

    st.altair_chart(c, use_container_width=True)

    btc_corrs = coins_stats.sort_values(by="btc_corr", ascending=False)

    btc_corrs = pd.concat([btc_corrs.iloc[:10], btc_corrs.iloc[-10:]])

    c = alt.Chart(btc_corrs).mark_bar().encode(
        x=alt.X("name:N", title="Coin"), y=alt.Y("btc_corr:Q", title="Correlación con BTC"), color="spy_corr:Q", tooltip = [alt.Tooltip("btc_corr:Q", format=".2f", title="Correlación con BTC"), alt.Tooltip("spy_corr:Q", format=".2f", title="Correlación con SPY")]
    )

    st.altair_chart(c, use_container_width=True)

    btc_beta = coins_stats.sort_values(by="beta_capm", ascending=False)

    btc_beta = pd.concat([btc_beta.iloc[:10], btc_beta.iloc[:-10]])

    c = alt.Chart(btc_corrs).mark_bar().encode(
        x=alt.X("name:N", title="Coin"), y=alt.Y("beta_capm:Q", title="Beta CAPM"), tooltip = [alt.Tooltip("beta_capm:Q", format=".2f", title="Beta CAPM")]
    )

    st.altair_chart(c, use_container_width=True)

    st.subheader("Análisis por criptomoneda")

    MIN_VALUE = datetime(2013,1,1)

    names = coins_by_market_cap.loc[:, "name"].tolist()
    names.sort()

    coin = st.selectbox(
            'Seleccionar una criptomoneda',
        names
    )

    dates = st.date_input(
        "Seleccione el rango de fechas",
        [datetime(2019, 1, 1), TODAY.date()],
        min_value=MIN_VALUE,
        max_value=datetime(TODAY.year, TODAY.month, TODAY.day)
    )

    coin = coins_by_market_cap.loc[
        coins_by_market_cap.loc[:, "name"] == coin,
        ["id", "symbol"]
    ].values.ravel()

    from_datetime = datetime(dates[0].year, dates[0].month, dates[0].day)

    to_datetime = datetime(dates[1].year, dates[1].month, dates[1].day)

    df, source = get_coin_prices(coin, from_datetime, to_datetime)

    st.markdown(f"**Source**: {source}")

    st.dataframe(df, width=660)

    price_chart, return_chart, return_dist = make_coin_plots(df)

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