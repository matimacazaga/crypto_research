import streamlit as st
from .finance import *
from .data_management import *
from .plotting import *
import altair as alt


alt.themes.enable("fivethirtyeight")
alt.renderers.set_embed_options(actions=False)


def run_app(cache_dict):

    INIT_DATE = cache_dict["init_date"]

    TODAY = cache_dict["end_date"]

    dm = cache_dict["dm"]

    coins_by_market_cap = cache_dict["coins_by_market_cap"]

    coins_stats = cache_dict["coins_stats"]

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

    try:
        st.markdown(f"**Sector**: {coins_dict[selected_symbol.lower()]['sector'].capitalize()}")
    except:
        st.markdown('Sector not available')


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
