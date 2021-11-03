import streamlit as st
from .finance import *
from .data_management import *
from .plotting import *
import altair as alt


alt.themes.enable("fivethirtyeight")
alt.renderers.set_embed_options(actions=False)

def run_app(cache_dict):

    coins_by_market_cap = cache_dict["coins_by_market_cap"]

    coins_stats = cache_dict["coins_stats"]

    sector_agg = cache_dict["sector_agg"]

    category_agg = cache_dict["category_agg"]

    portfolios_by_sector = cache_dict["portfolios_by_sector"]

    portfolios_by_category = cache_dict["portfolios_by_category"]

    # MKT CAP TABLE
    coins_by_market_cap = coins_by_market_cap.merge(coins_stats, on="symbol")


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
