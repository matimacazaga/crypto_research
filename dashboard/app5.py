import streamlit as st
import altair as alt
from .config import METRICS_FOR_ANALYSIS
import numpy as np
from .factor_analysis import get_clean_factor_and_forward_returns, analyze_factor

alt.themes.enable("fivethirtyeight")
alt.renderers.set_embed_options(actions=False)


def run_app(cache_dict):

    symbols_metrics = cache_dict["symbols_metrics"]

    st.header("Factor Analysis")

    factor = st.selectbox(
        'Seleccionar factor',
        list(filter(lambda x: x!="PriceUSD", METRICS_FOR_ANALYSIS))
    )

    init=False

    for symbol, df in symbols_metrics.items():

        if factor in df.columns and "PriceUSD" in df.columns:
            if not init:
                factor_df = df.loc[:, ["date", factor]].rename(
                    {factor: symbol}, axis=1
                )
                price_df = df.loc[:, ["date", "PriceUSD"]].rename(
                    {"PriceUSD": symbol}, axis=1
                )
                init=True
            else:
                factor_df = factor_df.merge(
                    df.loc[:, ["date", factor]].rename(
                        {factor: symbol}, axis=1
                    ), on="date", how="outer"
                )
                price_df = price_df.merge(
                    df.loc[:, ["date", "PriceUSD"]].rename(
                        {"PriceUSD": symbol}, axis=1
                    ), on="date", how="outer"
                )

    factor_df = factor_df.replace(
        {None: np.nan}
    ).astype({col: float for col in factor_df.columns if col != "date"})

    price_df = price_df.replace(
        {None: np.nan}
    ).astype({col: float for col in price_df.columns if col != "date"})

    factor_data = get_clean_factor_and_forward_returns(
        factor_df.set_index("date").stack(),
        price_df.set_index("date"),
        quantiles=5,
        periods=(1, 5, 10, 15)
    )

    returns_table, quantiles_returns_chart, cum_ret_by_quantile_chart, spread_return_chart = analyze_factor(factor_data)

    st.subheader("Análisis de los retornos")

    st.markdown(f"Se considera un total de {factor_df.shape[1]} coins.")

    st.dataframe(returns_table)

    st.altair_chart(quantiles_returns_chart, use_container_width=True)

    st.altair_chart(cum_ret_by_quantile_chart, use_container_width=True)

    st.altair_chart(spread_return_chart, use_container_width=True)
