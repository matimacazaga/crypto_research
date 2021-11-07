import pickle
import streamlit as st
from .finance import *
from .data_management import *
from .plotting import *
import altair as alt
import pandas as pd

alt.themes.enable("fivethirtyeight")
alt.renderers.set_embed_options(actions=False)


def run_app(cache_dict):

    TODAY_STR = cache_dict["end_date_str"]

    INIT_DATE_STR = cache_dict["init_date_str"]

    coins_by_market_cap = cache_dict["coins_by_market_cap"]

    coins_stats = cache_dict["coins_stats"]

    # MAIN TITLE
    st.title(f"Top 100 criptomonedas según Market Cap actualizado al {TODAY_STR}")

    # MKT CAP TABLE
    coins_by_market_cap = coins_by_market_cap.merge(coins_stats, on="symbol")

    st.markdown(
        f"Estadísticas obtenidas entre {INIT_DATE_STR} y {TODAY_STR}"
    )

    st.write(coins_by_market_cap)

    DATA_DESC = pickle.load(open(f"{BASE_PATH}/data_description", "rb"))

    def describe_metric(metric_to_describe:str)->None:

        d = [d for d in DATA_DESC if d["id"] == metric_to_describe][0]

        st.markdown(
            f"**Description**: {d['desc']}"
        )

    with st.expander("Descripción de las métricas"):

        metric_to_describe = st.selectbox(
            'Seleccionar una métrica',
            [d["id"] for d in DATA_DESC],
        )

        describe_metric(metric_to_describe)


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

