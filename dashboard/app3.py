
import pickle
import streamlit as st

from .finance import *
from .data_management import *
from .plotting import *
import altair as alt

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

def run_app(cache_dict):

    dm = cache_dict["dm"]

    coins_by_market_cap = cache_dict["coins_by_market_cap"]

    coins_stats = cache_dict["coins_stats"]

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


