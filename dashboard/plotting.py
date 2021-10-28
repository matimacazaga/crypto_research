import pandas as pd
import altair as alt
import numpy as np
from scipy.stats import norm
alt.renderers.set_embed_options(actions=False)

def make_volume_vs_mkt_cap_plot(coins_by_market_cap):
    c = alt.Chart(coins_by_market_cap).mark_circle(size=100).encode(
        x=alt.X('mkt_cap:Q', title="Market Cap"),
        y=alt.Y('volume:Q', title="Volumen Total"),
        tooltip=["symbol:N"]
    ).interactive()

    return c


def make_beta_capm_crypto_plot(btc_corrs):
    c = alt.Chart(btc_corrs).mark_bar().encode(
        x=alt.X("symbol:N", title="Coin"),
        y=alt.Y("beta_capm_crypto:Q", title="Beta CAPM Crypto"),
        tooltip = [
            alt.Tooltip(
                "beta_capm_crypto:Q", format=".2f", title="Beta CAPM Crypto"
            )
        ]
    )

    return c


def make_btc_spy_corr_plot(btc_corrs):
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

    return c


def make_mean_ret_vs_std_plot(coins_stats):
    c = alt.Chart(coins_stats).mark_circle(size=100).encode(
        x=alt.X('mean_return:Q', title="Retorno Promedio"),
        y=alt.Y('volatility:Q', title="Volatilidad"),
        color="sharpe_ratio:Q",
        tooltip=[
            alt.Tooltip("symbol:N", title="Coin"),
            alt.Tooltip('sharpe_ratio:Q', format='.3f', title="Sharpe Ratio")
        ]
    ).interactive()

    return c


def make_coin_plots(df:pd.DataFrame):

    interval = alt.selection_interval(encodings=['x'])

    price_base = alt.Chart(df).mark_line().encode(
        x=alt.X('date:T', title="Fecha"),
        y=alt.Y('price:Q', title="Precio"),
    ) + alt.Chart(df).mark_point().encode(
        x=alt.X('date:T', title="Fecha"),
        y=alt.Y('price:Q', title="Precio"),
        tooltip=[alt.Tooltip("date:T", title="Fecha"), alt.Tooltip("price:Q", title="Precio")]
    )

    price_chart = price_base.encode(
        x=alt.X("date:T", scale=alt.Scale(domain=interval.ref()))
    ).properties(
        width=600
    )

    price_view = price_base.add_selection(
        interval
    ).properties(
        height=50,
        width=600
    )

    return_base = alt.Chart(df).mark_line().encode(
        x=alt.X('date:T', title="Fecha"),
        y=alt.Y('return:Q', title="Retorno"),
    ) + alt.Chart(df).mark_point().encode(
        x=alt.X('date:T', title="Fecha"),
        y=alt.Y('return:Q', title="Retorno"),
        tooltip=[alt.Tooltip("date:T", title="Fecha"), alt.Tooltip('return:Q', format='.2%', title="Retorno")]
    )

    return_chart = return_base.encode(
        x=alt.X("date:T", scale=alt.Scale(domain=interval.ref()))
    ).properties(
        width=600
    )

    return_view = return_base.add_selection(
        interval
    ).properties(
        width=600,
        height=50,
    )

    return_dist = alt.Chart(df).transform_density(
        'return',
        as_=['return', 'density'],
    ).mark_area().encode(
        x=alt.X("return:Q", title="Retorno"),
        y=alt.Y('density:Q', title="PDF"),
    )

    ret_range = np.linspace(
        df.loc[:, "return"].min(),
        df.loc[:, "return"].max(),
        1000,
    )
    norm_ret = pd.DataFrame({
        "return": ret_range,
        "density": norm.pdf(
            ret_range,
            loc=df.loc[:, "return"].mean(),
            scale=df.loc[:, "return"].std(ddof=1)
        )
    })

    return_dist_norm = alt.Chart(norm_ret).mark_line(color="red").encode(
        x=alt.X("return:Q", title="Retorno"),
        y=alt.Y('density:Q', title="PDF"),
    )

    df_ = df.copy()

    df_.loc[:, "month"] = df.loc[:, "date"].dt.strftime("%B")

    boxplot = alt.Chart(df_).mark_boxplot().encode(
        x = alt.X(
            "month:N",
            sort=[
                "January", "February", "March", "April", "May", "June", "July",
                "August", "September", "October", "November", "December"
            ]
        ),
        y = "return:Q",
    )

    return price_chart & price_view, return_chart & return_view, return_dist + return_dist_norm, boxplot


def make_cluster_plot(df:pd.DataFrame):
    c = alt.Chart(df).mark_circle().encode(
        x = "PC1:Q",
        y = "PC2:Q",
        tooltip=[
            "symbol:N",
            alt.Tooltip("CapRealUSD:Q", format=",.0f", title="Realized Cap."),
            alt.Tooltip("TxCntSec:Q", format=".3f", title="TPS"),
            alt.Tooltip("CapMVRVCur:Q", format=".2f", title="Mkt. Cap. / Real. Cap."),
            alt.Tooltip("AdrActCnt:Q", format=",.0f", title="Average active addresses per day (15 days)")
        ],
        color="cluster:N",
        size=alt.Size("CapRealUSD:Q", scale=alt.Scale(type="log"), title="Realized Cap.")
    ).interactive()

    return c


def make_clusters_stats_plot(clusters_stats:pd.DataFrame):

    c = alt.Chart(clusters_stats).mark_point(size=100).encode(
        x=alt.X("mean_ret:Q", title="Retorno promedio (Anualizado)"),
        y=alt.Y("std:Q", title="Volatilidad (Anualizada)"),
        color=alt.Color("sharpe:Q", scale=alt.Scale(scheme="viridis")),
        shape="cluster:N",
        tooltip=[
            alt.Tooltip(
                "mean_ret:Q",
                title="Retorno promedio (Anualizado)",
                format=".2%"
            ),
            alt.Tooltip(
                "std:Q",
                title="Volatilidad (Anualizada)",
                format=".2%"
            ),
            alt.Tooltip(
                "sharpe:Q",
                title="Sharpe Ratio (Anualizado)",
                format=".2f"
            ),
            alt.Tooltip(
                "symbols",
                title="Simbolos"
            )
        ]
    ).properties(height=400)

    return c


def make_symbols_info_plots(sector_agg, category_agg):

    sector_plot = alt.Chart(sector_agg).mark_bar().encode(
        x=alt.X("sector:N", title="Sector"),
        y = alt.Y("count:Q", title="Numero de coins"),
        color = alt.Color("mkt_cap_pct:Q", scale=alt.Scale(scheme="turbo"), title="% de Market Cap."),
        tooltip = [
            alt.Tooltip("mkt_cap:Q", title="Mkt. Cap."),
            alt.Tooltip("mkt_cap_pct:Q", title="% de Mkt. Cap", format=".2%"),
            alt.Tooltip("symbols"),
        ],
    ).interactive()

    category_plot = alt.Chart(category_agg).mark_bar().encode(
        x=alt.X("category:N", title="Categoría"),
        y = alt.Y("count:Q", title="Numero de coins"),
        color = alt.Color("mkt_cap_pct:Q", scale=alt.Scale(scheme="turbo"), title="% de Market Cap."),
        tooltip = [
            alt.Tooltip("mkt_cap:Q", title="Mkt. Cap."),
            alt.Tooltip("mkt_cap_pct:Q", title="% de Mkt. Cap", format=".2%"),
            alt.Tooltip("symbols"),
        ],
    ).interactive()

    return sector_plot, category_plot


def make_portfolio_plots(portfolio:pd.DataFrame):

    temp = portfolio.drop(
        ["price", "return"],axis=1
    ).stack().reset_index().rename({"level_1": "symbol", 0: "price"}, axis=1)

    selection = alt.selection_multi(fields=["symbol"])

    c1 = alt.Chart(temp).mark_area(opacity=0.7).encode(
        x=alt.X("date:T", title="Fecha"),
        y=alt.Y("price:Q", title="Precio Portfolio"),
        color=alt.Color("symbol:N", scale=alt.Scale(scheme="sinebow")),
        tooltip=["price:Q", "symbol:N"]
    ).transform_filter(selection).interactive()

    legend = alt.Chart(temp).mark_point().encode(
    x=alt.X('symbol:N', axis=alt.Axis(orient='bottom')),
    color = alt.condition(selection,
                      alt.Color('symbol:N', legend=None),
                      alt.value('lightgray')),

    ).add_selection(
        selection
    )

    c1 = c1 & legend

    temp = portfolio.copy()
    temp.loc[:, "cum_ret"] = (1 + temp.loc[:, "return"]).cumprod()
    temp.loc[:, "const"] = 1
    temp.loc[:, "color"] = temp.loc[:, "cum_ret"].apply(lambda x: "orangered" if x<1. else "forestgreen")

    interval = alt.selection_interval(encodings=["x"])

    base = alt.Chart(temp.reset_index()).mark_area().encode(
        x=alt.X("date:T", title="Fecha"),
        y=alt.Y("cum_ret:Q", title="Retorno cumulativo", scale=alt.Scale(zero=False), impute={"value": 0}),
        y2="const:Q",
        color=alt.Color("color:N", scale=None),
        tooltip=[alt.Tooltip("date:T", title="Fecha"), alt.Tooltip("return:Q", title="Retorno", format=".2%")]
    )


    c2 = base.encode(
        x=alt.X("date:T", scale=alt.Scale(domain=interval.ref()))
    ).properties(
        height=300
    )

    view = base.add_selection(
        interval
    ).properties(
        height=50
    )

    return c1, c2 & view
