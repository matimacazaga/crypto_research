import pandas as pd
import numpy as np
from datetime import datetime
import altair as alt
from joblib import Parallel, delayed
from .data_management import get_coin_price
import pickle
import os
import yfinance as yf
from scipy import stats
from scipy.stats import norm

def top_100_by_mkt_cap(coins_markets:dict, columns:list)->pd.DataFrame:

    coins_market_cap = pd.DataFrame(coins_markets)

    coins_market_cap.sort_values("market_cap_rank", ascending=True, inplace=True)

    return coins_market_cap.loc[:, columns]

def compute_stats(coin, from_datetime, to_datetime, spy:pd.DataFrame, btc:pd.DataFrame):

    coin_data, _ = get_coin_price(coin, from_datetime, to_datetime)

    df = coin_data.set_index("date")

    df.index = pd.to_datetime(df.index)

    dates_spy = spy.dropna().index.intersection(df.dropna().index)

    dates_btc = btc.dropna().index.intersection(df.dropna().index)

    spy_corr = np.corrcoef([df.loc[dates_spy, "return"].values, spy.loc[dates_spy, "return"].values])[0,1]

    btc_corr = np.corrcoef([df.loc[dates_btc, "return"].values, btc.loc[dates_btc, "return"].values])[0,1]

    slope, _, _, _, _ = stats.linregress(btc.loc[dates_btc, "return"].values, df.loc[dates_btc, "return"].values)

    mean_return = coin_data.loc[:, "return"].mean() * 365.

    volatility = coin_data.loc[:, "return"].std(ddof=1) * np.sqrt(365.)

    sharpe_ratio = (mean_return / volatility)

    return {"name": coin[1], "spy_corr": spy_corr, "btc_corr": btc_corr, "beta_capm_crypto": slope, "mean_return": mean_return, "volatility": volatility, "sharpe_ratio": sharpe_ratio}

def get_stats(coins:list, from_datetime:str, to_datetime:str, spy:pd.DataFrame, btc:pd.DataFrame):

    stats = Parallel(n_jobs=20, backend="threading")(delayed(compute_stats)(coin, from_datetime, to_datetime, spy, btc) for coin in coins)

    return pd.DataFrame(stats)

def get_coin_stats(df:pd.DataFrame)->pd.DataFrame:

    coin_stats = {}

    for i in [7, 30, 90]:

        df_ = df.iloc[-i:]["return"]

        coin_stats[f"Last {i} days"] = {
            "Mean": df_.mean(),
            "Volatility": df_.std(ddof=1),
            "Min": df_.min(),
            "Max": df_.max(),
            "1Q": df_.quantile(0.25),
            "3Q": df_.quantile(0.75),
        }

    coin_stats["Historical"] = {
        "Mean": df.loc[:, "return"].mean(),
        "Volatility": df.loc[:, "return"].std(ddof=1),
        "Min": df.loc[:, "return"].min(),
        "Max": df.loc[:, "return"].max(),
        "1Q": df.loc[:, "return"].quantile(0.25),
        "3Q": df.loc[:, "return"].quantile(0.75),
    }

    coin_stats = pd.DataFrame(coin_stats)

    return coin_stats

def make_coin_plots(df):

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

    return price_chart & price_view, return_chart & return_view, return_dist + return_dist_norm