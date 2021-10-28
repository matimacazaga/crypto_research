import pandas as pd
import numpy as np
from datetime import datetime
import altair as alt
from joblib import Parallel, delayed
#from .data_management import get_coin_price
import pickle
import os
import yfinance as yf
from scipy import stats
from scipy.stats import norm

def top_100_by_mkt_cap(coins_markets:dict, columns:list)->pd.DataFrame:

    coins_market_cap = pd.DataFrame(coins_markets)

    coins_market_cap.sort_values("market_cap_rank", ascending=True, inplace=True)

    return coins_market_cap.loc[:, columns]

def compute_coin_stats(coin_history:tuple, spy:pd.DataFrame, btc:pd.DataFrame):
    """
    Computes coin's statistics:
        - Correlation with spy and btc
        - Mean return, volatility, Sharpe ratio and Sortino ratio.

    Parameters
    ----------
    coin_history: tuple[str, tuple[pd.Dataframe, str]]
        Tuple containing the symbol of the coin in the first element, and
        another tuple in the second element. The tuple in the second element
        has the coin history in the first element and the data source in the
        second element.
    spy: pd.Dataframe
        History of the SPY etf.
    btc: pd.DataFrame
        History of BTC.

    Returns
    -------
    output: dict
        Dictionary containing the statistics and name of the coin.
    """
    df = coin_history[1][0].copy()

    # junto los dataframes en un nuevo dataframe
    result = df.merge(spy, on="date", how="outer", suffixes=["", "_spy"])
    result = result.merge(btc, on="date", how="outer", suffixes=["", "_btc"])
    result['agg_ret'] = result.loc[:,"return"]

    mask_1 = result.shift(1).loc[:, "return_spy"].isna()
    mask_2 = result.shift(2).loc[:, "return_spy"].isna()

    result.loc[mask_1 & mask_2,'agg_ret'] = result.loc[:,"return"] +\
         result.shift(1).loc[:,"return"] +\
              result.shift(2).loc[:,"return"]

    spy_corr = result.loc[:, "return"].corr(result.loc[:, "return_spy"])
    spy_corr_2 = result.loc[:,"agg_ret"].corr(result.loc[:,"return_spy"])
    btc_corr = result.loc[:, "return"].corr(result.loc[:, "return_btc"])


    slope, _, _, _, _ = stats.linregress(
        result.dropna(subset=["return", "return_btc"]).loc[:, "return_btc"].values,
        result.dropna(subset=["return", "return_btc"]).loc[:, "return"].values
    )

    mean_return = result.loc[:, "return"].mean() * 365.

    volatility = result.loc[:, "return"].std(ddof=1) * np.sqrt(365.)

    sharpe_ratio = (mean_return / volatility)

    dd = result.loc[result.loc[:, 'return'] < 0,'return'].std() * np.sqrt(365.)

    sortino_ratio = (mean_return / dd)

    output = {
        "symbol": coin_history[0],
        "spy_corr": spy_corr,
        "spy_corr_2": spy_corr_2,
        "btc_corr": btc_corr,
        "beta_capm_crypto": slope,
        "mean_return": mean_return,
        "volatility": volatility,
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio
    }

    return output


# def compute_stats(coin, from_datetime, to_datetime, spy:pd.DataFrame, btc:pd.DataFrame):

#     coin_data, _ = get_coin_price(coin, from_datetime, to_datetime)

#     df = coin_data.set_index("date")

#     df.index = pd.to_datetime(df.index)

#     # junto los dataframes en un nuevo dataframe
#     result = pd.concat([df, spy], axis=1)

#     # creo la nueva columna de aggregate returns y la igualo al retorno del dia
#     result['agg_ret'] = result.iloc[:,2]

#     # si los datos para el spy de los dos dias anteriores son NaN, el agg_ret pasa a ser la suma de las ultimas 3 observaciones
#     result.loc[pd.isna(result.shift(1).iloc[:,3]) & pd.isna(result.shift(2).iloc[:,3]),
#      'agg_ret'] = result.iloc[:,2] + result.shift(1).iloc[:,2] + result.shift(2).iloc[:,2]

#     result = result.dropna()

#     # encuentro las correlaciones
#     spy2_corr = result.iloc[:,4].astype(float).corr(result.iloc[:,5].astype(float))

#     dates_spy = spy.dropna().index.intersection(df.dropna().index)

#     dates_btc = btc.dropna().index.intersection(df.dropna().index)

#     spy_corr = np.corrcoef([df.loc[dates_spy, "return"].values, spy.loc[dates_spy, "return"].values])[0,1]

#     btc_corr = np.corrcoef([df.loc[dates_btc, "return"].values, btc.loc[dates_btc, "return"].values])[0,1]

#     slope, _, _, _, _ = stats.linregress(btc.loc[dates_btc, "return"].values, df.loc[dates_btc, "return"].values)

#     mean_return = coin_data.loc[:, "return"].mean() * 365.

#     volatility = coin_data.loc[:, "return"].std(ddof=1) * np.sqrt(365.)

#     sharpe_ratio = (mean_return / volatility)

#     dd = df[df['return'] < 0]['return'].std() * np.sqrt(365.)

#     sortino_ratio = (mean_return / dd)

#     return {"name": coin[1], "spy_corr": spy_corr, "spy2_corr": spy2_corr, "btc_corr": btc_corr, "beta_capm_crypto": slope, "mean_return": mean_return, "volatility": volatility, "sharpe_ratio": sharpe_ratio, "sortino_ratio": sortino_ratio}

def get_coins_stats(coins_history:dict, spy:pd.DataFrame, btc:pd.DataFrame)->pd.DataFrame:

    # stats = Parallel(n_jobs=2)(delayed(compute_coin_stats)(coin_history, spy, btc) for coin_history in coins_history.items())

    stats = []
    for coin_history in coins_history.items():
        stats.append(
            compute_coin_stats(coin_history, spy, btc)
        )
    return pd.DataFrame(stats)

# def get_stats(coins:list, from_datetime:str, to_datetime:str, spy:pd.DataFrame, btc:pd.DataFrame):

#     stats = Parallel(n_jobs=-1, backend="threading")(delayed(compute_stats)(coin, from_datetime, to_datetime, spy, btc) for coin in coins)

#     return pd.DataFrame(stats)

def get_coin_returns_stats(df:pd.DataFrame)->pd.DataFrame:

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



def get_portfolios_by_col(coins_history:dict, df_agg:pd.DataFrame, col:str):
    """
    Get portfolios by sector or category.

    Parameters
    ----------
    coins_history: dict
        Dictionary with the history of each coin (price and mkt cap).
    df_agg: DataFrame
        DataFrame with the aggregated information by col.
    col: str
        Name of the col used to group the data.

    Returns
    -------
    portfolios: dict
        Dictionary with the groups as keys and DataFrames as values containing
        portfolio price, return and mkt_cap weighted price of its components.
    """
    portfolios = {}

    for _, row in df_agg.iterrows():
        if row["count"]>1:
            mkt_caps = []
            prices = []
            for symbol in row["symbols"]:
                if symbol not in coins_history:
                    continue

                mkt_caps.append(coins_history[symbol][0].rename({"mkt_cap": symbol}, axis=1).set_index("date").loc[:, symbol])
                prices.append(coins_history[symbol][0].rename({"price": symbol}, axis=1).set_index("date").loc[:, symbol])
            mkt_caps = pd.concat(mkt_caps, axis=1)
            prices = pd.concat(prices, axis=1)
            portfolio = (mkt_caps.divide(mkt_caps.sum(axis=1), axis=0).shift(1).iloc[1:] * prices.iloc[1:]).fillna(0.)
            portfolio.loc[:, "price"] = portfolio.sum(axis=1)
            #TODO: add log returns
            portfolio.loc[:, "return"] = portfolio.loc[:, "price"].pct_change()
            portfolios[row[col]] = portfolio
        else:
            portfolio = coins_history[row["symbols"][0]][0].loc[:, ["date", "price"]]
            #TODO: add log returns
            portfolio.loc[:, "return"] = portfolio.loc[:, "price"].pct_change()
            portfolios[row[col]] = portfolio

    return portfolios


def get_portfolios_stats(portfolios:dict, colname:str):
    dfs = []

    for col, portfolio in portfolios.items():

        mean_return = portfolio.loc[:, "return"].mean()*365
        volatility = portfolio.loc[:, "return"].std()*np.sqrt(365)
        sharpe_ratio = mean_return / volatility
        dd = portfolio.loc[portfolio.loc[:, 'return'] < 0, "return"].std() * np.sqrt(365.)
        sortino_ratio = mean_return / dd

        cum_ret = (1. + portfolio.loc[:, "return"]).cumprod()
        peak = cum_ret.expanding(min_periods=1).max()
        max_drawdown = ((cum_ret/peak) - 1.).min()
        var_5 = portfolio.loc[:, "return"].quantile(0.05)

        dfs.append(
            {
                colname.capitalize(): col,
                "Retorno Promedio (Anualiz.)": mean_return,
                "Volatilidad (Anualiz.)": volatility,
                "Sharpe Ratio (Anualiz.)": sharpe_ratio,
                "Sortino Ratio (Anualiz.)": sortino_ratio,
                "Max Drawdown": max_drawdown,
                "Historical VaR 5%": var_5
            }
        )

    return pd.DataFrame(dfs)


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
        x="date:T",
        y="price:Q",
        color=alt.Color("symbol:N", scale=alt.Scale(scheme="sinebow")),
        tooltip=["price:Q", "symbol:N"]
    ).transform_filter(selection)

    legend = alt.Chart(temp).mark_point().encode(
    y=alt.Y('symbol:N', axis=alt.Axis(orient='right')),
    color = alt.condition(selection,
                      alt.Color('symbol:N', legend=None),
                      alt.value('lightgray'))
    ).add_selection(
        selection
    )

    c1 = c1 | legend

    temp = portfolio.copy()
    temp.loc[:, "cum_ret"] = (1 + temp.loc[:, "return"]).cumprod()
    temp.loc[:, "const"] = 1
    c2 = alt.Chart(temp[temp["cum_ret"] > 1.].reset_index()).mark_bar(
        color="green",
    ).encode(
        x=alt.X("date:T", title="Fecha"),
        y=alt.Y("cum_ret:Q", title="Retorno cumulativo", scale=alt.Scale(zero=False)),
        y2="const:Q",
        tooltip=[alt.Tooltip("date:T", title="Fecha"), alt.Tooltip("return:Q", title="Retorno", format=".2%")]
    ).interactive() + alt.Chart(temp[temp["cum_ret"] < 1.].reset_index()).mark_bar(
        color="red",
    ).encode(
        x=alt.X("date:T", title="Fecha"),
        y=alt.Y("cum_ret:Q", title="Retorno cumulativo", scale=alt.Scale(zero=False)),
        y2="const:Q",
        tooltip=[alt.Tooltip("date:T", title="Fecha"), alt.Tooltip("return:Q", title="Retorno", format=".2%")]
    ).interactive()

    return c1, c2
