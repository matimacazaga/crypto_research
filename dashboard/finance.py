import pandas as pd
import numpy as np
from scipy import stats

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

    try:
        slope, _, _, _, _ = stats.linregress(
            result.dropna(subset=["return", "return_btc"]).loc[:, "return_btc"].values,
            result.dropna(subset=["return", "return_btc"]).loc[:, "return"].values
        )

    except ValueError:
        print(coin_history[0])
        slope = np.nan

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


def get_coins_stats(coins_history:dict, spy:pd.DataFrame, btc:pd.DataFrame)->pd.DataFrame:

    # stats = Parallel(n_jobs=2)(delayed(compute_coin_stats)(coin_history, spy, btc) for coin_history in coins_history.items())

    stats = []
    for coin_history in coins_history.items():
            stats.append(
                compute_coin_stats(coin_history, spy, btc)
            )
    return pd.DataFrame(stats)


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
            if row["symbols"][0] not in coins_history:
                continue
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

