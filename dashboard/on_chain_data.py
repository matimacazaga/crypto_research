from dateutil.relativedelta import relativedelta
from datetime import datetime
import coinmetrics
from joblib.parallel import delayed
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from yellowbrick.cluster.elbow import KElbowVisualizer
from sklearn.decomposition import PCA
import pandas as pd

from dashboard.config import INIT_DATE
from data_management import get_coin_price_coingecko
import time

COINMETRICS = coinmetrics.Community()
END_DATE = datetime.now()
INIT_DATE_INTERVAL = END_DATE - relativedelta(days=15)
INIT_DATE_AGG = END_DATE - relativedelta(days=1)
INIT_STR_INTERVAL = INIT_DATE_INTERVAL.strftime("%Y-%m-%d")
INIT_STR_AGG = INIT_DATE_AGG.strftime("%Y-%m-%d")
END_STR = END_DATE.strftime("%Y-%m-%d")

INTERVAL_METRICS = [
    "AdrActCnt", "TxTfrValMeanUSD",
    "CapMVRVCur" ,"CapMVRVFF", "CapMrktFFUSD", "TxCntSec", "NVTAdjFF",
    "CapRealUSD", "IssContPctAnn", "RevUSD", "FeeMeanUSD", "FeeMedUSD"
]
AGG_METRICS = [
    "NVTAdjFF90", "NVTAdj90", "ROI30d", "VtyDayRet30d", "RevAllTimeUSD",
    "BlkCnt", "VelCur1yr"
]

SUPPORTED_ASSETS = COINMETRICS.get_supported_assets()

def get_metrics_info(metrics:list)->str:
    res = COINMETRICS.get_metric_info(','.join(metrics))
    return res


def get_coin_metrics(coin:str, aggregated_metrics:list, interval_metrics:list,
    init_date_agg:str, init_date_int:str, end_date:str)->dict:

    metrics = COINMETRICS.get_available_data_types_for_asset(coin)

    intersection = set(metrics).intersection(interval_metrics+aggregated_metrics)

    if intersection:
        agg_metrics = [m for m in intersection if m in aggregated_metrics]
        int_metrics = [m for m in intersection if m in interval_metrics]

        metrics = {}

        if int_metrics:
            int_metrics = COINMETRICS.get_asset_data_for_time_range(
                coin, ",".join(int_metrics), init_date_int, end_date
            )
            if int_metrics["series"]:
                for m, metric in enumerate(int_metrics["metrics"]):
                    values = [
                        float(
                            d["values"][m]
                        ) for d in int_metrics["series"] if d["values"][m]
                    ]
                    metrics[metric] = np.mean(values) if values else None

        if agg_metrics:
            agg_metrics = COINMETRICS.get_asset_data_for_time_range(
                coin, ",".join(agg_metrics), init_date_agg, end_date
            )
            if agg_metrics["series"]:
                values = [
                    float(d) if d != None else d for d in agg_metrics["series"][0]["values"]
                ]
                for m, v in zip(agg_metrics["metrics"],values):
                    metrics[m] = v

        metrics["symbol"] = coin

    else:
        metrics["symbol"] = coin

    return metrics

def perform_clustering(coins_metrics:pd.DataFrame, col_thresh:int=55):

    df = coins_metrics.dropna(axis=1, thresh=col_thresh).dropna(axis=0, how="any")

    scaler = StandardScaler()
    X = df.drop("symbol", axis=1)
    X_scaled = scaler.fit_transform(X)
    model = KMeans()
    vis = KElbowVisualizer(
        model, k=(4,12), timings=False
    )
    vis.fit(X_scaled)
    k_opt = vis.elbow_value_

    model = KMeans(n_clusters=k_opt).fit(X_scaled)

    df.loc[:, "cluster"] = model.predict(X_scaled)

    pca = PCA(n_components=2)

    pca.fit(X_scaled)

    X_scaled_reduced = pca.transform(X_scaled)

    df.loc[:, "PC1"] = X_scaled_reduced[:, 0]

    df.loc[:, "PC2"] = X_scaled_reduced[:, 1]

    return df

def get_prices_and_market_caps(coins:list, from_datetime:datetime, to_datetime:datetime)->list:

    def get_data(coin):

        df, _ = get_coin_price_coingecko(coin, from_datetime, to_datetime, include_mkt_cap=True)

        return (coin[1], df)

    chunks = list(range(1, len(coins)//30 + 2))

    data = []

    for i in chunks:
        print(f"Downloading chunk {i} of {chunks[-1]}")
        temp = Parallel(n_jobs=30, backend="threading")(
            delayed(get_data)(coin) for coin in coins[(i-1)*30:i*30])

        data.extend(temp)

        if i != max(chunks):
            print("Sleeping...")
            time.sleep(120)

    return data

def get_clusters_stats(df_clustering:pd.DataFrame, prices_and_mkt_caps:list):

    prices_df = pd.DataFrame(index=pd.date_range(INIT_DATE, END_DATE), columns=[t[0] for t in prices_and_mkt_caps])
    mkt_caps_df = pd.DataFrame(index=pd.date_range(INIT_DATE, END_DATE), columns=[t[0] for t in prices_and_mkt_caps])
    for t in prices_and_mkt_caps:
        #REMEMBER TO DELETE [0]
        temp_df = t[1].set_index("date")
        prices_df.loc[:, t[0]] = temp_df.loc[:, "price"]
        mkt_caps_df.loc[:, t[0]] = temp_df.loc[:, "mkt_cap"]
    clusters = df_clustering.groupby("cluster").apply(lambda g: g["symbol"].tolist())
    clusters_stats = {}
    for c, cluster in enumerate(clusters):
        symbols = [s for s in cluster if s in prices_df.columns.tolist()]
        weights = mkt_caps_df.loc[:, symbols].divide(mkt_caps_df.loc[:, symbols].sum(axis=1), axis=0).fillna(0.)
        symbols_returns = prices_df.loc[:, symbols].pct_change()
        for col in symbols:
            symbols_returns.loc[symbols_returns.loc[:, col] > 400, col] = 0.
        port_returns = np.log1p((weights * symbols_returns).sum(axis=1)).replace({np.inf: np.nan}).dropna()
        mean_ret = port_returns.mean()*365.
        std = port_returns.std(ddof=1)*np.sqrt(365.)
        sharpe = mean_ret/std
        clusters_stats[c] = {"mean_ret": mean_ret, "std": std, "sharpe": sharpe, "symbols": cluster}

    return pd.DataFrame(clusters_stats).T.reset_index().rename({"index": "cluster"}, axis=1)


if __name__ == "__main__":

    import pickle
    from joblib import Parallel, delayed
    import pandas as pd
    from pycoingecko import CoinGeckoAPI

    cg = CoinGeckoAPI()

    print("Getting metrics info...")
    res = get_metrics_info(INTERVAL_METRICS + AGG_METRICS)

    pickle.dump(res, open("./dashboard/data/metrics_description", "wb"))
    print("Done!")

    print("Getting coins metrics...")
    coins_metrics = Parallel(n_jobs=40, backend="threading")(delayed(get_coin_metrics)(coin, AGG_METRICS, INTERVAL_METRICS, INIT_STR_AGG, INIT_STR_INTERVAL, END_STR) for coin in SUPPORTED_ASSETS)
    coins_metrics = pd.DataFrame(coins_metrics)
    pickle.dump(coins_metrics, open("./dashboard/data/coins_metrics", "wb"))
    print("Done!")

    coins_list = pd.DataFrame(cg.get_coins_list()).drop_duplicates(subset="symbol")
    valid_symbols = list(set(coins_metrics.loc[:, "symbol"].values.tolist()).intersection(
        set(coins_list.loc[:, "symbol"].values.tolist())
    ))
    coins_metrics = coins_metrics.loc[
        coins_metrics.loc[:, "symbol"].isin(valid_symbols)
    ]

    print("Performing clustering...")
    df_clustering = perform_clustering(coins_metrics)

    pickle.dump(df_clustering, open("./dashboard/data/df_clustering", "wb"))
    print("Done!")

    print("Clusters stats...")

    mask = coins_list.loc[:, "symbol"].isin(valid_symbols)
    print("\tDownloading data...")
    prices_and_market_caps = get_prices_and_market_caps(
        coins_list.loc[mask, ["id", "symbol"]].values.tolist(), INIT_DATE, END_DATE
    )
    pickle.dump(prices_and_market_caps, open("./dashboard/data/prices_and_market_caps", "wb"))
    print("\tCalculating stats...")
    clusters_stats = get_clusters_stats(df_clustering, prices_and_market_caps)

    pickle.dump(clusters_stats, open("./dashboard/data/clusters_stats", "wb"))