from dateutil.relativedelta import relativedelta
from datetime import datetime
import coinmetrics
from joblib.parallel import delayed
import numpy as np


COINMETRICS = coinmetrics.Community()
END_DATE = datetime.now()
INIT_DATE_INTERVAL = END_DATE - relativedelta(days=15)
INIT_DATE_AGG = END_DATE - relativedelta(days=1)
INIT_STR_INTERVAL = INIT_DATE_INTERVAL.strftime("%Y-%m-%d")
INIT_STR_AGG = INIT_DATE_AGG.strftime("%Y-%m-%d")
END_STR = END_DATE.strftime("%Y-%m-%d")
METRICS = [
    "AdrActCnt", "BlkCnt", "CapMVRVCur", "CapMVRVFF", "CapMrktFFUSD",
    "CapRealUSD", "NVTAdjFF", "NVTAdjFF90", "FeeMeanUSD", "FeeMedUSD",
    "IssContPctAnn", "NVTAdj90", "RevUSD", "RevAllTimeUSD",
    "TxCntSec", "TxTfrValMeanUSD", "VelCur1yr",
    "ROI30d", "VtyDayRet30d"
]

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


def get_coin_metrics(coin:str, aggregated_metrics:list, interval_metrics:list, init_date_agg:str, init_date_int:str, end_date:str):
    metrics = COINMETRICS.get_available_data_types_for_asset(coin)

    intersection = set(metrics).intersection(interval_metrics+aggregated_metrics)

    if intersection:
        agg_metrics = [m for m in intersection if m in aggregated_metrics]
        int_metrics = [m for m in intersection if m in interval_metrics]

        metrics = {}

        if int_metrics:
            int_metrics = COINMETRICS.get_asset_data_for_time_range(coin, ",".join(int_metrics), init_date_int, end_date)
            if int_metrics["series"]:
                for m, metric in enumerate(int_metrics["metrics"]):
                    values = [float(d["values"][m]) for d in int_metrics["series"] if d["values"][m]]
                    metrics[metric] = np.mean(values)

        if agg_metrics:
            agg_metrics = COINMETRICS.get_asset_data_for_time_range(coin, ",".join(agg_metrics), init_date_int, end_date)
            if agg_metrics["series"]:
                for m, v in zip(agg_metrics["metrics"], [float(d) if d != None else d for d in agg_metrics["series"][0]["values"]]):
                    metrics[m] = v

        metrics["coin"] = coin

    else:
        metrics["coin"] = coin

    return metrics

if __name__ == "__main__":

    import pickle
    from joblib import Parallel, delayed
    import pandas as pd

    res = get_metrics_info(METRICS)

    pickle.dump(res, open("./dashboard/data/metrics_description", "wb"))

    coins_metrics = Parallel(n_jobs=40, backend="threading")(delayed(get_coin_metrics)(coin, AGG_METRICS, INTERVAL_METRICS, INIT_STR_AGG, INIT_STR_INTERVAL, END_STR) for coin in SUPPORTED_ASSETS)

    coins_metrics = pd.DataFrame(coins_metrics)

    pickle.dump(coins_metrics, open("./dashboard/data/coins_metrics", "wb"))
