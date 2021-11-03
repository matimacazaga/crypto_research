import pandas as pd
from pandas.tseries.offsets import CustomBusinessDay, Day, BusinessDay
import numpy as np


class NonMatchingTimezoneError(Exception):
    pass


def diff_custom_calendar_timedeltas(start, end, freq):
    """
    Compute the difference between two pd.Timedelta taking into consideration
    custom frequency, which is used to deal with custom calendars, such as a
    trading calendar
    Parameters
    ----------
    start : pd.Timestamp
    end : pd.Timestamp
    freq : CustomBusinessDay (see infer_trading_calendar)
    freq : pd.DataOffset (CustomBusinessDay, Day or BDay)
    Returns
    -------
    pd.Timedelta
        end - start
    """
    if not isinstance(freq, (Day, BusinessDay, CustomBusinessDay)):
        raise ValueError("freq must be Day, BusinessDay or CustomBusinessDay")

    weekmask = getattr(freq, 'weekmask', None)
    holidays = getattr(freq, 'holidays', None)

    if weekmask is None and holidays is None:
        if isinstance(freq, Day):
            weekmask = 'Mon Tue Wed Thu Fri Sat Sun'
            holidays = []
        elif isinstance(freq, BusinessDay):
            weekmask = 'Mon Tue Wed Thu Fri'
            holidays = []

    if weekmask is not None and holidays is not None:
        # we prefer this method as it is faster
        actual_days = np.busday_count(np.array(start).astype('datetime64[D]'),
                                      np.array(end).astype('datetime64[D]'),
                                      weekmask, holidays)
    else:
        # default, it is slow
        actual_days = pd.date_range(start, end, freq=freq).shape[0] - 1
        if not freq.onOffset(start):
            actual_days -= 1

    timediff = end - start
    delta_days = timediff.components.days - actual_days
    return timediff - pd.Timedelta(days=delta_days)


def infer_trading_calendar(factor_idx, prices_idx):

    full_idx = factor_idx.union(prices_idx)

    traded_weekdays = []
    holidays = []

    days_of_the_week = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    for day, day_str in enumerate(days_of_the_week):

        weekday_mask = (full_idx.dayofweek == day)

        if not weekday_mask.any():
            continue
        traded_weekdays.append(day_str)

        used_weekdays = full_idx[weekday_mask].normalize()

        all_weekdays = pd.date_range(
            full_idx.min(), full_idx.max(),
            freq=CustomBusinessDay(weekmask=day_str)
        ).normalize()

        _holidays = all_weekdays.difference(used_weekdays)
        _holidays = [timestamp.date() for timestamp in _holidays]
        holidays.extend(_holidays)

    traded_weekdays = ' '.join(traded_weekdays)

    return CustomBusinessDay(weekmask=traded_weekdays, holidays=holidays)


def compute_forward_returns(
    factor, prices, periods=(1,5,10), filter_zscore=20, cumulative_returns=True
):

    factor_dateindex = factor.index.levels[0]

    if factor_dateindex.tz != prices.index.tz:
        raise NonMatchingTimezoneError("The timezone of 'factor' is not the "
                                       "same as the timezone of 'prices'. See "
                                       "the pandas methods tz_localize and "
                                       "tz_convert.")

    freq = infer_trading_calendar(factor_dateindex, prices.index)

    factor_dateindex = factor_dateindex.intersection(prices.index)

    if len(factor_dateindex) == 0:

        raise ValueError("Factor and prices indices don't match: make sure "
                         "they have the same convention in terms of datetimes "
                         "and symbol-names")

    prices = prices.filter(items=factor.index.levels[1])

    raw_values_dict = {}
    columns_list = []

    for period in sorted(periods):

        if cumulative_returns:
            returns = prices.pct_change(period)
        else:
            returns = prices.pct_change()

        forward_returns = returns.shift(-period).reindex(factor_dateindex)

        if filter_zscore is not None:

            mask = abs(
                forward_returns - forward_returns.mean()
            ) > (filter_zscore * forward_returns.std())

            forward_returns[mask] = np.nan

        days_diffs = []
        for i in range(30):
            if i >= len(forward_returns.index):
                break
            p_idx = prices.index.get_loc(forward_returns.index[i])
            if p_idx is None or p_idx < 0 or (
                    p_idx + period) >= len(prices.index):
                continue
            start = prices.index[p_idx]
            end = prices.index[p_idx + period]
            period_len = diff_custom_calendar_timedeltas(start, end, freq)
            days_diffs.append(period_len.components.days)

def get_clean_factor_and_forward_returns(
    factor, prices, groupby=None, binning_by_group=False, quantiles=5,
    bins=None, periods=(1, 5, 10), filter_zscore=20, groupby_labels=None,
    max_loss=0.35, zero_aware=False, cumulative_returns=True
):

    forward_returns = compute_forward_returns(
        factor, prices, periods, filter_zscore, cumulative_returns
    )