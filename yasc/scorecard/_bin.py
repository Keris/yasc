# Author: Liqiang Du <keris.du@gmail.com>
import pandas as pd
import numpy as np
from scipy import stats

from ..scorecard.util._check import check_target


__all__ = ["mono_bin"]


def mono_bin(Y, X, n=20, precision=3, duplicates="raise"):
    """Generate monotonous bins.

    Parameters
    ----------
    Y : Series
        A series of labels.
    X : Series
        The series to bin, it should be of numeric type.
    n : int or list-like of int, optional
        Number of quantiles, by default 20
    precision : int, optional
        The precision at which to store and display the bins labels, by default 3
    duplicates : str, optional
        Argument used by :func:`pandas.qcut()`, by default "raise"

    Returns
    -------
    DataFrame
        Descriptive statistics of binning.

    Examples
    --------

        >>> from yasc.data import german_data
        >>> from yasc.scorecard import mono_bin
        >>> import pandas as pd; pd.set_option('max_columns', None)
        >>> data = german_data()
        >>> mono_bin(data.Creditability, data.DurationInMonth, duplicates='drop')
        >>> bin_stat
                       min  max  bad_count  good_count  total  bad_rate  good_rate  \\
        Bucket
        (3.999, 12.0]    4   12         76         283    359  0.253333   0.404286
        (12.0, 24.0]    13   24        122         289    411  0.406667   0.412857
        (24.0, 72.0]    26   72        102         128    230  0.340000   0.182857
                            woe        iv    iv_sum                     bins
        Bucket
        (3.999, 12.0] -0.467416  0.070558  0.168117  [-inf, 12, 24, 72, inf]
        (12.0, 24.0]  -0.015108  0.000094  0.168117  [-inf, 12, 24, 72, inf]
        (24.0, 72.0]   0.620240  0.097466  0.168117  [-inf, 12, 24, 72, inf]

    """
    check_target(Y, inplace=True)
    total_bad = Y.sum()
    total_good = Y.count() - total_bad
    rho = 0
    while np.abs(rho) < 1:
        df = pd.DataFrame(
            {"X": X, "Y": Y, "Bucket": pd.qcut(X, n, duplicates=duplicates)}
        )
        gb = df.groupby("Bucket", as_index=True)
        rho, pval = stats.spearmanr(gb.mean().X, gb.mean().Y)
        n = n - 1
    bin_stat = pd.DataFrame()
    bin_stat["min"] = gb.min().X
    bin_stat["max"] = gb.max().X
    bin_stat["bad_count"] = gb.sum().Y
    bin_stat["good_count"] = gb.count().Y - gb.sum().Y
    bin_stat["total"] = gb.count().Y
    bin_stat["bad_rate"] = bin_stat["bad_count"] / total_bad
    bin_stat["good_rate"] = bin_stat["good_count"] / total_good
    bin_stat["woe"] = np.log(bin_stat["bad_rate"] / bin_stat["good_rate"])
    bin_stat["iv"] = (bin_stat["bad_rate"] - bin_stat["good_rate"]) * bin_stat[
        "woe"
    ]
    bin_stat["iv_sum"] = bin_stat["iv"].sum()
    bins = list(bin_stat["max"].round(precision))
    bins.insert(0, float("-inf"))
    bins.append(float("inf"))
    bin_stat["bins"] = str(bins)
    return bin_stat
