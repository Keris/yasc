# Author: Liqiang Du <keris.du@gmail.com>
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def missing_stat(data, x=None):
    """Return missing values' statistics.

    Only columns with missing values are included in the result.

    Parameters
    ----------
    data : DataFrame.
        A data frame to make statistics.
    x : str or list, optional
        A column name or a list of column names. Defaults to None.

    Returns
    -------
    A string if x is passed as a str else a DataFrame.

    Examples
    --------

    Check missing statistics of total DataFrame:

        >>> from yasc.eda import missing_stat
        >>> import pandas as pd
        >>> import numpy as np
        >>> df = pd.DataFrame({"a": [1, np.nan, np.nan], "b": [2, np.nan, 3], "c": [4, 5, 6]})
        >>> missing_stat(df)
          Column #Missing  MissingRate   Dtype
        0      a        2     0.666667  float64
        1      b        1     0.333333  float64

    Check missing statistics of one column:
        >>> missing_stat(df, "a")
        'Column a of dtype float64, 2 missings (0.67)'

    """
    stat_df = pd.DataFrame(
        columns=("Column", "#Missing", "MissingRate", "Dtype")
    )
    _len = len(data)
    if x is None:
        s = data.isnull().sum()  # Series
    else:
        s = data[x].isnull().sum()
    if isinstance(s, np.int64):  # x is passed as a str
        return "Column {} of dtype {}, {} missings ({:.2f})".format(
            x, data[x].dtype, s, s / _len
        )
    templist = []
    for idx in s[s > 0].index:
        templist.append(
            {
                "Column": idx,
                "#Missing": s[idx],
                "MissingRate": s[idx] / _len,
                "Dtype": data[idx].dtype,
            }
        )
    stat_df = stat_df.append(templist, ignore_index=True)
    return stat_df.sort_values(by="#Missing")


def missing_plot(data, is_missing_stat=False, kws=None):
    """Plot missing statistics of data.

    Parameters
    ----------
    data : DataFrame
        Observed data or missing values statistics.
    is_missing_stat : bool, optional
        Observed data if False else missing values statistics. Defaults to False.
    kws : dict, optional
        Keyword arguments for :func:`barplot`.

    Returns
    -------
    ax : matplotlib Axes or None
         Returns the Axes object with the plot for further tweaking.

    """
    stat = data if is_missing_stat else missing_stat(data)
    if not stat.empty:  # Plot only there are missing values
        kws = {} if kws is None else kws.copy()
        ax = sns.barplot(x="Column", y="#Missing", data=stat, **kws)
        for a, b in zip(range(len(stat.Column)), stat["#Missing"]):
            plt.text(a, b + 0.02, b, ha="center", va="bottom")
        plt.plot(range(len(stat.Column)), stat["MissingRate"], "r")
        plt.plot(range(len(stat.Column)), stat["MissingRate"], "rs")
        for a, b in zip(range(len(stat.Column)), stat["MissingRate"]):
            plt.text(
                a,
                b + 0.02,
                "missing_rate: {:.2f}".format(b),
                ha="center",
                va="bottom",
            )
        return ax
    return None
