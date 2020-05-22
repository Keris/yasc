# Author: Liqiang Du <keris.du@gmail.com>
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.api.types import is_numeric_dtype, is_object_dtype


def missing_stat(
    data, columns=None, show_print=True, only_missing_columns=False
):
    """Return missing values' statistics.

    Parameters
    ----------
    data : :class:`DataFrame`
        A data frame to make statistics.
    columns : :class:`str` or :class:`list`, optional
        A column name or a list of column names. Defaults to None.
    show_print : bool, optional
        Whether to print summay information. Defaults to True.
    only_missing_columns : bool, optional
        Whether to only include columns with missing values in the output.
        Defaults to False.

    Returns
    -------
    A string if `columns` is passed as a :class:`str` else a :class:`DataFrame`.

    Examples
    --------

    Check missing statistics of all columns:

        >>> from yasc.eda import missing_stat
        >>> import pandas as pd
        >>> import numpy as np
        >>> df = pd.DataFrame({"a": [1, np.nan, np.nan], "b": [2, np.nan, 3], "c": [4, 5, 6]})
        >>> missing_stat(df)
          Column #missing  missing_rate
        0      a        2      0.666667
        1      b        1      0.333333

    Check missing statistics of one column:
        >>> missing_stat(df, "a")
        Column a of dtype float64, 2 missings (0.67)

    """
    _len = len(data)
    if columns is None:
        s = data.isnull().sum()  # Series
    else:
        s = data[columns].isnull().sum()
    if isinstance(s, np.int64):  # columns is passed as a str
        print(
            "Column {} of dtype {}, {} missing(s) ({:.2f})".format(
                columns, data[columns].dtype, s, s / _len
            )
        )
    else:
        stat_df = pd.DataFrame(s).reset_index()
        stat_df.rename(columns={"index": "column", 0: "#missing"}, inplace=True)
        if only_missing_columns:
            stat_df = stat_df[stat_df["#missing"] > 0]
        stat_df["missing_rate"] = stat_df["#missing"] / _len
        n_missing_columns = len(stat_df[stat_df["#missing"] > 0])
        if show_print:
            if n_missing_columns:
                print(
                    "{} columns, of which {} columns with missing values".format(
                        len(data.columns), n_missing_columns
                    )
                )
            else:
                print("No missing values")
        return stat_df.sort_values(by="#missing")


def numeric_stat(data, percentiles=None):
    """Describe numeric columns.

    Parameters
    ----------
    data : DataFrame
        Observed data.
    percentils : list-like of numbers, optional
        The percentiles to include in the ouput.

    Returns
    -------
    desc : DataFrame
        A descriptive statistics for numeric columns.

    """
    desc = data.describe(percentiles, include=np.number)
    return desc


def categorical_stat(data):
    """Generate descriptive statistics for categorical columns.

    Categorical columns here are columns of dtype `dtype('O')`.

    Parameters
    ----------
    data : DataFrame
        Observed Data.

    Returns
    -------
    desc : DataFrame
        A descriptive statistics for categorical columns.

    """
    desc = data.describe(include=np.object)
    return desc


def describe(data, percentiles=None):
    """Generate descriptive statistics.

    Parameters
    ----------
    data : DataFrame
        Observed data
    percentiles : list-like of numbers, optional
        The percentiles to include in the output. Defaults to None.

    Returns
    -------
    :class:`pandas.core.frame.DataFrame`
        Descriptive statistics including numeric columns, categorical columns
        and missing values.
    """
    col_dtypes = pd.DataFrame(data.dtypes)
    col_dtypes.rename(columns={0: "dtype"}, inplace=True)

    def get_type(dtype):
        if is_numeric_dtype(dtype):
            return "numeric"
        elif is_object_dtype(dtype):
            return "categorical"
        else:
            return str(dtype)

    col_dtypes["type"] = col_dtypes.dtype.apply(get_type)
    desc = data.describe(percentiles, include="all")
    result_df = pd.DataFrame(columns=desc.columns)
    missing_desc = (
        missing_stat(data, show_print=False).sort_index().set_index("column")
    )
    result_df = result_df.append(col_dtypes.dtype)
    result_df = result_df.append(col_dtypes.type)
    result_df = result_df.append(missing_desc["#missing"])
    result_df = result_df.append(missing_desc.missing_rate)
    result_df = result_df.append(desc)

    return result_df


def corr_analysis(
    data, tight_layout=False, show_plot=False, title=None, rot=None, **kwargs
):
    """Correlation analysis.

    Parameters
    ----------
    data : DataFrame
        Observed data.
    tight_layout : bool, optional
        Whether to make figure tight layout. Defaults to False.
    show_plot : bool, optional
        Whether to show heatmap of correlation matrix.
    title : :class:`str`
        Title of heatmap of correlation matrix. Defautls to `None`.
    rot : int
        Degrees of rotation for `xticklabels`.
    kwargs : Keyword arguments
        All keyword arguments that are passed to :func:`seaborn.heatmap`.

    Returns
    -------
    :class:`tuple`
        Return correlation matrix and axes object with the heatmap.

    """
    if title is None:
        title = "Heatmap of correlation matrix"
    if rot is None:
        rot = 45
    numeric_cols = [
        col for col in data.columns if is_numeric_dtype(data[col].dtype)
    ]
    corr = data[numeric_cols].corr()
    fig, ax = plt.subplots()
    ax = sns.heatmap(
        corr,
        vmin=-1,
        vmax=1,
        center=0,
        annot=True,
        cmap="YlGnBu",
        ax=ax,
        **kwargs,
    )
    ax.set_title(title)
    plt.setp(
        ax.get_xticklabels(), rotation=rot, ha="right", rotation_mode="anchor"
    )
    if tight_layout:
        fig.tight_layout()
    if show_plot:
        plt.show()
    return corr, ax
