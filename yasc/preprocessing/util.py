# Author: Liqiang Du <keris.du@gmail.com>
import warnings
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from pandas.api.types import is_numeric_dtype


__all__ = ["rf_fill_missing", "replace_blank"]


def rf_fill_missing(data, column, target=None, n_digits=0, **kwargs):
    """Fill missing values with predicted values output by a random forest regressor.

    Parameters
    ----------
    data : :class:`pandas.core.frame.DataFrame`
        Observed data.
    column : :class:`str`
        Name of the column to fill. This column should be a numeric.
    target : :class:`str`, optional
        Name of target column to predict. Defaults to ``None`` if `data` contains no target.
    n_digits : int, optional
        Precision in decimal digits to round the predicted values.
    kwargs : Keyword arguments
        All keyword arguments that can be passed to
        :class:`sklearn.ensemble.RandomForestRegressor`.

    Returns
    -------
    :class:`pandas.core.frame.DataFrame`
        Returns a data frame with missing values filled in column `column`.

    Raises
    ------
    TypeError
        Raises a :class:`TypeError` when `column` is not numeric.
    """
    if not is_numeric_dtype(data[column].dtype):
        raise TypeError("Column {} should be a numeric".format(column))
    if target is not None:
        df = data.drop(target, axis=1)
    df_known = df[df[column].notnull()]
    df_missing = df[df[column].isnull()]

    X = df_known.loc[:, df_known.columns != column]
    y = df_known.loc[:, column]

    # Create a random forest regressor
    rf = RandomForestRegressor(**kwargs)
    rf.fit(X, y)
    # Predict missing values and assign
    pred = rf.predict(df_missing.loc[:, df_missing.columns != column]).round(
        n_digits
    )
    df.loc[df[column].isnull(), column] = pred

    return df


def replace_blank(data, inplace=False):
    """Replace blank strings in data if any.

    Replace blank strings with ``np.nan``.

    Parameters
    ----------
    data : DataFrame
        Observed data.
    inplace : bool, optional
        Whether to change `data` in place. Defaults to ``False``.

    Returns
    -------
    DataFrame
        Returns ``None`` if there are no blanks or `inplace` is ``True`` else
        returns changed data with blanks replaced with ``np.nan``.
    """
    blank_cols = [
        col
        for col in data.columns
        if data[col]
        .astype(str)
        .str.findall(r"^\s*$")
        .apply(lambda x: len(x) > 0)
        .sum()
        > 0
    ]
    if blank_cols:
        warnings.warn(
            "Blank strings in columns: {} will be replaced with `np.nan`".format(blank_cols)
        )
        return data.replace(r"^\s*$", np.nan, regex=True, inplace=inplace)
    else:
        return None  # Do nothing when there are no blanks
