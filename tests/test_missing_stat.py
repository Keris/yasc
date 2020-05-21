# Author: Liqiang Du <keris.du@gmail.com>
import pandas as pd
import numpy as np

from yasc.eda import missing_stat


def test_missing_stat():
    df = pd.DataFrame(
        {"a": [1, np.nan, np.nan], "b": [2, np.nan, 3], "c": [4, 5, 6]}
    )

    # check all columns
    assert isinstance(missing_stat(df), pd.DataFrame)
    # check some column
    assert isinstance(missing_stat(df, "a"), str)
