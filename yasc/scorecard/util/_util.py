# Author: Liqiang Du <keris.du@gmail.com>
import pandas as pd
import numpy as np


def compute_ks_lift(preds, labels, ascending=False, tile_num=None):
    df = pd.DataFrame(dict(pred=preds, label=labels))
    if tile_num is None:
        tile_num = len(df.index)

    def good(x):
        return sum(x == 0)

    def bad(x):
        return sum(x == 1)

    df_ks_lift = (
        df.sort_values(by="pred", ascending=ascending)
        .reset_index(drop=True)
        .assign(
            tile=lambda x: np.ceil((x.index + 1) / (len(x.index) / tile_num))
        )
        .groupby("tile")["label"]
        .agg([good, bad])
        .reset_index()
        .assign(
            tile=lambda x: (x.index + 1) / len(x.index),
            good_distri=lambda x: x.good / sum(x.good),
            bad_distri=lambda x: x.bad / sum(x.bad),
            bad_rate=lambda x: x.bad / (x.bad + x.good),
            cum_bad_rate=lambda x: np.cumsum(x.bad) / np.cumsum(x.bad + x.good),
            lift=lambda x: (np.cumsum(x.bad) / np.cumsum(x.bad + x.good))
            / (sum(x.bad) / sum(x.bad + x.good)),
            cum_good=lambda x: np.cumsum(x.good) / sum(x.good),
            cum_bad=lambda x: np.cumsum(x.bad) / sum(x.bad),
        )
        .assign(ks=lambda x: abs(x.cum_bad - x.cum_good))
    )
    # Prepend 0
    df_ks_lift = pd.concat(
        [
            pd.DataFrame(
                {
                    c: 0 if c not in ["cum_bad_rate", "lift"] else np.nan
                    for c in df_ks_lift.columns
                },
                index=np.arange(1),
            ),
            df_ks_lift,
        ],
        ignore_index=True,
    )
    return df_ks_lift
