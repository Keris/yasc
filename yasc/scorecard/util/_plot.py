# Author: Liqiang Du <keris.du@gmail.com>
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

from ._util import compute_ks_lift


def rocplot(y_true, y_preds, equal_aspect=False):
    """Plot a ROC curve.

    Parameters
    ----------
    y_true : array, shape=[n_samples]
        True binary labels.
    y_preds : array, shape=[n_samples]
        Predicted probability estimates of the positive class.
    equal_aspect : bool, optional
        Whether to make the aspect equal. Defaults to ``False``.

    Returns
    -------
    roc_auc : float
        Returns area under the curve.
    ax : matplotlib.axes.Axes
        Returns axes object with the plot drawn onto it.

    Examples
    --------

    .. plot::
        :context: close-figs

        >>> from yasc.scorecard.util import rocplot
        >>> import numpy as np
        >>> np.random.seed(0)
        >>> preds = np.random.rand(1000)
        >>> labels = np.random.choice(2, 1000)
        >>> rocplot(labels, preds)
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_preds)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots()
    ax.plot(
        fpr,
        tpr,
        color="blue",
        linestyle="-",
        linewidth=2,
        label="AUC={:.2f}".format(roc_auc),
    )
    ax.plot([0, 1], [0, 1], color="red", linestyle="--", linewidth=2)
    ax.fill_between(fpr, 0, tpr, color="blue", alpha=0.2)
    ax.set_title("ROC")
    ax.legend(loc="lower right")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    if equal_aspect:
        ax.set_aspect("equal", adjustable="box")
    return roc_auc, ax


def ksplot(preds, labels, data=None, n=50, is_prob=True, equal_aspect=False):
    """Plot distributions of good and bad clients, including an estimate of the KS statistics.

    Parameters
    ----------
    preds : array, shape=[n_samples]
        Predicted values of the positive class, either scores or probabilities.
    labels : array, shape=[n_samples]
        True binary labels. A label takes value in {0, 1} with 0 indicating a good client,
        1 a bad client.
    data : DataFrame, optional
        A data frame returned by :func:`yasc.scorecard.util._util.compute_ks_lift`.
        Defaults to ``None``.
    n : int, optional
        Number of segments to compute KS, by default 50
    is_prob : bool, optional
        If True given, `preds` are probabilities else scores, by default True
    equal_aspect : bool, optional
        Whether to make aspect equal. Defaults to ``False``.

    Returns
    -------
    ks_value : float
        The value of KS statistics.
    ax : matplotlib.axes.Axes
        Axes object with the plot drawn onto it.

    Examples
    --------
    Plot a simple KS curve.

    .. plot::
        :context: close-figs

        >>> import numpy as np
        >>> np.random.seed(0)
        >>> preds = np.random.rand(1000)
        >>> labels = np.random.choice(2, 1000)
        >>> from yasc.scorecard.util import ksplot
        >>> ksplot(preds, labels)

    """
    if data is None:
        df_ks_lift = compute_ks_lift(
            preds, labels, ascending=(not is_prob), tile_num=n
        )
    else:
        df_ks_lift = data.copy()

    ks_value = df_ks_lift.ks.max()
    ks_pop = df_ks_lift.tile[df_ks_lift.ks.idxmax()]

    # Make the plot
    _, ax = plt.subplots()
    line_settings = {"linestyle": "-", "linewidth": 2}
    ax.plot(
        df_ks_lift.tile,
        df_ks_lift.cum_good,
        label="cum_good",
        color="blue",
        **line_settings,
    )
    ax.plot(
        df_ks_lift.tile,
        df_ks_lift.cum_bad,
        label="cum_bad",
        color="red",
        **line_settings,
    )
    ax.plot(
        df_ks_lift.tile,
        df_ks_lift.ks,
        label="ks",
        color="green",
        **line_settings,
    )
    cum_good = df_ks_lift.cum_good[df_ks_lift.ks.idxmax()]
    cum_bad = df_ks_lift.cum_bad[df_ks_lift.ks.idxmax()]
    ax.plot(
        [ks_pop, ks_pop],
        [cum_good, cum_bad],
        color="k",
        label="max ks",
        **line_settings,
    )
    ax.text(
        ks_pop + 0.05,
        cum_good + ks_value / 2,
        "ks={:.4f}, pop={:.4f}".format(ks_value, ks_pop),
    )
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.legend(loc="best")
    ax.set_xlabel(r"% of The Population")
    ax.set_title("KS Statistics")
    if equal_aspect:
        ax.set_aspect("equal", adjustable="box")
    return df_ks_lift, ax


def woebinplot(
    data,
    stacked=False,
    grouped=True,
    width=0.2,
    loc1="best",
    loc2="best",
    **kwargs,
):
    """Visulize the binning.

    Parameters
    ----------
    data : DataFrame
        A data frame describing the binning of certain predictor.
    stacked : bool, optional
        Whether to make a stacked bar plot, by default False
    grouped : bool, optional
        Whether to make a grouped bar plot, by default True
    width : float, optional
        The width of bar, by default 0.2
    loc1 : str, optional
        Specify how to put legend in left subplot, by default "best"
    loc2 : str, optional
        Specify how to put legend in right subplot, by default "best"

    Returns
    -------
    fig : matplotlib.figure.Figure
        The result figure.
    ax1 : matplotlib.axes.Axes
        The axes object with the left subplot drawn onto it.
    ax2 : matplotlib.axes.Axes
        The axes object with the right subplot drawn onto it.

    Examples
    --------

    Make a stacked bar plot of the binning result.

    .. plot::
        :context: close-figs

        >>> from yasc.data import german_data
        >>> from yasc.scorecard import mono_bin
        >>> from yasc.scorecard.util import woebinplot
        >>> data = german_data()
        >>> bin_stat = mono_bin(data.Creditability, data.DurationInMonth, duplicates='drop')
        >>> woebinplot(bin_stat, stacked=True, figsize=(8, 6), loc2="lower center")

    Make a grouped bar plot of the binning result.

    .. plot::
        :context: close-figs

        >>> from yasc.data import german_data
        >>> from yasc.scorecard import mono_bin
        >>> from yasc.scorecard.util import woebinplot
        >>> data = german_data()
        >>> bin_stat = mono_bin(data.Creditability, data.DurationInMonth, duplicates='drop')
        >>> woebinplot(bin_stat, figsize=(8, 6), loc2="lower center")
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, **kwargs)
    if stacked:
        grouped = False
    x1 = np.arange(len(data.index))

    # Make the plot
    if stacked:
        ax1.bar(
            x1,
            data.good_count,
            color="forestgreen",
            width=width,
            label="good",
            edgecolor="w",
        )
        ax1.bar(
            x1,
            data.bad_count,
            bottom=data.good_count,
            color="coral",
            width=width,
            label="bad",
            edgecolor="w",
            tick_label=data.index,
        )
        # Annotate the plot
        for x, y in zip(x1, data.total):
            ax1.annotate(
                "{}".format(y),
                (x, y),
                textcoords="offset points",
                xytext=(0, 5),
                ha="center",
            )
    if grouped:
        x2 = [i + width for i in x1]
        x3 = [i + width for i in x2]
        ax1.bar(
            x1,
            data.total,
            color="silver",
            width=width,
            label="good + bad",
            edgecolor="w",
        )
        ax1.bar(
            x2,
            data.good_count,
            color="forestgreen",
            width=width,
            label="good",
            edgecolor="w",
            tick_label=data.index,
        )
        ax1.bar(
            x3,
            data.bad_count,
            color="coral",
            width=width,
            label="bad",
            edgecolor="w",
        )
        # Annotate the plot
        for x, y in zip(x1, data.total):
            ax1.annotate(
                "{}".format(y),
                (x, y),
                textcoords="offset points",
                xytext=(0, 5),
                ha="center",
            )
        for x, y in zip(x2, data.good_count):
            ax1.annotate(
                "{}".format(y),
                (x, y),
                textcoords="offset points",
                xytext=(0, 5),
                ha="center",
            )
        for x, y in zip(x3, data.bad_count):
            ax1.annotate(
                "{}".format(y),
                (x, y),
                textcoords="offset points",
                xytext=(0, 5),
                ha="center",
            )
    ax1.set_xlabel("Bins")
    ax1.set_ylabel("Count per Bin")
    ax1.legend(loc=loc1)
    ax1.set_title("iv: {:.4f}".format(data.iv_sum[0]))
    ax2.plot(x1, data.bad_rate, marker="o", color="coral", label="bad rate")
    ax2.plot(
        x1,
        data.total / data.total.sum(),
        marker="o",
        color="skyblue",
        label="count dist",
    )
    for x, y in zip(x1, data.bad_rate):
        ax2.text(
            x,
            y,
            r"{:.1f}%".format(y * 100),
            color="coral",
            va="bottom",
            ha="left",
        )
    for x, y in zip(x1, data.total / data.total.sum()):
        ax2.text(
            x,
            y,
            r"{:.1f}%".format(y * 100),
            color="skyblue",
            va="bottom",
            ha="right",
        )
    ax2.set_xlabel("Bins")
    ax2.legend(loc=loc2)
    plt.xticks(x1, data.index)
    fig.tight_layout()
    return fig, ax1, ax2
