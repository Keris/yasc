# Author: Liqiang Du <keris.du@gmail.com>
from ...exception import LabelCountError, LabelValueError


def check_target(Y, inplace=False):
    """Check validity of target.

    Target is expected to be categorical including only two values, either
    'bad' and 'good', or 0 and 1. 'bad' or 1 indicates a bad case while 'good'
    or 0 a good case.

    Parameters
    ----------
    Y : Series
        The target column to check.
    inplace : bool, optional
        Whether to change Y` in place, by default ``False``.

    Returns
    -------
    Series
        Return ``None`` if `inplace` is ``True`` else changed `Y` with
        'good` replaced with 0 and 'bad' 1.

    Raises
    ------
    LabelCountError
        Raises when unique count of labels doesn't equal 2.
    LabelValueError
        Raises when label values are not valid.

    Examples
    --------

        >>> from yasc.data import german_data
        >>> from yasc.scorecard.util import check_target
        >>> data = german_data()
        >>> data.Creditability
        0      good
        1       bad
        2      good
        3      good
        4       bad
            ...
        995    good
        996    good
        997    good
        998     bad
        999    good
        Name: Creditability, Length: 1000, dtype: object
        >>> check_target(data.Creditability)
        0      0
        1      1
        2      0
        3      0
        4      1
            ..
        995    0
        996    0
        997    0
        998    1
        999    0
        Name: Creditability, Length: 1000, dtype: int64
        >>> data.Creditability
        0      good
        1       bad
        2      good
        3      good
        4       bad
            ...
        995    good
        996    good
        997    good
        998     bad
        999    good
        Name: Creditability, Length: 1000, dtype: object
        >>> check_target(data.Creditability, inplace=True)  # Returns None
        >>> data.Creditability
        0      0
        1      1
        2      0
        3      0
        4      1
            ..
        995    0
        996    0
        997    0
        998    1
        999    0
        Name: Creditability, Length: 1000, dtype: int64
    """
    if Y.nunique() != 2:
        raise LabelCountError("unique count of labels expects to be 2")
    label_values = sorted(Y.unique())
    if label_values != ["bad", "good"] and label_values != [0, 1]:
        raise LabelValueError(
            "label values are either in ['bad', 'good'] or in [0, 1]"
        )
    if label_values == ["bad", "good"]:
        return Y.replace({"bad": 1, "good": 0}, inplace=inplace)
    else:
        return None  # Y is already valid
