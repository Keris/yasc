.. _api_ref:

API reference
=============

.. _data_api:

.. currentmodule:: yasc.data

Data
----------------

.. autosummary::
    :toctree: generated/

    german_data

.. _eda_api:

.. currentmodule:: yasc.eda

Exploratory data analysis
-------------------------

.. autosummary::
    :toctree: generated/

    missing_stat
    numeric_stat
    categorical_stat
    describe
    corr_analysis

.. _preprocessing_api:

.. currentmodule:: yasc.preprocessing

Preprocessing
-------------

General precessing
^^^^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: generated/

    replace_blank

Handle missing values
^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: generated/

    rf_fill_missing


.. _scorecard_api:

.. currentmodule:: yasc.scorecard

Score card
----------

.. autosummary::
    :toctree: generated/

    check_target
    mono_bin

Plot utilities
^^^^^^^^^^^^^^

.. autosummary::
    :toctree: generated/

    rocplot
    ksplot
    woebinplot
