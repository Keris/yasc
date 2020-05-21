# Author: Liqiang Du <keris.du@gmail.com>
import pandas as pd
import os


def german_data():
    """Return german data as a data frame."""
    this_dir = os.path.dirname(os.path.realpath(__file__))
    df = pd.read_csv(os.path.join(this_dir, "german.csv"))
    return df
