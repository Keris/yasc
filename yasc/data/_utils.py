# Author: Liqiang Du <keris.du@gmail.com>
import pandas as pd
import pkg_resources


def german_data():
    """Return german data as a data frame."""
    filename = pkg_resources.resource_filename("yasc", "data/german.csv")
    df = pd.read_csv(filename)
    return df
