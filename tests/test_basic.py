import grizzlies as gr
import pandas as pd
import numpy as np


# def test_read_csv(df):
#     assert isinstance(df, gr.Grizzlies), "read_csv() should return a Grizzlies object"
#     assert isinstance(df, pd.DataFrame), "Grizzlies should inherit from pandas.DataFrame"

# def test_read_json(df):
#     assert isinstance(df, gr.Grizzlies), "read_json() should return a Grizzlies object"

def test_head_tail(df):
    assert df.head().shape[0] > 0, "head() should return data"
    assert df.tail().shape[0] > 0, "tail() should return data"

# def test_indexing(df):
#     assert df.iloc[0] is not None, "iloc should return a row"
#     assert df.loc[df.index[0]] is not None, "loc should return a row"
#     assert df.at[df.index[0], df.columns[0]] is not None, "at should return a single value"
#     assert df.iat[0, 0] is not None, "iat should return a single value"

# def test_slicing(df):
#     sliced = df[:2]
#     assert isinstance(sliced, gr.Grizzlies), "Slicing should return a Grizzlies object"
#     assert sliced.shape[0] == 2, "Slicing should return correct number of rows"

def test_boolean_indexing(df):
    filtered = df[df[df.columns[0]] > df[df.columns[0]].median()]
    assert isinstance(filtered, gr.Grizzlies), "Boolean indexing should return a Grizzlies object"

def test_isin(df):
    values = df[df.columns[0]].unique()[:2]  # Pick two unique values
    filtered = df[df[df.columns[0]].isin(values)]
    assert isinstance(filtered, gr.Grizzlies), "isin() should return a Grizzlies object"

def test_fillna_dropna(df):
    filled = df.fillna(0)
    dropped = df.dropna()
    assert isinstance(filled, gr.Grizzlies), "fillna() should return a Grizzlies object"
    assert isinstance(dropped, gr.Grizzlies), "dropna() should return a Grizzlies object"

def test_isna(df):
    result = df.isna()
    assert isinstance(result, pd.DataFrame), "isna() should return a DataFrame (not Grizzlies)"

# def test_mean(df):
#     means = df.mean()
#     assert isinstance(means, pd.Series), "mean() should return a pandas Series"

def test_merge(df):
    df2 = df
    merged = df.merge(df2, on=df.columns[0], how="inner")
    assert isinstance(merged, gr.Grizzlies), "merge() should return a Grizzlies object"

# def test_groupby(df):
#     grouped = df.groupby(df.columns[0]).mean()
#     assert isinstance(grouped, pd.DataFrame), "groupby().mean() should return a DataFrame"