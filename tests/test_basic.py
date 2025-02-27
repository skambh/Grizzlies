import grizzlies as gr
import pandas as pd
import numpy as np

# Sample test data
data_csv = "data/data.csv"
data_json = "data/data.json"

def test_read_csv():
    df = gr.read_csv(data_csv)
    assert isinstance(df, gr.Grizzlies), "read_csv() should return a Grizzlies object"
    assert isinstance(df, pd.DataFrame), "Grizzlies should inherit from pandas.DataFrame"

def test_read_json():
    df = gr.read_json(data_json)
    assert isinstance(df, gr.Grizzlies), "read_json() should return a Grizzlies object"

def test_head_tail():
    df = gr.read_csv(data_csv)
    assert df.head().shape[0] > 0, "head() should return data"
    assert df.tail().shape[0] > 0, "tail() should return data"

def test_indexing():
    df = gr.read_csv(data_csv)
    assert df.iloc[0] is not None, "iloc should return a row"
    assert df.loc[df.index[0]] is not None, "loc should return a row"
    assert df.at[df.index[0], df.columns[0]] is not None, "at should return a single value"
    assert df.iat[0, 0] is not None, "iat should return a single value"

def test_slicing():
    df = gr.read_csv(data_csv)
    sliced = df[:2]
    assert isinstance(sliced, gr.Grizzlies), "Slicing should return a Grizzlies object"
    assert sliced.shape[0] == 2, "Slicing should return correct number of rows"

def test_boolean_indexing():
    df = gr.read_csv(data_csv)
    filtered = df[df[df.columns[0]] > df[df.columns[0]].median()]
    assert isinstance(filtered, gr.Grizzlies), "Boolean indexing should return a Grizzlies object"

def test_isin():
    df = gr.read_csv(data_csv)
    values = df[df.columns[0]].unique()[:2]  # Pick two unique values
    filtered = df[df[df.columns[0]].isin(values)]
    assert isinstance(filtered, gr.Grizzlies), "isin() should return a Grizzlies object"

def test_fillna_dropna():
    df = gr.read_csv(data_csv)
    filled = df.fillna(0)
    dropped = df.dropna()
    assert isinstance(filled, gr.Grizzlies), "fillna() should return a Grizzlies object"
    assert isinstance(dropped, gr.Grizzlies), "dropna() should return a Grizzlies object"

def test_isna():
    df = gr.read_csv(data_csv)
    result = df.isna()
    assert isinstance(result, pd.DataFrame), "isna() should return a DataFrame (not Grizzlies)"

def test_mean():
    df = gr.read_csv(data_csv)
    means = df.mean()
    assert isinstance(means, pd.Series), "mean() should return a pandas Series"

def test_merge():
    df1 = gr.read_csv(data_csv)
    df2 = gr.read_csv(data_csv)
    merged = df1.merge(df2, on=df.columns[0], how="inner")
    assert isinstance(merged, gr.Grizzlies), "merge() should return a Grizzlies object"

def test_groupby():
    df = gr.read_csv(data_csv)
    grouped = df.groupby(df.columns[0]).mean()
    assert isinstance(grouped, pd.DataFrame), "groupby().mean() should return a DataFrame"

if __name__ == "__main__":
    test_read_csv()
    test_read_json()
    test_head_tail()
    test_indexing()
    test_slicing()
    test_boolean_indexing()
    test_isin()
    test_fillna_dropna()
    test_isna()
    test_mean()
    test_merge()
    test_groupby()
    print("✅ All tests passed!")
