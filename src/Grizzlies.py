import pandas as pd

class Grizzlies:
    def __init__(self, data=None, *args, **kwargs):
        if isinstance(data, pd.DataFrame):
            self._df = data
        else:
            self._df = pd.DataFrame(data, *args, **kwargs)

    def __getattr__(self, attr):
        """Delegate attribute access to the underlying DataFrame."""
        return getattr(self._df, attr)

    def __getitem__(self, key):
        """Support indexing like df['column'] or df[['col1', 'col2']]."""
        result = self._df[key]
        return Grizzlies(result) if isinstance(result, pd.DataFrame) else result

    def __setitem__(self, key, value):
        """Support assignment like df['new_col'] = values."""
        self._df[key] = value

    def __repr__(self):
        """Ensure the object prints like a normal DataFrame."""
        return repr(self._df)

    def head(self, n=5):
        return Grizzlies(self._df.head(n))

    def tail(self, n=5):
        return Grizzlies(self._df.tail(n))

    def dropna(self, *args, **kwargs):
        return Grizzlies(self._df.dropna(*args, **kwargs))

    def fillna(self, *args, **kwargs):
        return Grizzlies(self._df.fillna(*args, **kwargs))

    def isna(self):
        return self._df.isna()

    def mean(self, *args, **kwargs):
        return self._df.mean(*args, **kwargs)

    def merge(self, right, *args, **kwargs):
        if isinstance(right, Grizzlies):
            right = right._df
        return Grizzlies(pd.merge(self._df, right, *args, **kwargs))

    def groupby(self, *args, **kwargs):
        return self._df.groupby(*args, **kwargs)

    def isin(self, values):
        return self._df.isin(values)

    def loc(self, *args):
        result = self._df.loc[*args]
        return Grizzlies(result) if isinstance(result, pd.DataFrame) else result

    def iloc(self, *args):
        result = self._df.iloc[*args]
        return Grizzlies(result) if isinstance(result, pd.DataFrame) else result

    def at(self, *args):
        return self._df.at[*args]

    def iat(self, *args):
        return self._df.iat[*args]

    def __setattr__(self, name, value):
        """Allow setting attributes on the underlying DataFrame."""
        if name == "_df":
            super().__setattr__(name, value)
        else:
            setattr(self._df, name, value)

    def __setitem__(self, key, value):
        """Allow setting values like df['col'] = data."""
        self._df[key] = value

    def __iter__(self):
        """Support iteration over columns like a normal DataFrame."""
        return iter(self._df)

    def __len__(self):
        """Support len(df)."""
        return len(self._df)

    def __contains__(self, item):
        """Support 'col' in df."""
        return item in self._df

    def __getitem__(self, key):
        """Support df[col] and df[row:row]."""
        result = self._df[key]
        return Grizzlies(result) if isinstance(result, pd.DataFrame) else result

    def __setitem__(self, key, value):
        """Support df[col] = value."""
        self._df[key] = value

    def __delitem__(self, key):
        """Support del df[col]."""
        del self._df[key]

    def __call__(self, *args, **kwargs):
        """Allow calling methods directly on the wrapped DataFrame."""
        return self._df(*args, **kwargs)

    def __eq__(self, other):
        """Support df == value."""
        return self._df == (other._df if isinstance(other, Grizzlies) else other)

# Module-level functions
def read_csv(*args, **kwargs):
    return Grizzlies(pd.read_csv(*args, **kwargs))

def read_excel(*args, **kwargs):
    return Grizzlies(pd.read_excel(*args, **kwargs))

def read_json(*args, **kwargs):
    return Grizzlies(pd.read_json(*args, **kwargs))

def read_parquet(*args, **kwargs):
    return Grizzlies(pd.read_parquet(*args, **kwargs))

def DataFrame(*args, **kwargs):
    return Grizzlies(pd.DataFrame(*args, **kwargs))

def Series(*args, **kwargs):
    return pd.Series(*args, **kwargs)  # Keeping Series as a normal pandas object for now
