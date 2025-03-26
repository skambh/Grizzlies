import pandas as pd
import os
import hashlib
import pickle
from collections import defaultdict


def print_hello():
    print("Hello, world!")

class Grizzlies:
    def __init__(self, data=None, *args, **kwargs):
        if isinstance(data, pd.DataFrame):
            self._df = data
        else:
            self._df = pd.DataFrame(data, *args, **kwargs)
            self._access_counts = {}  # Track accesses per column
        self._access_threshold = 5
        self._hash_indices = {}
        os.makedirs("stats", exist_ok=True)
        object.__setattr__(self, "name", self._default_name())
        object.__setattr__(self, "stats_path", os.path.join("stats", f"{self.name}.pkl"))
        object.__setattr__(self, "_column_access_stats", self._load_stats())

    def _default_name(self):
        # You can customize what you want to hash
        # Sort by columns and index to avoid ordering affecting the hash
        hash_input = str(sorted(self.df.columns.tolist())) + str(self.df.shape)
        return hashlib.md5(hash_input.encode()).hexdigest()
    
    def _save_stats(self):
        print("teehee")
        with open(self.stats_path, 'wb') as f:
            pickle.dump(dict(self._column_access_stats), f)

    def _load_stats(self):
        print("------checked here------")
        if os.path.exists(self.stats_path):
            with open(self.stats_path, 'rb') as f:
                print("------found something------")
                return defaultdict(int, pickle.load(f))
        return defaultdict(int)
    
    def save(self):
        self._save_stats()

    def get_stats(self):
        return dict(self._column_access_stats)

    
    def __getattr__(self, attr):
        """Delegate attribute access to the underlying DataFrame."""
        return getattr(self._df, attr)

    def __getitem__(self, key):
        """Support indexing like df['column'] or df[['col1', 'col2']]."""
        if key in self._hash_indices:
            print(f"Using hash index for fast access on '{key}'")
            result = self._hash_indices[key]
        
        if key not in self._df.columns:
            raise KeyError(f"Column '{key}' not found in DataFrame")

        self._increment_access_count(key)
        self._check_hash_index_creation(key)
        
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
        if name.startswith("_"):
            super().__setattr__(name, value)
        elif hasattr(self._df, name):
            setattr(self._df, name, value)
        else:
            raise AttributeError(
                f"Cannot set unknown attribute '{name}'. Use item assignment like df['{name}'] = ... to add new columns."
            )

    def __setitem__(self, key, value):
        """Allow setting values like df['col'] = data."""
        self._increment_access_count(key)
        self._check_hash_index_creation(key)
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


    def __delitem__(self, key):
        """Support del df[col]."""
        del self._df[key]

    def __call__(self, *args, **kwargs):
        """Allow calling methods directly on the wrapped DataFrame."""
        return self._df(*args, **kwargs)

    def __eq__(self, other):
        """Support df == value."""
        return self._df == (other._df if isinstance(other, Grizzlies) else other)
    
    def _increment_access_count(self, key):
        """Increase access count for the column"""
        if key not in self._access_counts:
            self._access_counts[key] = 0
        self._access_counts[key] += 1
        print(f"------at {self._access_counts[key]} accesses------")

    def _check_hash_index_creation(self, key):
        """Create a hash index when a column is accessed frequently"""
        if self._access_counts[key] >= self._access_threshold and key not in self._hash_indices:
            # Build a hash index: mapping column values to row indices
            self._hash_indices[key] = {value: idx for idx, value in self._df[key].items()}
            print(f"------Hash index created for column: {key}------")


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

