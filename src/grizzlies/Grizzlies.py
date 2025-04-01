import pandas as pd
import os
import hashlib
import pickle
from collections import defaultdict, deque, Counter

class Grizzlies:
    def __init__(self, data=None, scheme = "basic", threshold=5, windowsize=15, xval=5, *args, **kwargs,):
        if isinstance(data, pd.DataFrame):
            self._df = data
        else:
            self._df = pd.DataFrame(data, *args, **kwargs)
        
        self._scheme = scheme
        self._hash_indices = {}
        os.makedirs("stats", exist_ok=True)
        self._stats_path = os.path.join("stats", f"{self._default_name()}.pkl")

        #  SCHEME = SLIDING - uses sliding logic
        if self._scheme == "basic":
            self._increment_access_count = self._increment_access_count_basic
            self._access_counts = self._load_stats()

        elif self._scheme == "sliding":
            self._window_size = windowsize
            self._everyxth = 0
            self._xval = xval
            self._increment_access_count = self._increment_access_count_sliding
            self._sliding_window = self._load_stats() # deque(maxlen=window)
            
        # SCHEME = BASIC - does not ever delete
        self._threshold = threshold 
        self._max_indices = int(windowsize/threshold) # this is a heuristic, can use a diff way that is dynamic?

    # removed update_threshold, specify when creating instead. too much logic otherwise


#################################################################################################################
#                                        schema specific functions below                                        #
#################################################################################################################

    def _drop_index(self, counts):
        for key in self._hash_indices.keys():
            if (key not in list(counts.keys())) or counts[key] < self._threshold:
                del self._hash_indices[key]

    def _increment_access_count_sliding(self, key):
        """Increase access count for the column for sliding scheme"""
        self._everyxth += 1
        self._sliding_window.append(key)
        if self._everyxth % self._xval:
            counts = Counter(self._sliding_window)
            for key, count in counts.items():
                if (count > self._threshold) and (key not in self._hash_indices.keys()):
                    if len(self._hash_indices.keys()) > self._max_indices:
                        self._drop_index(counts)
                    self._create_index(key)


    def _increment_access_count_basic(self, key):
        """Increase access count for the column for basic scheme and check to create index"""
        if key not in self._access_counts:
            self._access_counts[key] = 0
        self._access_counts[key] += 1
        print(f"------at {self._access_counts[key]} accesses------")

        if self._access_counts[key] >= self._threshold and key not in self._hash_indices:
            self._create_index(key)
        
    def _create_index(self, key):
        """Create a hash index when a column is accessed frequently"""
        self._hash_indices[key] = {value: idx for idx, value in self._df[key].items()}
        print(f"------Hash index created for column: {key}------")

#################################################################################################################
#                                      NON-schema specific functions below                                      #
#################################################################################################################     

    def _default_name(self):
        # Sort by columns and index to avoid ordering affecting the hash
        hash_input = str(sorted(self._df.columns.tolist())) + str(self._df.shape) + self._scheme
        return hashlib.md5(hash_input.encode()).hexdigest()
    
    def _load_stats(self):
        print("------checked here------")
        if self._scheme == "basic":
            if os.path.exists(self._stats_path):
                with open(self._stats_path, 'rb') as f:
                    print("------found something------")
                    return defaultdict(int, pickle.load(f))
            return defaultdict(int)
        elif self._scheme == "sliding":
            if os.path.exists(self._stats_path):
                with open(self._stats_path, 'rb') as f:
                    print("------found something------")
                    return pickle.load(f)
            return deque(maxlen=self._window_size)

    
    def save(self):
        print("Saved the stats")
        if self._scheme == "basic":
            with open(self._stats_path, 'wb') as f:
                pickle.dump(self._access_counts, f)
        elif self._scheme == "sliding":
            with open(self._stats_path, 'wb') as f:
                pickle.dump(self._sliding_window, f)

    def get_stats(self):
        if self._scheme == "basic":
            return dict(self._access_counts)
        elif self._scheme == "sliding":
            return dict(Counter(self._sliding_window))
            
    def __getitem__(self, key):
        """Support indexing like df['column'] or df[['col1', 'col2']]."""
        if isinstance(key, list):
            result = self._df[key]
            for k in key:
              self._increment_access_count(k)
            return Grizzlies(result)

        # handle boolean series ex. df[df['col1'] == x]
        if isinstance(key, pd.Series) and key.dtype == bool:
          self._increment_access_count(key.name)
          return Grizzlies(self._df[key])

        if key in self._hash_indices:
            print(f"Using hash index for fast access on '{key}'")
            result = self._hash_indices[key]
        print("reached here")
        if key not in self._df.columns:
            raise KeyError(f"Column '{key}' not found in DataFrame")

        self._increment_access_count(key)
        
        result = self._df[key]
        return Grizzlies(result) if isinstance(result, pd.DataFrame) else result

    def __setitem__(self, key, value):
        """Allow setting values like df['col'] = data."""
        self._increment_access_count(key)
        self._df[key] = value

#################################################################################################################
#                                    do not edit overloaded functions below!                                    #
#################################################################################################################

    def __getattr__(self, attr):
        """Delegate attribute access to the underlying DataFrame."""
        return getattr(self._df, attr)
    
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

    @property
    def loc(self):
        class LocWrapper:
            def __init__(self, parent, loc_obj):
                self._parent = parent
                self._loc = loc_obj

            def __getitem__(self, key):
                result = self._loc[key]
                return Grizzlies(result) if isinstance(result, pd.DataFrame) else result

            def __setitem__(self, key, value):
                self._loc[key] = value  # Modify the underlying DataFrame

        return LocWrapper(self, self._df.loc)

    @property
    def iloc(self):
        class IlocWrapper:
            def __init__(self, parent, iloc_obj):
                self._parent = parent
                self._iloc = iloc_obj

            def __getitem__(self, key):
                result = self._iloc[key]
                return Grizzlies(result) if isinstance(result, pd.DataFrame) else result

            def __setitem__(self, key, value):
                self._iloc[key] = value  # Modify the underlying DataFrame

        return IlocWrapper(self, self._df.iloc)


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
