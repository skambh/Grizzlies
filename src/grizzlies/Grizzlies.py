import pandas as pd
import os
import hashlib
import pickle
from collections import defaultdict, deque, Counter
import operator
from itertools import chain
from sortedcontainers import SortedDict

class Grizzlies:
    def __init__(self, data=None, create_scheme = "basic", threshold=5, windowsize=16, xval=10, drop_scheme = "none", index_type = 'hash', *args, **kwargs,):
        if isinstance(data, pd.DataFrame):
            self._df = data
        else:
            self._df = pd.DataFrame(data, *args, **kwargs)
        
        self._create_scheme = create_scheme
        
        os.makedirs("stats", exist_ok=True)
        self._stats_path = os.path.join("stats", f"{self._default_name()}.pkl")
        self._hash_indices = {}
        if index_type.lower() == 'ordered' or index_type.lower() == 'sorted':
            self._create_index = self._create_index_ordered  
            object.__setattr__(self, "evalfunc", self.evalfunc_ordered)
        else: # else is hash
            self._create_index = self._create_index_hash 
            object.__setattr__(self, "evalfunc", self.evalfunc_hash) 

        if self._create_scheme == "basic":
            self._access_counts = self._load_stats()
            if drop_scheme=="lru":
                self._everyxth = 0
                self._xval = xval
                self._lru_ctr = 0
                self._lru = {}
                self._increment_access_count = self._increment_access_count_lrubasic
            else: # none
                self._increment_access_count = self._increment_access_count_basic
                

        elif self._create_scheme == "sliding":
            self._window_size = windowsize
            self._everyxth = 0
            self._xval = xval
            self._increment_access_count = self._increment_access_count_sliding
            self._sliding_window = self._load_stats() # deque(maxlen=window)
            self._lru_ctr = 0
            self._lru = self._set_lru()
            if drop_scheme == "min":
                self._drop_index = self._drop_index_min
            elif drop_scheme == "lru":
                self._drop_index = self._drop_index_lru
            else:
                self._drop_index = self._drop_index_threshold

        
            
        # create_scheme = BASIC - does not ever delete
        self._threshold = threshold 
        self._max_indices = int(windowsize/threshold) # this is a heuristic, can use a diff way that is dynamic?

    # removed update_threshold, specify when creating instead. too much logic otherwise

#################################################################################################################
#                                        schema specific functions below                                        #
#################################################################################################################

    def _set_lru(self):
        i = 0
        lru = {}
        for item in self._sliding_window:
            lru[item] = i
            i += 1
        self._lru_ctr = i
        return lru

    def _drop_index_threshold(self, counts):
        keys_to_del = []
        # print(counts)
        for key in self._hash_indices.keys():
            if (key not in list(counts.keys())) or counts[key] < self._threshold:
                keys_to_del.append(key)
        for key in keys_to_del:
            del self._hash_indices[key]
        if len(keys_to_del) == 0:
            self._drop_index_lru(counts)

    
    def _drop_index_lru(self, counts):
        min_key, min_val = None, float('inf')
        for key in self._hash_indices.keys():
            if self._lru[key] < min_val:
                min_val = self._lru[key]
                min_key = key
        # print(self._lru)
        # print(min_key)
        del self._hash_indices[min_key]

    def _drop_index_min(self, counts):
        min_count = self._threshold
        min_key = None
        for key in self._hash_indices.keys():
            if (key not in list(counts.keys())):
                min_key = key
                break
            if counts[key] < min_count:
                min_count = counts[key]
                min_key = key
        # print(min_key)
        del self._hash_indices[min_key]
        

    def _increment_access_count_sliding(self, key):
        """Increase access count for the column for sliding scheme, updates lru"""
        # update lru - might be able to remove
        self._lru[key] = self._lru_ctr
        self._lru_ctr += 1

        self._everyxth += 1
        self._sliding_window.append(key)
        if self._everyxth % self._xval == 0:
            counts = Counter(self._sliding_window)
            # print("-------")
            for key, count in counts.items():
                # print(key, count)
                if (count >= self._threshold) and (key not in self._hash_indices.keys()):
                    # print(f"we need ta create one on {key}")
                    # print("here")
                    if len(self._hash_indices.keys()) >= self._max_indices:
                        # print("gotsta drop")
                        self._drop_index(counts)
                    self._create_index(key)


    def _increment_access_count_basic(self, key):
        """Increase access count for the column for basic scheme and check to create index"""
        if key not in self._access_counts:
            self._access_counts[key] = 0
        self._access_counts[key] += 1
        # print(f"------at {self._access_counts[key]} accesses------")

        if self._access_counts[key] >= self._threshold and key not in self._hash_indices:
            self._create_index(key)

    def _increment_access_count_lrubasic(self, key):
        """Increase access count for the column for basic scheme and check to create index"""
        self._lru[key] = self._lru_ctr
        self._lru_ctr += 1
        self._everyxth += 1

        if key not in self._access_counts:
            self._access_counts[key] = 0
        self._access_counts[key] += 1
        # print(f"------at {self._access_counts[key]} accesses------")
        if self._access_counts[key] >= self._threshold and key not in self._hash_indices:
            self._create_index(key)
        if self._everyxth % self._xval == 0 and len(self._hash_indices.keys()) >= self._max_indices:
            # print("ya boi boutta drop based on lru")
            self._drop_index_lru(counts=None)

        
    def _create_index_hash(self, key):
        """Create a hash index when a column is accessed frequently"""
        self._hash_indices[key] = defaultdict(list)
        for idx, value in self._df[key].items():
          self._hash_indices[key][value].append(idx)
        # self._hash_indices[key] = {value: idx for idx, value in self._df[key].items()}
        # print(f"------Hash index created for column: {key}------")
        # print(self._hash_indices[key])
    
    def _create_index_ordered(self, key):
        """Create a hash index when a column is accessed frequently"""
        self._hash_indices[key] = SortedDict()
        for idx, value in self._df[key].items():
          if value not in self._hash_indices[key]:
              self._hash_indices[key][value] = []
          self._hash_indices[key][value].append(idx)
        # print(f"------Sorted index created for column: {key}------")
        # print(self._hash_indices[key])

#################################################################################################################
#                                      NON-schema specific functions below                                      #
#################################################################################################################     

    def _default_name(self):
        # Sort by columns and index to avoid ordering affecting the hash
        hash_input = str(sorted(self._df.columns.tolist())) + str(self._df.shape) + self._create_scheme
        return hashlib.md5(hash_input.encode()).hexdigest()
    
    def _load_stats(self):
        # print("load stats called")
        if self._create_scheme == "basic":
            if os.path.exists(self._stats_path):
                with open(self._stats_path, 'rb') as f:
                    # print("------found something------")
                    return defaultdict(int, pickle.load(f))
            return defaultdict(int)
        elif self._create_scheme == "sliding":
            if os.path.exists(self._stats_path):
                with open(self._stats_path, 'rb') as f:
                    # print("------found something------")
                    return pickle.load(f)
            return deque(maxlen=self._window_size)

    
    def save(self):
        # print("Saved the stats")
        if self._create_scheme == "basic":
            with open(self._stats_path, 'wb') as f:
                pickle.dump(self._access_counts, f)
        elif self._create_scheme == "sliding":
            with open(self._stats_path, 'wb') as f:
                pickle.dump(self._sliding_window, f)

    def get_stats(self):
        if self._create_scheme == "basic":
            return dict(self._access_counts)
        elif self._create_scheme == "sliding":
            return dict(Counter(self._sliding_window))

        
    def evalfunc_ordered(self, colname, op, val):
        self._increment_access_count(colname)
        if colname in self._hash_indices:        
            if op == operator.gt:
                keys = self._hash_indices[colname].irange(minimum=val, inclusive=(False, True))
            elif op == operator.ge:
                keys = self._hash_indices[colname].irange(minimum=val, inclusive=(True, True))
            elif op == operator.lt:
                keys = self._hash_indices[colname].irange(maximum=val, inclusive=(False, True))
            elif op == operator.le:
                keys = self._hash_indices[colname].irange(maximum=val, inclusive=(True, True))
            elif op == operator.eq:
                return self._df.iloc[self._hash_indices[colname].get(val, [])]
            else:
                raise NotImplementedError("Unsupported operator")

            return self._df.iloc[list(chain.from_iterable(self._hash_indices[colname][k] for k in keys))]
        else:
            return self._df[op(self._df[colname], val)]

    
    def evalfunc_hash(self, colname, op ,val):
        self._increment_access_count(colname)
        if colname in self._hash_indices and op == operator.eq:
            return self._df.iloc[self._hash_indices[colname][val]]
        else:
            return self._df[op(self._df[colname], val)]
        
                
    def query(self, expr, **kwargs):
        # print(f"Intercepted query: {expr}") 
        return self._df.query(expr, **kwargs)

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

        # if key in self._hash_indices:
        #     # print(f"Using hash index for fast access on '{key}'")
        #     result = self._hash_indices[key]
        # if key not in self._df.columns:
        #     raise KeyError(f"Column '{key}' not found in DataFrame")

        self._increment_access_count(key)

        # print(self._access_counts)
        
        result = self._df[key]
        return Grizzlies(result) if isinstance(result, pd.DataFrame) else result

    def __setitem__(self, key, value):
        """Allow setting values like df['col'] = data."""
        # print("UPDATING SMTH")
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
            def __init__(self, parent):
                self._parent = parent
                self._df = self.parent._df

            def __getitem__(self, key):
                if isinstance(key, pd.Series) and key.dtype == bool:
                    return self._df.loc[key]

                cols = None
                if isinstance(key, tuple) and len(key) == 2:
                    cols = key[1]
                else:
                    cols = key

                if isinstance(cols, str):
                    if cols in self._df.columns:
                        self._parent._increment_access_count(cols)
                elif isinstance(cols, list):
                    for col in cols:
                        if col in self._df.columns:
                            self._parent._increment_access_count(col)

                result = self._df.loc[key]

                return Grizzlies(result) if isinstance(result, pd.DataFrame) else result

            def __setitem__(self, key, value):
                if isinstance(key, pd.Series) and key.dtype == bool:
                    self._df.loc[key] = value
                    return

                cols = None
                if isinstance(key, tuple) and len(key) == 2:
                    cols = key[1]
                else:
                    cols = key

                if isinstance(cols, str):
                    if cols in self._df.columns:
                        self._parent._increment_access_count(cols)
                elif isinstance(cols, list):
                    for col in cols:
                        if col in self._df.columns:
                            self._parent._increment_access_count(col)

                self._df.loc[key] = value

        return LocWrapper(self, self._parent._df.loc)

    @property
    def iloc(self):
        class IlocWrapper:
            def __init__(self, parent):
                self._parent = parent
                self._df = self.parent._df

            def __getitem__(self, key):
                if isinstance(key, tuple) and len(key) == 2:
                    cols = key[1]
                else:
                    cols = key

                if isinstance(cols, int):
                    if cols < len(self._df.columns):
                        colname = self._df.columns[cols]
                        self._parent._increment_access_count(colname)
                elif isinstance(cols, list):
                    for col in cols:
                        if col < len(self._df.columns):
                            colname = self._df.columns[col]
                            self._parent._increment_access_count(colname)

                result = self._df.iloc[key]

                return Grizzlies(result) if isinstance(result, pd.DataFrame) else result

            def __setitem__(self, key, value):
                if isinstance(key, tuple) and len(key) == 2:
                    cols = key[1]
                else:
                    cols = key

                if isinstance(cols, int):
                    if cols < len(self._df.columns):
                        colname = self._df.columns[cols]
                        self._parent._increment_access_count(colname)
                elif isinstance(cols, list):
                    for col in cols:
                        if col < len(self._df.columns):
                            colname = self._df.columns[col]
                            self._parent._increment_access_count(colname)

                self._df.iloc[key] = value

        return IlocWrapper(self, self._parent._df.iloc)


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

def merge(*args, **kwargs):
    return Grizzlies(pd.merge(*args, **kwargs))

def merge_ordered(*args, **kwargs):
    return Grizzlies(pd.merge_ordered(*args, **kwargs))

def concat(*args, **kwargs):
    return Grizzlies(pd.concat(*args, **kwargs))

def unique(*args, **kwargs):
    return pd.unique(*args, **kwargs)

def isnull(*args, **kwargs):
    return pd.isnull(*args, **kwargs)

def notnull(*args, **kwargs):
    return pd.notnull(*args, **kwargs)

def isna(*args, **kwargs):
    return pd.isna(*args, **kwargs)

def notna(*args, **kwargs):
    return pd.notna(*args, **kwargs)

def to_numeric(*args, **kwargs):
    return pd.to_numeric(*args, **kwargs)

def to_datetime(*args, **kwargs):
    return pd.to_datetime(*args, **kwargs)

def to_timedelta(*args, **kwargs):
    return pd.to_timedelta(*args, **kwargs)

def date_range(*args, **kwargs):
    return pd.date_range(*args, **kwargs)