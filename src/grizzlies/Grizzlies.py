"""
Grizzlies: A pandas wrapper with auto-indexing capabilities
"""

import pandas as pd
import time
import warnings
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np
import functools


class IndexTracker:
    """Tracks column access and determines when to create indexes"""
    
    def __init__(self, threshold: int = 3):
        self.access_counts = {}  # Column -> access count
        self.threshold = threshold
        self.indexed_columns = set()
    
    def record_access(self, column: str) -> bool:
        """
        Record access to a column and return True if an index should be created
        """
        if column in self.indexed_columns:
            return False
            
        if column not in self.access_counts:
            self.access_counts[column] = 1
        else:
            self.access_counts[column] += 1
        
        if self.access_counts[column] >= self.threshold:
            self.indexed_columns.add(column)
            return True
        return False
    
    def get_stats(self) -> Dict:
        """Return statistics about column access and indexing"""
        return {
            "access_counts": self.access_counts.copy(),
            "indexed_columns": list(self.indexed_columns),
            "threshold": self.threshold
        }


class GrizzlyLoc:
    """
    Class to handle .loc functionality with auto-indexing capabilities
    """
    
    def __init__(self, parent):
        self.parent = parent
    
    def __getitem__(self, key):
        """
        Get items using loc indexing with auto-indexing support
        """
        # If it's a tuple, it might be a (row, column) selection
        if isinstance(key, tuple) and len(key) == 2:
            _, col_key = key
            # If column key is a string, record access
            if isinstance(col_key, str) and col_key in self.parent._df.columns:
                if self.parent._auto_index and self.parent._index_tracker.record_access(col_key):
                    self.parent._create_index(col_key)
            # If column key is a list of strings, record access for each
            elif isinstance(col_key, list):
                for col in col_key:
                    if isinstance(col, str) and col in self.parent._df.columns:
                        if self.parent._auto_index and self.parent._index_tracker.record_access(col):
                            self.parent._create_index(col)
        
        # If it's a string, it's a column name
        elif isinstance(key, str) and key in self.parent._df.columns:
            if self.parent._auto_index and self.parent._index_tracker.record_access(key):
                self.parent._create_index(key)
        
        result = self.parent._df.loc[key]
        
        # Wrap DataFrame results in GrizzlyFrame
        if isinstance(result, pd.DataFrame):
            return GrizzlyFrame.from_pandas(result, 
                                           self.parent._index_tracker.threshold, 
                                           self.parent._auto_index)
        return result  # Return as is if it's a Series or other type
    
    def __setitem__(self, key, value):
        """
        Set items using loc indexing
        """
        self.parent._df.loc[key] = value
        
        # If indexing by column, update index if needed
        if isinstance(key, tuple) and len(key) == 2:
            _, col_key = key
            if isinstance(col_key, str) and col_key in self.parent._index_tracker.indexed_columns:
                self.parent._create_index(col_key)
        elif isinstance(key, str) and key in self.parent._index_tracker.indexed_columns:
            self.parent._create_index(key)


class GrizzlyILoc:
    """
    Class to handle .iloc functionality - pure integer location
    """
    
    def __init__(self, parent):
        self.parent = parent
    
    def __getitem__(self, key):
        """
        Get items using iloc indexing
        """
        result = self.parent._df.iloc[key]
        
        # Wrap DataFrame results in GrizzlyFrame
        if isinstance(result, pd.DataFrame):
            return GrizzlyFrame.from_pandas(result, 
                                           self.parent._index_tracker.threshold, 
                                           self.parent._auto_index)
        return result  # Return as is if it's a Series or other type
    
    def __setitem__(self, key, value):
        """
        Set items using iloc indexing (no auto-indexing since it's integer-based)
        """
        self.parent._df.iloc[key] = value


class GrizzlyFrame:
    """
    A wrapper around pandas DataFrame that provides auto-indexing capabilities
    """
    
    def __init__(self, data=None, index_threshold: int = 3, auto_index: bool = True):
        """
        Initialize a GrizzlyFrame
        
        Parameters:
        -----------
        data : various formats accepted by pandas.DataFrame
            Data to initialize the DataFrame with
        index_threshold : int
            Number of accesses to a column before creating an index
        auto_index : bool
            Whether to automatically create indexes
        """
        self._df = pd.DataFrame(data)
        self._index_tracker = IndexTracker(threshold=index_threshold)
        self._auto_index = auto_index
        self._query_times = []  # Store query execution times
        self._loc = GrizzlyLoc(self)
        self._iloc = GrizzlyILoc(self)
    
    @property
    def df(self) -> pd.DataFrame:
        """Access the underlying pandas DataFrame"""
        return self._df
    
    @property
    def loc(self) -> GrizzlyLoc:
        """Label-based indexing"""
        return self._loc
    
    @property
    def iloc(self) -> GrizzlyILoc:
        """Integer-based indexing"""
        return self._iloc
    
    def __getitem__(self, key):
        """
        Handle column access with bracket notation and auto-indexing
        """
        # If it's a string, it's likely a column access
        if isinstance(key, str):
            if self._auto_index and self._index_tracker.record_access(key):
                self._create_index(key)
            return self._df[key]  # This returns a Series, which is fine
        
        # Handle boolean masks (e.g., df[df['col'] > 5])
        elif isinstance(key, pd.Series) and key.dtype == bool:
            # Try to identify which columns were used in the boolean mask
            if hasattr(key, 'name') and key.name in self._df.columns:
                if self._auto_index and self._index_tracker.record_access(key.name):
                    self._create_index(key.name)
            
            # The key difference: wrap the result in a GrizzlyFrame before returning
            result_df = self._df[key]
            return GrizzlyFrame.from_pandas(result_df, self._index_tracker.threshold, self._auto_index)
        
        # Handle slice objects and other indexers
        else:
            result = self._df[key]
            # If the result is a DataFrame, wrap it in a GrizzlyFrame
            if isinstance(result, pd.DataFrame):
                return GrizzlyFrame.from_pandas(result, self._index_tracker.threshold, self._auto_index)
            return result  # Return as is if it's a Series or other type`
        
    def __setitem__(self, key, value):
        """Handle column assignment with bracket notation"""
        self._df[key] = value
        # If this column was indexed, we need to recreate the index
        if key in self._index_tracker.indexed_columns:
            self._create_index(key)
    
    def _create_index(self, column: str) -> None:
        """Create an index on a column"""
        if column not in self._df.columns:
            raise KeyError(f"Column '{column}' not found in DataFrame")
        
        # Check if column is appropriate for indexing
        if self._df[column].nunique() < len(self._df) * 0.01:
            # Too few unique values, might not be worth indexing
            warnings.warn(f"Column '{column}' has few unique values. Indexing may not improve performance.")
        
        # Use pandas set_index but don't drop the column
        try:
            # Create the index in-place if it doesn't exist
            if column not in self._df.index.names:
                self._df.set_index(column, drop=False, append=True, inplace=True)
                print(f"Created index on column '{column}'")
        except Exception as e:
            warnings.warn(f"Failed to create index on column '{column}': {str(e)}")
    
    def query(self, expr: str) -> pd.DataFrame:
        """
        Query the DataFrame with auto-indexing optimization
        
        Parameters:
        -----------
        expr : str
            Query expression (same syntax as pandas.DataFrame.query)
        
        Returns:
        --------
        pd.DataFrame
            Result of the query
        """
        start_time = time.time()
        
        # Simple parser to identify columns in the query
        # This is a basic implementation; a more robust parser would be needed for complex queries
        possible_columns = set()
        parts = expr.replace('(', ' ').replace(')', ' ').replace('==', ' ').replace('!=', ' ').replace('<', ' ').replace('>', ' ').replace('<=', ' ').replace('>=', ' ').split()
        
        for part in parts:
            part = part.strip()
            if part in self._df.columns:
                possible_columns.add(part)
                if self._auto_index and self._index_tracker.record_access(part):
                    self._create_index(part)
        
        # Execute the query
        result = self._df.query(expr)
        
        end_time = time.time()
        self._query_times.append((expr, end_time - start_time))
        
        # Return a GrizzlyFrame instead of a pandas DataFrame
        return GrizzlyFrame.from_pandas(result, self._index_tracker.threshold, self._auto_index)
    
    def ensure_index(self, column: str) -> None:
        """
        Manually ensure a column is indexed
        
        Parameters:
        -----------
        column : str
            Column to index
        """
        if column not in self._index_tracker.indexed_columns:
            self._index_tracker.indexed_columns.add(column)
            self._create_index(column)
    
    def drop_index(self, column: str) -> None:
        """
        Drop an index on a column
        
        Parameters:
        -----------
        column : str
            Column to drop index for
        """
        if column in self._index_tracker.indexed_columns:
            # Reset index if needed
            if column in self._df.index.names:
                self._df = self._df.reset_index(column)
            self._index_tracker.indexed_columns.remove(column)
            print(f"Dropped index on column '{column}'")
    
    def get_index_stats(self) -> Dict:
        """
        Get statistics about index usage
        
        Returns:
        --------
        Dict
            Dictionary with index statistics
        """
        stats = self._index_tracker.get_stats()
        stats["query_times"] = self._query_times[-10:]  # Last 10 queries
        return stats
    
    def head(self, n: int = 5):
        """Show the first n rows"""
        result = self._df.head(n)
        return GrizzlyFrame.from_pandas(result, self._index_tracker.threshold, self._auto_index)

    def tail(self, n: int = 5):
        """Show the last n rows"""
        result = self._df.tail(n)
        return GrizzlyFrame.from_pandas(result, self._index_tracker.threshold, self._auto_index)
    
    @property
    def shape(self) -> Tuple[int, int]:
        """Return the shape of the DataFrame"""
        return self._df.shape
    
    @property
    def columns(self) -> pd.Index:
        """Return the columns of the DataFrame"""
        return self._df.columns
    
    def describe(self) -> pd.DataFrame:
        """Generate descriptive statistics"""
        return self._df.describe()
    
    def info(self) -> None:
        """Print concise summary of the DataFrame"""
        print("GrizzlyFrame with auto-indexing")
        print(f"Auto-indexing: {self._auto_index}")
        print(f"Index threshold: {self._index_tracker.threshold}")
        print(f"Indexed columns: {list(self._index_tracker.indexed_columns)}")
        print("\nDataFrame Info:")
        return self._df.info()
    
    # Null value handling methods
    def isna(self) -> pd.DataFrame:
        """
        Detect missing values
        
        Returns:
        --------
        pd.DataFrame
            Boolean DataFrame indicating which cells are missing
        """
        return self._df.isna()
    
    def isnull(self) -> pd.DataFrame:
        """
        Alias for isna
        """
        return self.isna()
    
    def notna(self) -> pd.DataFrame:
        """
        Detect non-missing values
        
        Returns:
        --------
        pd.DataFrame
            Boolean DataFrame indicating which cells are not missing
        """
        return self._df.notna()
    
    def notnull(self) -> pd.DataFrame:
        """
        Alias for notna
        """
        return self.notna()
    
    def dropna(self, axis=0, how='any', thresh=None, subset=None, inplace=False) -> Optional['GrizzlyFrame']:
        """
        Remove missing values
        
        Parameters:
        -----------
        axis : {0 or 'index', 1 or 'columns'}, default 0
            Determine if rows or columns which contain missing values are removed
        how : {'any', 'all'}, default 'any'
            Determine if row or column is removed from DataFrame, when we have at least one NA or all NA
        thresh : int, optional
            Require that many non-NA values
        subset : array-like, optional
            Labels along other axis to consider
        inplace : bool, default False
            If True, do operation inplace and return None
            
        Returns:
        --------
        GrizzlyFrame or None
            DataFrame with NA entries dropped or None if inplace=True
        """
        # Record column access for columns in subset if specified
        if subset is not None:
            if isinstance(subset, str):
                subset = [subset]
            for col in subset:
                if col in self._df.columns and self._auto_index and self._index_tracker.record_access(col):
                    self._create_index(col)
        
        # Perform the operation
        result = self._df.dropna(axis=axis, how=how, thresh=thresh, subset=subset, inplace=False)
        
        if inplace:
            self._df = result
            return None
        else:
            return GrizzlyFrame.from_pandas(result, self._index_tracker.threshold, self._auto_index)
    
    def fillna(self, value=None, method=None, axis=None, inplace=False, limit=None, downcast=None) -> Optional['GrizzlyFrame']:
        """
        Fill NA/NaN values using the specified method
        
        Parameters:
        -----------
        value : scalar, dict, Series, or DataFrame
            Value to use to fill holes
        method : {'backfill', 'bfill', 'pad', 'ffill', None}, default None
            Method to use for filling holes in reindexed Series
        axis : {0 or 'index', 1 or 'columns'}
            Axis along which to fill missing values
        inplace : bool, default False
            If True, fill in-place. Note: this will modify any other views on this object
        limit : int, default None
            If method is specified, this is the maximum number of consecutive NaN values to forward/backward fill
        downcast : dict, default is None
            A dict of item->dtype of what to downcast if possible
            
        Returns:
        --------
        GrizzlyFrame or None
            Object with missing values filled or None if inplace=True
        """
        # Handle dictionary of values to ensure auto-indexing
        if isinstance(value, dict):
            for col in value.keys():
                if col in self._df.columns and self._auto_index and self._index_tracker.record_access(col):
                    self._create_index(col)
        
        # Perform the operation
        result = self._df.fillna(value=value, method=method, axis=axis, inplace=False, limit=limit, downcast=downcast)
        
        if inplace:
            self._df = result
            return None
        else:
            return GrizzlyFrame.from_pandas(result, self._index_tracker.threshold, self._auto_index)
    
    # Merge functionality
    def merge(self, right, how='inner', on=None, left_on=None, right_on=None, 
              left_index=False, right_index=False, sort=False, 
              suffixes=('_x', '_y'), copy=True, indicator=False,
              validate=None) -> 'GrizzlyFrame':
        """
        Merge GrizzlyFrame objects with smart indexing
        
        Parameters:
        -----------
        right : GrizzlyFrame, pandas.DataFrame, or other DataFrame-like object
            Object to merge with
        how : {'left', 'right', 'outer', 'inner'}, default 'inner'
            Type of merge to be performed
        on : label or list of labels, optional
            Column(s) to join on. Must be found in both DataFrames.
        left_on : label or list of labels, optional
            Column(s) from the left DataFrame to use as keys
        right_on : label or list of labels, optional
            Column(s) from the right DataFrame to use as keys
        left_index : bool, default False
            Use the index from the left DataFrame as the join key
        right_index : bool, default False
            Use the index from the right DataFrame as the join key
        sort : bool, default False
            Sort the join keys lexicographically in the result
        suffixes : tuple of (str, str), default ('_x', '_y')
            Suffix to apply to overlapping column names
        copy : bool, default True
            If False, avoid copy if possible
        indicator : bool or str, default False
            Add a column to output to indicate merge source
        validate : str, optional
            Validate merge keys. Options: 'one_to_one', 'one_to_many', 'many_to_one', 'many_to_many'
            
        Returns:
        --------
        GrizzlyFrame
            Merged DataFrame
        """
        # Smart auto-indexing: create indexes on join columns before merging
        if on is not None:
            columns = [on] if isinstance(on, str) else on
            for col in columns:
                if col in self._df.columns and self._auto_index:
                    self.ensure_index(col)
                    
        if left_on is not None:
            columns = [left_on] if isinstance(left_on, str) else left_on
            for col in columns:
                if col in self._df.columns and self._auto_index:
                    self.ensure_index(col)
        
        # Extract the pandas DataFrame from right if it's a GrizzlyFrame
        right_df = right._df if isinstance(right, GrizzlyFrame) else right
        
        # Also add indexes on the right side if it's a GrizzlyFrame
        if isinstance(right, GrizzlyFrame) and right._auto_index:
            if on is not None:
                columns = [on] if isinstance(on, str) else on
                for col in columns:
                    if col in right._df.columns:
                        right.ensure_index(col)
                        
            if right_on is not None:
                columns = [right_on] if isinstance(right_on, str) else right_on
                for col in columns:
                    if col in right._df.columns:
                        right.ensure_index(col)
        
        # Perform the merge
        result_df = self._df.merge(
            right_df, how=how, on=on, left_on=left_on, right_on=right_on,
            left_index=left_index, right_index=right_index, sort=sort,
            suffixes=suffixes, copy=copy, indicator=indicator, validate=validate
        )
        
        # Create a new GrizzlyFrame with the result
        result = GrizzlyFrame.from_pandas(result_df, self._index_tracker.threshold, self._auto_index)
        
        # Preserve indexing information from both frames
        # For overlapping columns, we prioritize the left (self) frame's indexing
        if isinstance(right, GrizzlyFrame):
            # Copy over information about indexed columns that don't conflict with self
            for col in right._index_tracker.indexed_columns:
                # Check if the column exists in the result (may have been renamed with suffixes)
                if col in result._df.columns and col not in self._index_tracker.indexed_columns:
                    result.ensure_index(col)
                # Check for columns that were renamed with suffixes
                elif col not in result._df.columns:
                    # Try with the right suffix
                    suffixed_col = f"{col}{suffixes[1]}"
                    if suffixed_col in result._df.columns:
                        result.ensure_index(suffixed_col)
        
        return result
    
    # Common operations with auto-indexing
    def groupby(self, by=None, **kwargs):
        """
        Group DataFrame using a mapper or by a Series of columns
        with auto-indexing support
        """
        if isinstance(by, str) and by in self._df.columns:
            if self._auto_index and self._index_tracker.record_access(by):
                self._create_index(by)
        elif isinstance(by, list):
            for col in by:
                if isinstance(col, str) and col in self._df.columns:
                    if self._auto_index and self._index_tracker.record_access(col):
                        self._create_index(col)
                    
        return self._df.groupby(by, **kwargs)
    
    def sort_values(self, by, **kwargs):
        """
        Sort by the values along either axis with auto-indexing support
        """
        cols = [by] if isinstance(by, str) else by
        
        for col in cols:
            if col in self._df.columns:
                if self._auto_index and self._index_tracker.record_access(col):
                    self._create_index(col)
        
        return self._df.sort_values(by, **kwargs)
    
    def to_pandas(self) -> pd.DataFrame:
        """
        Convert back to a standard pandas DataFrame
        
        Returns:
        --------
        pd.DataFrame
            The underlying pandas DataFrame with all indexes reset
        """
        # Reset all indexes to get a clean DataFrame
        if len(self._df.index.names) > 1 or self._df.index.name is not None:
            return self._df.reset_index()
        return self._df.copy()
    
    @classmethod
    def from_pandas(cls, df: pd.DataFrame, index_threshold: int = 3, auto_index: bool = True):
        """
        Create a GrizzlyFrame from a pandas DataFrame
        
        Parameters:
        -----------
        df : pd.DataFrame
            Pandas DataFrame to wrap
        index_threshold : int
            Number of accesses to a column before creating an index
        auto_index : bool
            Whether to automatically create indexes
        
        Returns:
        --------
        GrizzlyFrame
            A new GrizzlyFrame wrapping the provided DataFrame
        """
        gf = cls(None, index_threshold, auto_index)
        gf._df = df.copy()
        return gf
    
    def __repr__(self) -> str:
        """String representation"""
        indexed = f"auto-indexed on {list(self._index_tracker.indexed_columns)}" if self._index_tracker.indexed_columns else "no auto-indexes yet"
        return f"GrizzlyFrame({indexed}):\n{self._df.__repr__()}"


# Utility functions for the module
def read_csv(filepath_or_buffer, **kwargs) -> GrizzlyFrame:
    """
    Read a comma-separated values (csv) file into GrizzlyFrame
    
    Parameters:
    -----------
    filepath_or_buffer : str, path object or file-like object
        Same as pandas.read_csv
    **kwargs : dict
        Additional arguments to pass to pandas.read_csv
    
    Returns:
    --------
    GrizzlyFrame
        A GrizzlyFrame containing the data
    """
    df = pd.read_csv(filepath_or_buffer, **kwargs)
    return GrizzlyFrame.from_pandas(df)


def read_excel(io, **kwargs) -> GrizzlyFrame:
    """
    Read an Excel file into GrizzlyFrame
    
    Parameters:
    -----------
    io : str, path object or file-like object
        Same as pandas.read_excel
    **kwargs : dict
        Additional arguments to pass to pandas.read_excel
    
    Returns:
    --------
    GrizzlyFrame
        A GrizzlyFrame containing the data
    """
    df = pd.read_excel(io, **kwargs)
    return GrizzlyFrame.from_pandas(df)


# Merge utility function
def merge(left, right, how='inner', on=None, left_on=None, right_on=None,
          left_index=False, right_index=False, sort=False,
          suffixes=('_x', '_y'), copy=True, indicator=False,
          validate=None) -> GrizzlyFrame:
    """
    Merge GrizzlyFrame or pandas DataFrame objects
    
    Parameters:
    -----------
    left : GrizzlyFrame or pandas.DataFrame
        Left object
    right : GrizzlyFrame or pandas.DataFrame
        Right object
    
    Additional parameters are the same as GrizzlyFrame.merge()
    
    Returns:
    --------
    GrizzlyFrame
        Merged DataFrame
    """
    # Convert to GrizzlyFrame if needed
    if not isinstance(left, GrizzlyFrame):
        left = GrizzlyFrame.from_pandas(left)
    
    return left.merge(
        right, how=how, on=on, left_on=left_on, right_on=right_on,
        left_index=left_index, right_index=right_index, sort=sort,
        suffixes=suffixes, copy=copy, indicator=indicator, validate=validate
    )

##############################################################################################################################################################################
##############################################################################################################################################################################
##############################################################################################################################################################################
##############################################################################################################################################################################
##############################################################################################################################################################################
##############################################################################################################################################################################

# import pandas as pd
# import os
# import hashlib
# import pickle
# from collections import defaultdict, deque, Counter
# import operator
# from itertools import chain
# from sortedcontainers import SortedDict

# class Grizzlies:
#     def __init__(self, data=None, create_scheme = "basic", threshold=5, windowsize=16, xval=10, drop_scheme = "none", index_type = 'hash', *args, **kwargs,):
#         # Initialize df either with a new empty df or a given pandas df
#         if isinstance(data, pd.DataFrame):
#             self._df = data
#         else:
#             self._df = pd.DataFrame(data, *args, **kwargs)
        
#         # initialize create scheme
#         self._create_scheme = create_scheme
        
#         # create folder to store usage stats
#         os.makedirs("stats", exist_ok=True)
#         self._stats_path = os.path.join("stats", f"{self._default_name()}.pkl")

#         # initialize index type
#         self._hash_indices = {}
#         if index_type.lower() == 'ordered' or index_type.lower() == 'sorted':
#             self._create_index = self._create_index_ordered  
#             object.__setattr__(self, "evalfunc", self.evalfunc_ordered)
#         else: # else is hash
#             self._create_index = self._create_index_hash 
#             object.__setattr__(self, "evalfunc", self.evalfunc_hash) 

#         # initialize vars for basic scheme and drop schemes
#         if self._create_scheme == "basic":
#             self._access_counts = self._load_stats()
#             if drop_scheme=="lru":
#                 self._everyxth = 0
#                 self._xval = xval
#                 self._lru_ctr = 0
#                 self._lru = {}
#                 self._increment_access_count = self._increment_access_count_lrubasic
#             else: # none
#                 self._increment_access_count = self._increment_access_count_basic
                
#         # initialize vars for sliding window scheme and drop schemes
#         elif self._create_scheme == "sliding":
#             self._window_size = windowsize
#             self._everyxth = 0
#             self._xval = xval
#             self._increment_access_count = self._increment_access_count_sliding
#             self._sliding_window = self._load_stats() # deque(maxlen=window)
#             self._lru_ctr = 0
#             self._lru = self._set_lru()
#             if drop_scheme == "min":
#                 self._drop_index = self._drop_index_min
#             elif drop_scheme == "lru":
#                 self._drop_index = self._drop_index_lru
#             else:
#                 self._drop_index = self._drop_index_threshold

#         # create_scheme = BASIC - does not ever delete
#         self._threshold = threshold 
#         self._max_indices = int(windowsize/threshold) # this is a heuristic, can use a diff way that is dynamic?

#     # removed update_threshold, specify when creating instead. too much logic otherwise

# #################################################################################################################
# #                                        schema specific functions below                                        #
# #################################################################################################################

#     def _set_lru(self):
#         i = 0
#         lru = {}
#         for item in self._sliding_window:
#             lru[item] = i
#             i += 1
#         self._lru_ctr = i
#         return lru

#     def _drop_index_threshold(self, counts):
#         keys_to_del = []
#         # print(counts)
#         for key in self._hash_indices.keys():
#             if (key not in list(counts.keys())) or counts[key] < self._threshold:
#                 keys_to_del.append(key)
#         for key in keys_to_del:
#             del self._hash_indices[key]
#         if len(keys_to_del) == 0:
#             self._drop_index_lru(counts)

    
#     def _drop_index_lru(self, counts):
#         min_key, min_val = None, float('inf')
#         for key in self._hash_indices.keys():
#             if self._lru[key] < min_val:
#                 min_val = self._lru[key]
#                 min_key = key
#         # print(self._lru)
#         # print(min_key)
#         del self._hash_indices[min_key]

#     def _drop_index_min(self, counts):
#         min_count = self._threshold
#         min_key = None
#         for key in self._hash_indices.keys():
#             if (key not in list(counts.keys())):
#                 min_key = key
#                 break
#             if counts[key] < min_count:
#                 min_count = counts[key]
#                 min_key = key
#         # print(min_key)
#         del self._hash_indices[min_key]
        

#     def _increment_access_count_sliding(self, key):
#         """Increase access count for the column for sliding scheme, updates lru"""
#         # update lru - might be able to remove
#         self._lru[key] = self._lru_ctr
#         self._lru_ctr += 1

#         self._everyxth += 1
#         self._sliding_window.append(key)
#         if self._everyxth % self._xval == 0:
#             counts = Counter(self._sliding_window)
#             # print("-------")
#             for key, count in counts.items():
#                 # print(key, count)
#                 if (count >= self._threshold) and (key not in self._hash_indices.keys()):
#                     # print(f"we need ta create one on {key}")
#                     # print("here")
#                     if len(self._hash_indices.keys()) >= self._max_indices:
#                         # print("gotsta drop")
#                         self._drop_index(counts)
#                     self._create_index(key)


#     def _increment_access_count_basic(self, key):
#         """Increase access count for the column for basic scheme and check to create index"""
#         if key not in self._access_counts:
#             self._access_counts[key] = 0
#         self._access_counts[key] += 1
#         # print(f"------at {self._access_counts[key]} accesses------")

#         if self._access_counts[key] >= self._threshold and key not in self._hash_indices:
#             self._create_index(key)

#     def _increment_access_count_lrubasic(self, key):
#         """Increase access count for the column for basic scheme and check to create index"""
#         self._lru[key] = self._lru_ctr
#         self._lru_ctr += 1
#         self._everyxth += 1

#         if key not in self._access_counts:
#             self._access_counts[key] = 0
#         self._access_counts[key] += 1
#         # print(f"------at {self._access_counts[key]} accesses------")
#         if self._access_counts[key] >= self._threshold and key not in self._hash_indices:
#             self._create_index(key)
#         if self._everyxth % self._xval == 0 and len(self._hash_indices.keys()) >= self._max_indices:
#             # print("ya boi boutta drop based on lru")
#             self._drop_index_lru(counts=None)

        
#     def _create_index_hash(self, key):
#         """Create a hash index when a column is accessed frequently"""
#         self._hash_indices[key] = defaultdict(list)
#         for idx, value in self._df[key].items():
#           self._hash_indices[key][value].append(idx)
#         # self._hash_indices[key] = {value: idx for idx, value in self._df[key].items()}
#         # print(f"------Hash index created for column: {key}------")
#         # print(self._hash_indices[key])
    
#     def _create_index_ordered(self, key):
#         """Create a hash index when a column is accessed frequently"""
#         self._hash_indices[key] = SortedDict()
#         for idx, value in self._df[key].items():
#           if value not in self._hash_indices[key]:
#               self._hash_indices[key][value] = []
#           self._hash_indices[key][value].append(idx)
#         # print(f"------Sorted index created for column: {key}------")
#         # print(self._hash_indices[key])

# #################################################################################################################
# #                                      NON-schema specific functions below                                      #
# #################################################################################################################     

#     def _default_name(self):
#         # Sort by columns and index to avoid ordering affecting the hash
#         hash_input = str(sorted(self._df.columns.tolist())) + str(self._df.shape) + self._create_scheme
#         return hashlib.md5(hash_input.encode()).hexdigest()
    
#     def _load_stats(self):
#         # print("load stats called")
#         if self._create_scheme == "basic":
#             if os.path.exists(self._stats_path):
#                 with open(self._stats_path, 'rb') as f:
#                     # print("------found something------")
#                     return defaultdict(int, pickle.load(f))
#             return defaultdict(int)
#         elif self._create_scheme == "sliding":
#             if os.path.exists(self._stats_path):
#                 with open(self._stats_path, 'rb') as f:
#                     # print("------found something------")
#                     return pickle.load(f)
#             return deque(maxlen=self._window_size)

    
#     def save(self):
#         # print("Saved the stats")
#         if self._create_scheme == "basic":
#             with open(self._stats_path, 'wb') as f:
#                 pickle.dump(self._access_counts, f)
#         elif self._create_scheme == "sliding":
#             with open(self._stats_path, 'wb') as f:
#                 pickle.dump(self._sliding_window, f)

#     def get_stats(self):
#         if self._create_scheme == "basic":
#             return dict(self._access_counts)
#         elif self._create_scheme == "sliding":
#             return dict(Counter(self._sliding_window))

        
#     def evalfunc_ordered(self, colname, op, val):
#         self._increment_access_count(colname)
#         if colname in self._hash_indices:        
#             if op == operator.gt:
#                 keys = self._hash_indices[colname].irange(minimum=val, inclusive=(False, True))
#             elif op == operator.ge:
#                 keys = self._hash_indices[colname].irange(minimum=val, inclusive=(True, True))
#             elif op == operator.lt:
#                 keys = self._hash_indices[colname].irange(maximum=val, inclusive=(False, True))
#             elif op == operator.le:
#                 keys = self._hash_indices[colname].irange(maximum=val, inclusive=(True, True))
#             elif op == operator.eq:
#                 return self._df.iloc[self._hash_indices[colname].get(val, [])]
#             else:
#                 raise NotImplementedError("Unsupported operator")

#             return self._df.iloc[list(chain.from_iterable(self._hash_indices[colname][k] for k in keys))]
#         else:
#             return self._df[op(self._df[colname], val)]

    
#     def evalfunc_hash(self, colname, op ,val):
#         self._increment_access_count(colname)
#         if colname in self._hash_indices and op == operator.eq:
#             return self._df.iloc[self._hash_indices[colname][val]]
#         else:
#             return self._df[op(self._df[colname], val)]
        
                
#     def query(self, expr, **kwargs):
#         # print(f"Intercepted query: {expr}") 
#         return self._df.query(expr, **kwargs)

#     def __getitem__(self, key):
#         """Support indexing like df['column'] or df[['col1', 'col2']]."""
#         if isinstance(key, list):
#             result = self._df[key]
#             for k in key:
#               self._increment_access_count(k)
#             return Grizzlies(result)

#         # handle boolean series ex. df[df['col1'] == x]
#         if isinstance(key, pd.Series) and key.dtype == bool:
#           self._increment_access_count(key.name)
#           return Grizzlies(self._df[key])

#         # if key in self._hash_indices:
#         #     # print(f"Using hash index for fast access on '{key}'")
#         #     result = self._hash_indices[key]
#         # if key not in self._df.columns:
#         #     raise KeyError(f"Column '{key}' not found in DataFrame")

#         self._increment_access_count(key)

#         # print(self._access_counts)
        
#         result = self._df[key]
#         return Grizzlies(result) if isinstance(result, pd.DataFrame) else result

#     def __setitem__(self, key, value):
#         """Allow setting values like df['col'] = data."""
#         # print("UPDATING SMTH")
#         self._increment_access_count(key)
#         self._df[key] = value

# #################################################################################################################
# #                                    do not edit overloaded functions below!                                    #
# #################################################################################################################

#     def __getattr__(self, attr):
#         """Delegate attribute access to the underlying DataFrame."""
#         return getattr(self._df, attr)
    
#     def __repr__(self):
#         """Ensure the object prints like a normal DataFrame."""
#         return repr(self._df)

#     def head(self, n=5):
#         return Grizzlies(self._df.head(n))

#     def tail(self, n=5):
#         return Grizzlies(self._df.tail(n))

#     def dropna(self, *args, **kwargs):
#         return Grizzlies(self._df.dropna(*args, **kwargs))

#     def fillna(self, *args, **kwargs):
#         return Grizzlies(self._df.fillna(*args, **kwargs))

#     def isna(self):
#         return self._df.isna()

#     def mean(self, *args, **kwargs):
#         return self._df.mean(*args, **kwargs)

#     def merge(self, right, *args, **kwargs):
#         if isinstance(right, Grizzlies):
#             right = right._df
#         return Grizzlies(pd.merge(self._df, right, *args, **kwargs))

#     def groupby(self, *args, **kwargs):
#         return self._df.groupby(*args, **kwargs)

#     def isin(self, values):
#         return self._df.isin(values)

#     @property
#     def loc(self):
#         class LocWrapper:
#             def __init__(self, parent):
#                 self._parent = parent
#                 self._df = self.parent._df

#             def __getitem__(self, key):
#                 if isinstance(key, pd.Series) and key.dtype == bool:
#                     return self._df.loc[key]

#                 cols = None
#                 if isinstance(key, tuple) and len(key) == 2:
#                     cols = key[1]
#                 else:
#                     cols = key

#                 if isinstance(cols, str):
#                     if cols in self._df.columns:
#                         self._parent._increment_access_count(cols)
#                 elif isinstance(cols, list):
#                     for col in cols:
#                         if col in self._df.columns:
#                             self._parent._increment_access_count(col)

#                 result = self._df.loc[key]

#                 return Grizzlies(result) if isinstance(result, pd.DataFrame) else result

#             def __setitem__(self, key, value):
#                 if isinstance(key, pd.Series) and key.dtype == bool:
#                     self._df.loc[key] = value
#                     return

#                 cols = None
#                 if isinstance(key, tuple) and len(key) == 2:
#                     cols = key[1]
#                 else:
#                     cols = key

#                 if isinstance(cols, str):
#                     if cols in self._df.columns:
#                         self._parent._increment_access_count(cols)
#                 elif isinstance(cols, list):
#                     for col in cols:
#                         if col in self._df.columns:
#                             self._parent._increment_access_count(col)

#                 self._df.loc[key] = value

#         return LocWrapper(self, self._parent._df.loc)

#     @property
#     def iloc(self):
#         class IlocWrapper:
#             def __init__(self, parent):
#                 self._parent = parent
#                 self._df = self.parent._df

#             def __getitem__(self, key):
#                 if isinstance(key, tuple) and len(key) == 2:
#                     cols = key[1]
#                 else:
#                     cols = key

#                 if isinstance(cols, int):
#                     if cols < len(self._df.columns):
#                         colname = self._df.columns[cols]
#                         self._parent._increment_access_count(colname)
#                 elif isinstance(cols, list):
#                     for col in cols:
#                         if col < len(self._df.columns):
#                             colname = self._df.columns[col]
#                             self._parent._increment_access_count(colname)

#                 result = self._df.iloc[key]

#                 return Grizzlies(result) if isinstance(result, pd.DataFrame) else result

#             def __setitem__(self, key, value):
#                 if isinstance(key, tuple) and len(key) == 2:
#                     cols = key[1]
#                 else:
#                     cols = key

#                 if isinstance(cols, int):
#                     if cols < len(self._df.columns):
#                         colname = self._df.columns[cols]
#                         self._parent._increment_access_count(colname)
#                 elif isinstance(cols, list):
#                     for col in cols:
#                         if col < len(self._df.columns):
#                             colname = self._df.columns[col]
#                             self._parent._increment_access_count(colname)

#                 self._df.iloc[key] = value

#         return IlocWrapper(self, self._parent._df.iloc)


#     def at(self, *args):
#         return self._df.at[*args]

#     def iat(self, *args):
#         return self._df.iat[*args]

#     def __setattr__(self, name, value):
#         """Allow setting attributes on the underlying DataFrame."""
#         if name.startswith("_"):
#             super().__setattr__(name, value)
#         elif hasattr(self._df, name):
#             setattr(self._df, name, value)
#         else:
#             raise AttributeError(
#                 f"Cannot set unknown attribute '{name}'. Use item assignment like df['{name}'] = ... to add new columns."
#             )

#     def __iter__(self):
#         """Support iteration over columns like a normal DataFrame."""
#         return iter(self._df)

#     def __len__(self):
#         """Support len(df)."""
#         return len(self._df)

#     def __contains__(self, item):
#         """Support 'col' in df."""
#         return item in self._df


#     def __delitem__(self, key):
#         """Support del df[col]."""
#         del self._df[key]

#     def __call__(self, *args, **kwargs):
#         """Allow calling methods directly on the wrapped DataFrame."""
#         return self._df(*args, **kwargs)

#     def __eq__(self, other):
#         """Support df == value."""
#         return self._df == (other._df if isinstance(other, Grizzlies) else other)
    

# # Module-level functions
# def read_csv(*args, **kwargs):
#     return Grizzlies(pd.read_csv(*args, **kwargs))

# def read_excel(*args, **kwargs):
#     return Grizzlies(pd.read_excel(*args, **kwargs))

# def read_json(*args, **kwargs):
#     return Grizzlies(pd.read_json(*args, **kwargs))

# def read_parquet(*args, **kwargs):
#     return Grizzlies(pd.read_parquet(*args, **kwargs))

# def DataFrame(*args, **kwargs):
#     return Grizzlies(pd.DataFrame(*args, **kwargs))

# def Series(*args, **kwargs):
#     return pd.Series(*args, **kwargs)  # Keeping Series as a normal pandas object for now

# def merge(*args, **kwargs):
#     return Grizzlies(pd.merge(*args, **kwargs))

# def merge_ordered(*args, **kwargs):
#     return Grizzlies(pd.merge_ordered(*args, **kwargs))

# def concat(*args, **kwargs):
#     return Grizzlies(pd.concat(*args, **kwargs))

# def unique(*args, **kwargs):
#     return pd.unique(*args, **kwargs)

# def isnull(*args, **kwargs):
#     return pd.isnull(*args, **kwargs)

# def notnull(*args, **kwargs):
#     return pd.notnull(*args, **kwargs)

# def isna(*args, **kwargs):
#     return pd.isna(*args, **kwargs)

# def notna(*args, **kwargs):
#     return pd.notna(*args, **kwargs)

# def to_numeric(*args, **kwargs):
#     return pd.to_numeric(*args, **kwargs)

# def to_datetime(*args, **kwargs):
#     return pd.to_datetime(*args, **kwargs)

# def to_timedelta(*args, **kwargs):
#     return pd.to_timedelta(*args, **kwargs)

# def date_range(*args, **kwargs):
#     return pd.date_range(*args, **kwargs)