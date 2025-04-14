from grizzlies import Grizzlies
from pandas import DataFrame

# Basic shi

# Sliding window shi

def test_sliding_window_initialization(simple_df):
    g = Grizzlies(simple_df, create_scheme="sliding", windowsize=10, threshold=3)
    print(g['A'])
    print(g.loc[:, 'A'])
    # assert len(g._sliding_window) == 0
    # assert g._window_size == 10
    # assert g._threshold == 3
    # assert g._everyxth == 0
    # assert g._lru_ctr == 0
    # assert g._lru == {}
    # assert g._hash_indices == {}

def test_sliding_window_index_creation(simple_df):
    # Test indices get created when access counts big than threshold
    g = Grizzlies(simple_df, create_scheme="sliding", windowsize=10, threshold=3, xval=1)
    for i in range(2):
      i = g['A']
    assert not g._hash_indices

    i = g['A']
    assert 'A' in g._hash_indices
    assert g._hash_indices['A'] == {1: 0, 2: 1, 3: 2, 4: 3, 5: 4} # FIXME: should probably make this reference simple_df instead of hardcoding
    
def test_sliding_window_lru_tracking(simple_df):
    """Test that LRU is properly tracked."""
    g = Grizzlies(simple_df, create_scheme="sliding", windowsize=10, threshold=3, xval=1)
    
    _ = g['A']
    _ = g['B']
    _ = g['C']
    _ = g['A']
    
    # Check LRU values
    assert g._lru['A'] == 3
    assert g._lru['B'] == 1
    assert g._lru['C'] == 2
    assert g._lru_ctr == 4
    
def test_sliding_window_threshold_drop(simple_df):
    """Test dropping indices with threshold scheme."""
    g = Grizzlies(simple_df, create_scheme="sliding", windowsize=6, 
                threshold=3, xval=1, drop_scheme="threshold")
    # Create some indices by accessing columns
    for _ in range(3):
        _ = g['A']
    for _ in range(3):
        _ = g['B']
    
    assert 'A' in g._hash_indices
    assert 'B' in g._hash_indices
    
    # Shift window to make A fall below threshold
    for _ in range(8):
        _ = g['C']
    
    # Access B again to trigger check with counts that place A below threshold
    _ = g['B']
    # for _ in range(4):
    #   _ = g['D'] 
    print(g._hash_indices)
    # # A should be dropped
    # assert 'A' not in g._hash_indices
    # assert 'B' in g._hash_indices
    
#     def test_sliding_window_lru_drop(self, df, cleanup_stats_dir):
#         """Test dropping indices with LRU scheme."""
#         g = Grizzlies(df, create_scheme="sliding", windowsize=10, 
#                     threshold=3, xval=1, drop_scheme="lru")
        
#         # Set max_indices to 1 to force drops
#         g._max_indices = 1
        
#         # Create index for A
#         for _ in range(3):
#             _ = g['A']
        
#         assert 'A' in g._hash_indices
        
#         # Now create index for B, which should drop A since it's least recently used
#         for _ in range(3):
#             _ = g['B']
        
#         assert 'A' not in g._hash_indices
#         assert 'B' in g._hash_indices
    
#     def test_sliding_window_min_drop(self, df, cleanup_stats_dir):
#         """Test dropping indices with min count scheme."""
#         g = Grizzlies(df, create_scheme="sliding", windowsize=10, 
#                     threshold=3, xval=1, drop_scheme="min")
        
#         # Set max_indices to 2 to force drops
#         g._max_indices = 2
        
#         # Create indices for A and B
#         for _ in range(5):
#             _ = g['A']
#         for _ in range(3):
#             _ = g['B']
        
#         assert 'A' in g._hash_indices
#         assert 'B' in g._hash_indices
        
#         # Now create index for C, which should drop B since it has lower count
#         for _ in range(4):
#             _ = g['C']
        
#         assert 'A' in g._hash_indices
#         assert 'B' not in g._hash_indices
#         assert 'C' in g._hash_indices
    
#     def test_sliding_window_persistence(self, df, cleanup_stats_dir):
#         """Test that sliding window state is saved and loaded correctly."""
#         g = Grizzlies(df, create_scheme="sliding", windowsize=5, threshold=3, xval=1)
        
#         # Access columns
#         _ = g['A']
#         _ = g['B']
#         _ = g['A']
        
#         # Save state
#         g.save()
        
#         # Create new instance with same data
#         g2 = Grizzlies(df, create_scheme="sliding", windowsize=5, threshold=3, xval=1)
        
#         # Check if state was loaded
#         assert list(g2._sliding_window) == ['A', 'B', 'A']
        
#         # Continue using the loaded instance
#         _ = g2['A']
        
#         # This should create an index for A
#         assert 'A' in g2._hash_indices
    
#     def test_sliding_window_multiple_column_access(self, df, cleanup_stats_dir):
#         """Test accessing multiple columns at once."""
#         g = Grizzlies(df, create_scheme="sliding", windowsize=10, threshold=3, xval=1)
        
#         # Access multiple columns
#         _ = g[['A', 'B']]
#         _ = g[['A', 'C']]
#         _ = g[['A', 'B']]
        
#         # Check window contents - should have each column added separately
#         assert list(g._sliding_window) == ['A', 'B', 'A', 'C', 'A', 'B']
        
#         # A should have index as it was accessed 3 times
#         assert 'A' in g._hash_indices
#         assert 'B' not in g._hash_indices
#         assert 'C' not in g._hash_indices
    
#     def test_sliding_window_with_boolean_indexing(self, df, cleanup_stats_dir):
#         """Test that boolean indexing correctly updates access counts."""
#         g = Grizzlies(df, create_scheme="sliding", windowsize=10, threshold=3, xval=1)
        
#         # Use boolean indexing with column A
#         _ = g[g['A'] > 2]
#         _ = g[g['A'] < 5]
#         _ = g[g['A'] == 3]
        
#         # Check that A was tracked in the window
#         assert list(g._sliding_window) == ['A', 'A', 'A']
        
#         # A should have index as it was accessed 3 times
#         assert 'A' in g._hash_indices
    
#     def test_max_indices_limit(self, df, cleanup_stats_dir):
#         """Test that the number of indices doesn't exceed max_indices."""
#         g = Grizzlies(df, create_scheme="sliding", windowsize=6, 
#                     threshold=2, xval=1, drop_scheme="threshold")
        
#         # Set max_indices explicitly to test limit
#         g._max_indices = 2
        
#         # Access columns to create indices
#         for _ in range(2):
#             _ = g['A']
#         for _ in range(2):
#             _ = g['B']
#         for _ in range(2):
#             _ = g['C']
        
#         # Should only have 2 indices
#         assert len(g._hash_indices) == 2
        
#         # The indices should be for B and C, as A was least recently used
#         assert 'B' in g._hash_indices
#         assert 'C' in g._hash_indices
#         assert 'A' not in g._hash_indices
    
#     def test_window_size_impact(self, df, cleanup_stats_dir):
#         """Test that window size properly limits the sliding window."""
#         g = Grizzlies(df, create_scheme="sliding", windowsize=3, 
#                     threshold=2, xval=1)
        
#         # Access columns
#         _ = g['A']
#         _ = g['B']
#         _ = g['C']
#         _ = g['D']
        
#         # Window should only have the 3 most recent accesses
#         assert len(g._sliding_window) == 3
#         assert list(g._sliding_window) == ['B', 'C', 'D']
        
#         # Access A and B multiple times to create indices
#         _ = g['A']
#         _ = g['A']
#         _ = g['B']
#         _ = g['B']
        
#         # Only B should have an index since A's earlier access was pushed out of the window
#         assert 'B' in g._hash_indices
#         assert 'A' not in g._hash_indices

# def test_sliding_every5():
#     data = {'ID': [1, 2, 3, 4], 'Value': [10, 20, 30, 40], 'Category': ['A', 'B', 'C', 'D'], 'lolol':[32, 44, 22, 33], 'hhe':['d','w','ee','w']}
#     df = Grizzlies(data, scheme="sliding", drop='min')
#     print("scheme: " + df._scheme)
#     print(df._max_indices)
#     for i in range(5):
#         df["Category"]
#     for i in range(5):
#         df["lolol"]
#     for i in range(5):
#         df['ID']
#     for i in range(5):
#         df['Value']