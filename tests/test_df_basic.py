import pytest
import pandas as pd
import numpy as np
from src.grizzlies.Grizzlies import Grizzlies
import pandas.testing as pdt

def test_basic_properties(sample_grizzlies_df):
    """Test basic DataFrame properties."""
    assert sample_grizzlies_df.shape == (5, 3)
    assert list(sample_grizzlies_df.columns) == ['A', 'B', 'C']
    assert sample_grizzlies_df.dtypes['A'] == np.int64
    assert sample_grizzlies_df.dtypes['B'] == object
    assert sample_grizzlies_df.dtypes['C'] == np.float64

def test_attribute_access(sample_grizzlies_df):
    """Test attribute access delegation to pandas."""
    assert sample_grizzlies_df.shape == (5, 3)
    assert len(sample_grizzlies_df.columns) == 3
    assert sample_grizzlies_df.size == 15
    
    # Test that pandas methods work
    mean_values = sample_grizzlies_df.mean(numeric_only=True)
    assert mean_values['A'] == 3
    assert mean_values['C'] == 3.3

def test_column_iteration(sample_grizzlies_df):
    """Test iteration over columns."""
    columns = [col for col in sample_grizzlies_df]
    assert columns == ['A', 'B', 'C']

def test_column_access(sample_grizzlies_df):
    """Test column access"""
    # print(sample_grizzlies_df['A'].tolist())
    expected_df = Grizzlies(pd.Series([1, 2, 3, 4, 5], name='A'))
    pdt.assert_series_equal(sample_grizzlies_df['A'], expected_df['A'])


# NYI
@pytest.mark.skip()
def test_groupby():
    assert 0 == -1
