import pytest
from src.grizzlies.Grizzlies import Grizzlies
from pandas import DataFrame

@pytest.fixture()
def sample_grizzlies_df():
    """Create a sample DataFrame for testing."""
    data = {
        'A': [1, 2, 3, 4, 5],
        'B': ['a', 'b', 'c', 'd', 'e'],
        'C': [1.1, 2.2, 3.3, 4.4, 5.5]
    }
    return Grizzlies(data)

@pytest.fixture()
def sample_pandas_df():
    """Create a sample DataFrame for testing."""
    data = {
        'A': [1, 2, 3, 4, 5],
        'B': ['a', 'b', 'c', 'd', 'e'],
        'C': [1.1, 2.2, 3.3, 4.4, 5.5]
    }
    return DataFrame(data)