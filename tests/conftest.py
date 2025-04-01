import pytest
import grizzlies as gr
from pandas import DataFrame


@pytest.fixture(name='df')
def sample_csv_grizzlies_df():
    """Create a sample DataFrame for testing."""
    data_csv = "tests/data/data.csv"
    data_json = "/test/data/data.json"

    df = gr.read_csv(data_csv)
    return df

@pytest.fixture()
def sample_grizzlies_df():
    """Create a sample DataFrame for testing."""
    data = {
        'A': [1, 2, 3, 4, 5],
        'B': ['a', 'b', 'c', 'd', 'e'],
        'C': [1.1, 2.2, 3.3, 4.4, 5.5]
    }
    return gr.Grizzlies(data)

@pytest.fixture()
def sample_pandas_df():
    """Create a sample DataFrame for testing."""
    data = {
        'A': [1, 2, 3, 4, 5],
        'B': ['a', 'b', 'c', 'd', 'e'],
        'C': [1.1, 2.2, 3.3, 4.4, 5.5]
    }
    return DataFrame(data)