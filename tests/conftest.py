import pytest
import grizzlies as gr
from pandas import DataFrame
from pathlib import Path

@pytest.fixture(name='df')
def sample_csv_grizzlies_df():
    """Sample Grizzlies DataFrame read from csv."""
    BASE_DIR = Path(__file__).resolve().parent
    data_csv = BASE_DIR / "data" / "data.csv"
    data_json = BASE_DIR / "data" / "data.json"

    df = gr.read_csv(data_csv)
    return df

@pytest.fixture()
def simple_df():
    """Much simpler DataFrame for testing."""
    data = {
        'A': [1, 2, 3, 4, 5],
        'B': ['a', 'b', 'c', 'd', 'e'],
        'C': [1.1, 2.2, 3.3, 4.4, 5.5],
    }
    return data

@pytest.fixture()
def sample_pandas_df():
    """Create a sample DataFrame for testing."""
    data = {
        'A': [1, 2, 3, 4, 5],
        'B': ['a', 'b', 'c', 'd', 'e'],
        'C': [1.1, 2.2, 3.3, 4.4, 5.5]
    }
    return DataFrame(data)