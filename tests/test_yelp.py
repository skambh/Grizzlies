import pandas as pd
import numpy as np
import time
# from memory_profiler import memory_usage

# Assuming Grizzles is your custom implementation
from src.grizzlies.Grizzlies import Grizzlies

def benchmark_function(func, *args, **kwargs):
    start_time = time.perf_counter()
    # mem_usage = memory_usage((func, args, kwargs), max_usage=True)
    result = func(*args, **kwargs)
    end_time = time.perf_counter()
    return result, end_time - start_time

if __name__ == "__main__":
    
    # Load Yelp dataset
    dataset_path = "tests/data/yelp_database.csv"  # Update with actual path
    df_pandas = pd.read_csv(dataset_path)
    df_grizzles = Grizzlies(df_pandas.to_dict(orient='list'))  # Convert to Grizzles format
    print(df_grizzles)

    # Benchmark Multiple Column Access Performance (without explicit indexing)
    def access_multiple_columns_pandas():
        for _ in range(6):  # Simulating repeated access
            _ = df_pandas[['ID', 'Rating', 'Organization']]

    def access_multiple_columns_grizzles():
        for _ in range(6):  # Simulating repeated access
            _ = df_grizzles['ID']
            __= df_grizzles['Rating']
            ___= df_grizzles['Organization']

    _, pandas_time = benchmark_function(access_multiple_columns_pandas)
    _, grizzles_time = benchmark_function(access_multiple_columns_grizzles)
    print(f"Multiple Column Access: Pandas: {pandas_time:.4f}s, Grizzles: {grizzles_time:.4f}s")

    # Benchmark Row Lookup (without explicit indexing)
    lookup_id = df_pandas['ID'].iloc[50000]
    _, pandas_time= benchmark_function(lambda: df_pandas['ID'])
    _, grizzles_time = benchmark_function(lambda: df_grizzles['ID'])
    print(f"Row Lookup: Pandas: {pandas_time:.6f}s, Grizzles: {grizzles_time:.6f}s")

    # Benchmark Sorting Performance
    _, pandas_time= benchmark_function(lambda: df_pandas.sort_values('Rating'))
    _, grizzles_time = benchmark_function(lambda: df_grizzles.sort_values('Rating'))
    print(f"Sorting: Pandas: {pandas_time:.4f}s| Grizzles: {grizzles_time:.4f}s")

    # Benchmark Merge Performance
    df_pandas2 = df_pandas.sample(frac=0.5, random_state=42)
    df_grizzles2 = df_grizzles.sample(frac=0.5, random_state=42)
    _, pandas_time = benchmark_function(lambda: df_pandas.merge(df_pandas2, on='ID'))
    _, grizzles_time= benchmark_function(lambda: df_grizzles.merge(df_grizzles2, on='ID'))
    print(f"Merge: Pandas: {pandas_time:.4f}s, Grizzles: {grizzles_time:.4f}s")
