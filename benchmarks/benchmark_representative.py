import pandas as pd
import numpy as np
from grizzlies import Grizzlies, read_csv
import sys
import operator
from benchmarks.utils.utils import benchmark_function, repeat_benchmark, export_benchmark_results
import random


def load_and_prepare_dataset(path: str, columns_to_drop:list =[]):
    df = pd.read_csv(path)
    if len(columns_to_drop) > 0:
        df.drop(columns=columns_to_drop, inplace=True)
    df.dropna(inplace=True)
    df["User_liking"] = [round(random.uniform(1.0, 10.0), 2) for _ in range(len(df))]  # Add a new column with random float values
    df["Avg_Visits"] = [round(random.uniform(1.0, 2000.0), 2) for _ in range(len(df))]  # Add a new column with random float values
    # print(len(df))
    # print(df.head())
    # df_grizzlies = read_csv(path)

    return df

def scale_dataframe(df: pd.DataFrame, row_scaling_factor: int, col_scaling_factor: int) -> pd.DataFrame:
     # Row scaling: duplicate rows
    if row_scaling_factor > 1:
        df = pd.concat([df] * row_scaling_factor, ignore_index=True)

    # Column scaling: duplicate columns with renamed headers
    if col_scaling_factor > 1:
        original_columns = df.columns
        for i in range(1, col_scaling_factor):
            new_cols = df[original_columns].copy()
            new_cols.columns = [f"{col}_copy{i}" for col in original_columns]
            df = pd.concat([df, new_cols], axis=1)
    
    return df

def simulate_accesses(df_grizzlies, index_columns):
    # Simulate random accesses to the DataFrame for index creation
    for _ in range(20):
            _ = df_grizzlies[index_columns]

def benchmark_row_lookup(df_pandas, df_grizzlies, lookup_column, lookup_value, n=10):

    resp, pt, pm = repeat_benchmark(lambda: df_pandas[df_pandas[lookup_column] == lookup_value], desc="Pandas Row Lookup", n=n)
    resg, gt, gm = repeat_benchmark(lambda: df_grizzlies.evalfunc(lookup_column, operator.eq, lookup_value), desc="Grizzlies Row Lookup", n=n)
    print(f"[Row Lookup] Pandas: {pt:.6f}s, {pm:.4f} MiB | Grizzlies: {gt:.6f}s, {gm:.4f} MiB")
    # print(resp)
    # print(resg)
    assert resp.equals(resg), "Row lookup results do not match!"

    return pt, pm, gt, gm

def main():
    dataset_path = "tests/data/yelp_database.csv"
    row_scaling_factor = 8
    col_scaling_factor = 1
    
    df_pandas= load_and_prepare_dataset(dataset_path, ["OLF"])
    df_pandas = scale_dataframe(df_pandas, row_scaling_factor, col_scaling_factor)
    df_temp=df_pandas.__deepcopy__()
    df_grizzlies = Grizzlies(df_temp, create_scheme="sliding", drop_scheme='lru',index_type='ordered')
    
    pandas_size=sys.getsizeof(df_pandas)/1e+6
    # Sample values pulled from df_pandas
    float_user_liking = df_pandas["User_liking"].sample(1).values[0]
    float_avg_visits = df_pandas["Avg_Visits"].sample(1).values[0]
    

    test_cases = [
        {
            "test_case": "String, less frequent",
            "column": "Organization",
            "value": "Zaxby's Chicken Fingers & Buffalo Wings",
            "index": ["Organization", "Rating", "ID"]
        },
        {
            "test_case": "String, Categorical",
            "column": "State",
            "value": "AL",
            "index": ["State", "Rating", "ID"]
        },
        {
            "test_case": "Integer, (ID)",
            "column": "ID",
            "value": 30428,
            "index": ["Rating", "Organization", "ID"]
        },
        {
            "test_case": "Integer, categorical",
            "column": "Rating",
            "value": 4,
            "index": ["Rating", "Organization", "ID"]
        },
        {
            "test_case": "Float, short range",
            "column": "User_liking",
            "value": float_user_liking,
            "index": ["User_liking", "Organization", "State"]
        },
        {
            "test_case": "Float, long range",
            "column": "Avg_Visits",
            "value": float_avg_visits,
            "index": ["Avg_Visits", "Organization", "State"]
        }
    ]

    # Convert to DataFrame
    test_cases_df = pd.DataFrame(test_cases)

    results_df = test_cases_df.copy()
    results_columns = ["pandas_time", "pandas_memory", "grizzlies_time", "grizzlies_memory"]
    results_df = pd.concat([results_df, pd.DataFrame(columns=results_columns)], axis=1)


    for index, row in test_cases_df.iterrows():
        print(f"Running test case: {row['test_case']}")
        # Simulate random accesses to the DataFrame for index creation
        simulate_accesses(df_grizzlies, row["index"])
        # Benchmark row lookup
        pt, pm, gt, gm = benchmark_row_lookup(df_pandas, df_grizzlies, row["column"], row["value"])
        print(f"[Row Lookup] Pandas: {pt:.6f}s, {pm:.4f} MiB | Grizzlies: {gt:.6f}s, {gm:.4f} MiB")
        # append results to DataFrame
        results_df.at[index, "pandas_time"] = pt
        results_df.at[index, "pandas_memory"] = pm
        results_df.at[index, "grizzlies_time"] = gt
        results_df.at[index, "grizzlies_memory"] = gm
        results_df.to_csv("benchmarks/results/benchmark_representative.csv", index=False)
        
    


if __name__ == "__main__":
    main()
     


