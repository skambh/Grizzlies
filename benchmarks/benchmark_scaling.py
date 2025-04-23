import pandas as pd
import numpy as np
from grizzlies import Grizzlies, read_csv
import sys
import operator
from benchmarks.utils.utils import benchmark_function, repeat_benchmark, export_benchmark_results

def load_and_prepare_dataset(path: str, columns_to_drop:list =[]):
    df = pd.read_csv(path)
    if len(columns_to_drop) > 0:
        df.drop(columns=columns_to_drop, inplace=True)
    df.dropna(inplace=True)
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

def benchmark_row_lookup(df_pandas, df_grizzlies, lookup_column, lookup_value):

    resp, pt, pm = repeat_benchmark(lambda: df_pandas[df_pandas[lookup_column] == lookup_value], desc="Pandas Row Lookup")
    resg, gt, gm = repeat_benchmark(lambda: df_grizzlies.evalfunc(lookup_column, operator.eq, lookup_value), desc="Grizzlies Row Lookup")
    print(f"[Row Lookup] Pandas: {pt:.6f}s, {pm:.4f} MiB | Grizzlies: {gt:.6f}s, {gm:.4f} MiB")
    # print(resp)
    # print(resg)
    assert resp.equals(resg), "Row lookup results do not match!"

    return pt, pm, gt, gm

def main():
    dataset_path = "tests/data/yelp_database.csv"
    row_scaling_factor = 2
    col_scaling_factor = 1
    
    df_pandas= load_and_prepare_dataset(dataset_path, ["OLF"])
    
    pandas_size=sys.getsizeof(df_pandas)/1e+6
    print(f"{pandas_size=}")
    row_scale_list = [2, 4, 8, 16]
    col_scale_list = [1, 2, 3, 4]
    results_df = pd.DataFrame(columns=["df_size", "row_scale", "col_scale", "pandas_time", "pandas_memory", "grizzlies_time", "grizzlies_memory"])
    
    print("Dataset loaded and scaled. Starting benchmarks...\n")
    for row_scale in row_scale_list:
        for col_scale in col_scale_list:
            print(f"Scaling: {row_scale=}, {col_scale=}")
            if row_scale == 16 and col_scale > 2:
                print("Skipping this combination due to memory constraints.")
                continue
            df_pandas= load_and_prepare_dataset(dataset_path, ["OLF"])
            df_pandas = scale_dataframe(df_pandas, row_scale, col_scale)
            df_size=sys.getsizeof(df_pandas)/1e+6
            print(f"{df_size=}")
            df_temp=df_pandas.__deepcopy__()
            df_grizzlies = Grizzlies(df_temp, create_scheme="sliding", drop_scheme='lru', treshhold=7, xval=10, index_type='ordered')

            # Simulate random accesses to the DataFrame for index creation
            simulate_accesses(df_grizzlies, ["Organization", "Rating", "ID"])

            # Benchmark row lookup
            pt, pm, gt, gm = benchmark_row_lookup(df_pandas, df_grizzlies, "Organization", "Zaxby's Chicken Fingers & Buffalo Wings")
            print(f"[Row Lookup] Pandas: {pt:.6f}s, {pm:.4f} MiB | Grizzlies: {gt:.6f}s, {gm:.4f} MiB")
            # append results to DataFrame
            results_df = pd.concat([results_df, pd.DataFrame([[df_size, row_scale, col_scale, pt, pm, gt, gm]], columns=results_df.columns)], ignore_index=True)
        results_df.to_csv("benchmarks/results/benchmark_results.csv", index=False)
            # print(f"Memory usage: Pandas: {pm:.4f} MiB | Grizzlies: {gm:.4f} MiB")


if __name__ == "__main__":
    main()
     


