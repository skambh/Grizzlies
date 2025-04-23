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

def benchmark_row_lookup(df_pandas, df_grizzlies, lookup_column, lookup_value, n=10):

    resp, pt, pm = repeat_benchmark(lambda: df_pandas[df_pandas[lookup_column] == lookup_value], desc="Pandas Row Lookup", n=n)
    resg, gt, gm = repeat_benchmark(lambda: df_grizzlies.evalfunc(lookup_column, operator.eq, lookup_value), desc="Grizzlies Row Lookup", n=n)
    print(f"[Row Lookup] Pandas: {pt:.6f}s, {pm:.4f} MiB | Grizzlies: {gt:.6f}s, {gm:.4f} MiB")
    assert resp.equals(resg), "Row lookup results do not match!"

    return pt, pm, gt, gm

def main():
    dataset_path = "tests/data/yelp_database.csv"
    row_scaling_factor = 8
    col_scaling_factor = 1
    
    df_pandas= load_and_prepare_dataset(dataset_path, ["OLF"])
    df_pandas = scale_dataframe(df_pandas, row_scaling_factor, col_scaling_factor)
    df_temp=df_pandas.__deepcopy__()
    
    pandas_size=sys.getsizeof(df_pandas)/1e+6
    create_scheme = ["sliding", "basic"]
    drop_scheme = ['lru', 'min', 'treshhold']
    index_type = ['ordered', 'hash']

    results_df = pd.DataFrame(columns=["create_scheme", "drop_scheme", "index_type", "pandas_time", "pandas_memory", "grizzlies_time", "grizzlies_memory"])
    
    print("Dataset loaded and scaled. Starting benchmarks...\n")
    for cs in create_scheme:
        for ds in drop_scheme:
            for it in index_type:
                print(f"Creating Grizzlies with {cs=}, {ds=}, {it=}")
                if cs == "basic" and ds != "lru":
                    print("Skipping this combination due to unsupported drop scheme.")
                    continue
                df_grizzlies = Grizzlies(df_temp, create_scheme=cs, drop_scheme=ds, index_type=it)
                # Simulate random accesses to the DataFrame for index creation
                simulate_accesses(df_grizzlies, ["Organization", "Rating", "ID"])
                benchmark_row_lookup(df_pandas, df_grizzlies, "Organization", "Zaxby's Chicken Fingers & Buffalo Wings", n=1)
                # Benchmark row lookup
                pt, pm, gt, gm = benchmark_row_lookup(df_pandas, df_grizzlies, "Organization", "Zaxby's Chicken Fingers & Buffalo Wings")
                print(f"[Row Lookup] Pandas: {pt:.6f}s, {pm:.4f} MiB | Grizzlies: {gt:.6f}s, {gm:.4f} MiB")
                # append results to DataFrame
                results_df = pd.concat([results_df, pd.DataFrame([[cs, ds, it, pt, pm, gt, gm]], columns=results_df.columns)], ignore_index=True)
            results_df.to_csv("benchmarks/results/benchmark_parameters.csv", index=False)

    


if __name__ == "__main__":
    main()
     


