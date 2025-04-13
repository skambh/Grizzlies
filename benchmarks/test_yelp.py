import pandas as pd
import numpy as np
from grizzlies import Grizzlies
import operator
from benchmarks.utils.utils import benchmark_function, repeat_benchmark, export_benchmark_results

benchmark_results = []

def load_and_prepare_dataset(path: str, row_scaling_factor: int = 1, col_scaling_factor: int = 1):
    df = pd.read_csv(path)

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

    return df, Grizzlies(df, create_scheme="sliding", drop_scheme='lru', treshhold=5, xval=20)


def benchmark_column_access(df_pandas, df_grizzlies):
    def pandas_func():
        for _ in range(10):
            _ = df_pandas[['ID', 'Rating', 'Organization']]

    def grizzlies_func():
        for _ in range(10):
            _ = df_grizzlies['ID']
            __ = df_grizzlies['Rating']
            ___ = df_grizzlies['Organization']

    _, pt, pm = repeat_benchmark(pandas_func, desc="Pandas Column Access")
    _, gt, gm = repeat_benchmark(grizzlies_func, desc="Grizzlies Column Access")
    print(f"[Column Access] Pandas: {pt:.6f}s, {pm:.4f} MiB | Grizzlies: {gt:.6f}s, {gm:.4f} MiB")
    benchmark_results.append(["Column Access", "Pandas", pt, pm])
    benchmark_results.append(["Column Access", "Grizzlies", gt, gm])


def benchmark_row_lookup(df_pandas, df_grizzlies):
    lookup_id = df_pandas['ID'].iloc[50000]

    resp, pt, pm = repeat_benchmark(lambda: df_pandas[df_pandas['Rating'] == 3], desc="Pandas Row Lookup")
    resg, gt, gm = repeat_benchmark(lambda: df_grizzlies.evalfunc('Rating', operator.eq, 3), desc="Grizzlies Row Lookup")
    print(f"[Row Lookup] Pandas: {pt:.6f}s, {pm:.4f} MiB | Grizzlies: {gt:.6f}s, {gm:.4f} MiB")
    # print(resp)
    # print(resg)
    benchmark_results.append(["Row Lookup", "Pandas", pt, pm])
    benchmark_results.append(["Row Lookup", "Grizzlies", gt, gm])


def benchmark_sort(df_pandas, df_grizzlies):
    _, pt, pm = repeat_benchmark(lambda: df_pandas.sort_values('Rating'), desc="Pandas Sort")
    _, gt, gm = repeat_benchmark(lambda: df_grizzlies.sort_values('Rating'), desc="Grizzlies Sort")
    print(f"[Sort] Pandas: {pt:.6f}s, {pm:.4f} MiB | Grizzlies: {gt:.6f}s, {gm:.4f} MiB")
    benchmark_results.append(["Sort", "Pandas", pt, pm])
    benchmark_results.append(["Sort", "Grizzlies", gt, gm])


def benchmark_merge(df_pandas, df_grizzlies):
    df_pandas2 = df_pandas.sample(frac=0.2, random_state=42)
    df_grizzlies2 = df_grizzlies.sample(frac=0.2, random_state=42)

    _, pt, pm = repeat_benchmark(lambda: df_pandas.merge(df_pandas2, on='ID'), desc="Pandas Merge")
    _, gt, gm = repeat_benchmark(lambda: df_grizzlies.merge(df_grizzlies2, on='ID'), desc="Grizzlies Merge")
    print(f"[Merge] Pandas: {pt:.6f}s, {pm:.4f} MiB | Grizzlies: {gt:.6f}s, {gm:.4f} MiB")
    benchmark_results.append(["Merge", "Pandas", pt, pm])
    benchmark_results.append(["Merge", "Grizzlies", gt, gm])


def benchmark_groupby(df_pandas, df_grizzlies):
    _, pt, pm = repeat_benchmark(lambda: df_pandas.groupby('Category')['Rating'].mean(), desc="Pandas GroupBy")
    _, gt, gm = repeat_benchmark(lambda: df_grizzlies.groupby('Category')['Rating'].mean(), desc="Grizzlies GroupBy")
    print(f"[GroupBy] Pandas: {pt:.6f}s, {pm:.4f} MiB | Grizzlies: {gt:.6f}s, {gm:.4f} MiB")
    benchmark_results.append(["GroupBy", "Pandas", pt, pm])
    benchmark_results.append(["GroupBy", "Grizzlies", gt, gm])


def benchmark_filter(df_pandas, df_grizzlies):
    _, pt, pm = repeat_benchmark(lambda: df_pandas[df_pandas['Rating'] > 3], desc="Pandas Filter")
    _, gt, gm = repeat_benchmark(lambda: df_grizzlies[df_grizzlies['Rating'] > 3], desc="Grizzlies Filter")
    print(f"[Filter] Pandas: {pt:.6f}s, {pm:.4f} MiB | Grizzlies: {gt:.6f}s, {gm:.4f} MiB")
    benchmark_results.append(["Filter", "Pandas", pt, pm])
    benchmark_results.append(["Filter", "Grizzlies", gt, gm])


def benchmark_apply(df_pandas, df_grizzlies):
    def simple_function(x):
        return x * 2 if isinstance(x, (int, float)) else x

    _, pt, pm = repeat_benchmark(lambda: df_pandas['Rating'].apply(simple_function), desc="Pandas Apply")
    _, gt, gm = repeat_benchmark(lambda: df_grizzlies['Rating'].apply(simple_function), desc="Grizzlies Apply")
    print(f"[Apply] Pandas: {pt:.6f}s, {pm:.4f} MiB | Grizzlies: {gt:.6f}s, {gm:.4f} MiB")
    benchmark_results.append(["Apply", "Pandas", pt, pm])
    benchmark_results.append(["Apply", "Grizzlies", gt, gm])


def benchmark_drop_column(df_pandas, df_grizzlies):
    _, pt, pm = repeat_benchmark(lambda: df_pandas.drop(columns=['Organization']), desc="Pandas Drop Column")
    _, gt, gm = repeat_benchmark(lambda: df_grizzlies.drop(columns=['Organization']), desc="Grizzlies Drop Column")
    print(f"[Drop Column] Pandas: {pt:.6f}s, {pm:.4f} MiB | Grizzlies: {gt:.6f}s, {gm:.4f} MiB")
    benchmark_results.append(["Drop Column", "Pandas", pt, pm])
    benchmark_results.append(["Drop Column", "Grizzlies", gt, gm])


def benchmark_fillna(df_pandas, df_grizzlies):
    df_pandas_with_na = df_pandas.copy()
    df_grizzlies_with_na = df_grizzlies.copy()
    df_pandas_with_na.loc[::1000, 'Rating'] = None
    df_grizzlies_with_na.loc[::1000, 'Rating'] = None

    _, pt, pm = repeat_benchmark(lambda: df_pandas_with_na.fillna(0), desc="Pandas FillNA")
    _, gt, gm = repeat_benchmark(lambda: df_grizzlies_with_na.fillna(0), desc="Grizzlies FillNA")
    print(f"[FillNA] Pandas: {pt:.6f}s, {pm:.4f} MiB | Grizzlies: {gt:.6f}s, {gm:.4f} MiB")
    benchmark_results.append(["FillNA", "Pandas", pt, pm])
    benchmark_results.append(["FillNA", "Grizzlies", gt, gm])


def benchmark_numpy_vector_math(df_pandas, df_grizzlies):
    _, pt, pm = repeat_benchmark(lambda: np.log1p(df_pandas['Rating'].values), desc="Pandas Vector Math")
    _, gt, gm = repeat_benchmark(lambda: np.log1p(df_grizzlies['Rating'].values), desc="Grizzlies Vector Math")
    print(f"[NumPy Vector Math] Pandas: {pt:.6f}s, {pm:.4f} MiB | Grizzlies: {gt:.6f}s, {gm:.4f} MiB")
    benchmark_results.append(["NumPy Vector Math", "Pandas", pt, pm])
    benchmark_results.append(["NumPy Vector Math", "Grizzlies", gt, gm])


def benchmark_numpy_stats(df_pandas, df_grizzlies):
    _, pt, pm = repeat_benchmark(lambda: np.mean(df_pandas['Rating'].values), desc="Pandas Stats")
    _, gt, gm = repeat_benchmark(lambda: np.mean(df_grizzlies['Rating'].values), desc="Grizzlies Stats")
    print(f"[NumPy Stats] Pandas: {pt:.6f}s, {pm:.4f} MiB | Grizzlies: {gt:.6f}s, {gm:.4f} MiB")
    benchmark_results.append(["NumPy Stats", "Pandas", pt, pm])
    benchmark_results.append(["NumPy Stats", "Grizzlies", gt, gm])


def benchmark_numpy_masking(df_pandas, df_grizzlies):
    _, pt, pm = repeat_benchmark(lambda: df_pandas['Rating'].values[df_pandas['Rating'].values > 3], desc="Pandas Masking")
    _, gt, gm = repeat_benchmark(lambda: df_grizzlies['Rating'].values[df_grizzlies['Rating'].values > 3], desc="Grizzlies Masking")
    print(f"[NumPy Masking] Pandas: {pt:.6f}s, {pm:.4f} MiB | Grizzlies: {gt:.6f}s, {gm:.4f} MiB")
    benchmark_results.append(["NumPy Masking", "Pandas", pt, pm])
    benchmark_results.append(["NumPy Masking", "Grizzlies", gt, gm])


def benchmark_numpy_custom_func(df_pandas, df_grizzlies):
    def custom_sigmoid(arr):
        return 1 / (1 + np.exp(-arr))

    _, pt, pm = repeat_benchmark(lambda: custom_sigmoid(df_pandas['Rating'].values), desc="Pandas Custom Func")
    _, gt, gm = repeat_benchmark(lambda: custom_sigmoid(df_grizzlies['Rating'].values), desc="Grizzlies Custom Func")
    print(f"[NumPy Custom Func] Pandas: {pt:.6f}s, {pm:.4f} MiB | Grizzlies: {gt:.6f}s, {gm:.4f} MiB")
    benchmark_results.append(["Numpy Custom Func", "Pandas", pt, pm])
    benchmark_results.append(["Numpy Custom Func", "Grizzlies", gt, gm])


def benchmark_to_numpy(df_pandas, df_grizzlies):
    _, pt, pm = repeat_benchmark(lambda: df_pandas[['Rating', 'ID']].values, desc="Pandas to NumPy")
    _, gt, gm = repeat_benchmark(lambda: df_grizzlies[['Rating', 'ID']].values, desc="Grizzlies to NumPy")
    print(f"[To NumPy] Pandas: {pt:.6f}s, {pm:.4f} MiB | Grizzlies: {gt:.6f}s, {gm:.4f} MiB")
    benchmark_results.append(["To NumPy", "Pandas", pt, pm])
    benchmark_results.append(["To NumPy", "Grizzlies", gt, gm])

def benchmark_head(df_pandas, df_grizzlies):
    _,pt, pm = repeat_benchmark(lambda: df_pandas.head(), desc="Pandas .head()")
    _,gt, gm = repeat_benchmark(lambda: df_grizzlies.head(), desc="Grizzlies .head()")
    print(f"[Head] Pandas: {pt:.6f}s, {pm:.4f} MiB | Grizzlies: {gt:.6f}s, {gm:.4f} MiB")
    benchmark_results.append(["Head", "Pandas", pt, pm])
    benchmark_results.append(["Head", "Grizzlies", gt, gm])

def benchmark_describe(df_pandas, df_grizzlies):
    _,pt, pm = repeat_benchmark(lambda: df_pandas.describe(), desc="Pandas .describe()")
    _,gt, gm = repeat_benchmark(lambda: df_grizzlies.describe(), desc="Grizzlies .describe()")
    print(f"[Describe] Pandas: {pt:.6f}s, {pm:.4f} MiB | Grizzlies: {gt:.6f}s, {gm:.4f} MiB")
    benchmark_results.append(["Describe", "Pandas", pt, pm])
    benchmark_results.append(["Describe", "Grizzlies", gt, gm])

def benchmark_nunique(df_pandas, df_grizzlies):
    _,pt, pm = repeat_benchmark(lambda: df_pandas.nunique(), desc="Pandas .nunique()")
    _,gt, gm = repeat_benchmark(lambda: df_grizzlies.nunique(), desc="Grizzlies .nunique()")
    print(f"[Nunique] Pandas: {pt:.6f}s, {pm:.4f} MiB | Grizzlies: {gt:.6f}s, {gm:.4f} MiB")
    benchmark_results.append(["Nunique", "Pandas", pt, pm])
    benchmark_results.append(["Nunique", "Grizzlies", gt, gm])

def benchmark_value_counts(df_pandas, df_grizzlies):
    _,pt, pm = repeat_benchmark(lambda: df_pandas['Category'].value_counts(), desc="Pandas .value_counts()")
    _,gt, gm = repeat_benchmark(lambda: df_grizzlies['Category'].value_counts(), desc="Grizzlies .value_counts()")
    print(f"[Value Counts] Pandas: {pt:.6f}s, {pm:.4f} MiB | Grizzlies: {gt:.6f}s, {gm:.4f} MiB")
    benchmark_results.append(["Value Counts", "Pandas", pt, pm])
    benchmark_results.append(["Value Counts", "Grizzlies", gt, gm])



def main():
    dataset_path = "tests/data/yelp_database.csv"
    row_scaling_factor = 5
    col_scaling_factor = 2

    df_pandas, df_grizzlies = load_and_prepare_dataset(dataset_path, row_scaling_factor, col_scaling_factor)
    print("Dataset loaded and scaled. Starting benchmarks...\n")

    # Existing benchmarks
    benchmark_column_access(df_pandas, df_grizzlies)
    benchmark_row_lookup(df_pandas, df_grizzlies)
    benchmark_sort(df_pandas, df_grizzlies)
    benchmark_merge(df_pandas, df_grizzlies)
    benchmark_groupby(df_pandas, df_grizzlies)
    benchmark_filter(df_pandas, df_grizzlies)
    benchmark_apply(df_pandas, df_grizzlies)
    benchmark_drop_column(df_pandas, df_grizzlies)
    benchmark_fillna(df_pandas, df_grizzlies)

    benchmark_numpy_vector_math(df_pandas, df_grizzlies)
    benchmark_numpy_stats(df_pandas, df_grizzlies)
    benchmark_numpy_masking(df_pandas, df_grizzlies)
    benchmark_numpy_custom_func(df_pandas, df_grizzlies)
    benchmark_to_numpy(df_pandas, df_grizzlies)

    benchmark_head(df_pandas, df_grizzlies)
    benchmark_describe(df_pandas, df_grizzlies)
    benchmark_nunique(df_pandas, df_grizzlies)
    benchmark_value_counts(df_pandas, df_grizzlies)
    # Save benchmark results to CSV 
    export_benchmark_results(benchmark_results, filename="benchmarks/results/yelp_benchmark_results.csv")

    # Optional save
    # df_grizzlies.save()


if __name__ == "__main__":
    main()