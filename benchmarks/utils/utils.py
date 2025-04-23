
import pandas as pd
import numpy as np
import time
from memory_profiler import profile, memory_usage
from tqdm import tqdm
import csv

# @profile
def benchmark_function(func, *args, **kwargs):
    
    
    start_time = time.perf_counter()
    result = func(*args, **kwargs)
    end_time = time.perf_counter()
    mem_usage = memory_usage((func, args, kwargs), max_usage=True)
    return result, end_time - start_time, mem_usage

def repeat_benchmark(func, n=10, desc="benchmarking"):
    total_time = 0.0
    total_memory = 0.0
    for _ in tqdm(range(n), desc=desc):
        res, time_taken, mem_usage = benchmark_function(func)
        total_time += time_taken
        total_memory += mem_usage
    avg_time = total_time / n
    avg_memory = total_memory / n
    return res, avg_time, avg_memory

def export_benchmark_results(benchmark_results, filename="benchmark_results.csv"):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Operation", "Library", "Avg Time (s)", "Avg Memory (MiB)"])
        for row in benchmark_results:
            writer.writerow(row)
    print(f"\nResults exported to {filename}") 