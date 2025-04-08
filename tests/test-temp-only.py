from src.grizzlies.Grizzlies import Grizzlies
import time

def test_my_function():
    data = {'ID': [1, 2, 3, 4], 'Value': [10, 20, 30, 40], 'Category': ['A', 'B', 'C', 'D'], 'lolol':[32, 44, 22, 33]}
    df = Grizzlies(data, create_scheme="sliding")
    for i in range(7):
        start_time = time.time()
        df['ID']
        end_time = time.time()
        print(f"time taken = {end_time - start_time}")
    df['lolol']
    df.save()
    print(df.get_stats())
    df = Grizzlies(data)


def test_sliding_print_shi():
    data = {'ID': [1, 2, 3, 4], 'Value': [10, 20, 30, 40], 'Category': ['A', 'B', 'C', 'D'], 'lolol':[32, 44, 22, 33]}
    df = Grizzlies(data, create_scheme="sliding")
    print(f"window size: {df._window_size}")
    print(f"every xth: {df._everyxth}")
    print(f"x val: {df._xval}")
    print(f"increment access count: {df._increment_access_count}")
    print(f"sliding window: {df._sliding_window}")
    print(f"lru counter: {df._lru_ctr}")
    print(f"lru: {df._lru}")
    print(f"drop index: {df._drop_index}")
    df.save()

def test_sliding_every5():
    data = {'ID': [1, 2, 3, 4], 'Value': [10, 20, 30, 40], 'Category': ['A', 'B', 'C', 'D'], 'lolol':[32, 44, 22, 33], 'hhe':['d','w','ee','w']}
    df = Grizzlies(data, create_scheme="sliding", drop_scheme='min')
    print(df._max_indices)
    for i in range(5):
        df["Category"]
    for i in range(5):
        df["lolol"]
    for i in range(5):
        df['ID']
    for i in range(5):
        df['Value']
        # print(df._sliding_window)


def test_basic_lru():
    data = {'ID': [1, 2, 3, 4], 'Value': [10, 20, 30, 40], 'Category': ['A', 'B', 'C', 'D'], 'lolol':[32, 44, 22, 33], 'hhe':['d','w','ee','w']}
    df = Grizzlies(data, create_scheme="basic", drop_scheme='lru')
    for i in range(5):
        df["Category"]
    for i in range(5):
        df["lolol"]
    for i in range(5):
        df['ID']
    print(df._hash_indices)
    for i in range(5):
        df['Value']
    print(df._hash_indices)



    
if __name__ == "__main__":
    # test_my_function()
    # test_sliding_print_shi()
    # test_sliding_every5()
    test_basic_lru()