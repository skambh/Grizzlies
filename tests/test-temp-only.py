from src.grizzlies.Grizzlies import Grizzlies, read_csv
import time

def test_my_function():
    data = {'ID': [1, 2, 3, 4], 'Value': [10, 20, 30, 40], 'Category': ['A', 'B', 'C', 'D']}
    df = Grizzlies(data)
    for i in range(6):
        start_time = time.time()
        df['ID']
        end_time = time.time()
        print(f"time taken = {end_time - start_time}")
    df.save()
    df = Grizzlies(data)

def read_in_csv():
    data_csv = "tests/data/data.csv"
    df = read_csv(data_csv)
    print(type(df))
    print(df['Date'])

    
if __name__ == "__main__":
    # test_my_function()
    read_in_csv()