# Grizzlies 🐻 - CSE 584 Final Project (Winter 2025)

![Tests](https://github.com/skambh/Grizzlies/actions/workflows/run_tests.yml/badge.svg)

## Team Members

- Kiran Bodipati
- Shruti Jain
- Shashank Kambhammettu
- Jai Narayanan

### Setup Instructions

1. Clone the repository:

   ```bash
   git clone https://github.com/skambh/Grizzlies
   cd Grizzlies
   ```

2. Setup and activate venv:

   ```bash
   python -m venv venv

   source venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Install grizzlies as package:

   ```bash
   pip install -e .
   ```

5. Benchmarking

#### Run Benchmarking

Remember to run using the memory profiler command:

```
python -m memory_profiler benchmarks/{filename}
```

For example

```bash
python -m memory_profiler benchmarks/test_yelp.py

```

To reproduce the results shown in the paper, make sure to download the Yelp Dataset from [Kaggle](https://www.kaggle.com/datasets/abdulmajid115/yelp-dataset-contains-1-million-rows) and place it in ```benchmarks/data``` folder. 

Run the following scripts -
1. Scalability Results
``` bash
python -m memory_profiler benchmarks/benchmark_scaling.py
```

2. Configuration Tests
``` bash
python -m memory_profiler benchmarks/benchmark_hyperparameters.py
```

3. Data Type Tests
``` bash
python -m memory_profiler benchmarks/benchmark_representative.py
```
The results are saved to ```benchmarks/results``` folder. 
Please feel free to modify the filename and pass the columns and search values to parameters to test your own files. 

The benchmark datasets used for our analyses include:
1. [Yelp Dataset](https://www.kaggle.com/datasets/abdulmajid115/yelp-dataset-contains-1-million-rows)
2. [Yahoo Finance Dataset](https://www.kaggle.com/datasets/eli2022/yahoo-finance-apple-inc-gspc) 
3. [Top Movies](https://www.kaggle.com/datasets/omkarborikar/top-10000-popular-movies)
4. [Movie Reviews](https://www.kaggle.com/datasets/parthdande/imdb-dataset-2024-updated?select=IMDb_Dataset_3.csv)
5. [Pokemon](https://www.kaggle.com/datasets/rzgiza/pokdex-for-all-1025-pokemon-w-text-description)
### TPC-H

#### Downloading the Dataset

To run the TPC-H benchmarks you will first need to download the TPC-H dataset from this [Kaggle Link](https://www.kaggle.com/datasets/davidalexander01/tpc-h-dataset/data)

Download all 8 of the .tbl files (can download as a zip), and put the files into some folder.

#### Run the benchmark

You can then run the TPC-H benchmarks by running the following from the project root

```
python tests/tpc_h_benchmark.py --data_set **path_to_folder_with_data**
```

When running the benchmark, you can configure the script to run using Grizzlies or Pandas, by configuring the test_mode variable on line 23.

The queries in the tpc-h benchmarks don't make enough queries to meet the threshold Grizzlies requires to auto create an index. To account for this, you can run the benchmark 5-6 times, which will cause Grizzlies to build persistent statistics across runs. After doing this for a few times, Grizzlies will then properly create the index on a later run.

### Running Tests
Pytests were used to unit test functions written, and ensure correctness against pandas.
#### Run all tests

```bash
pytest
```

#### Run specific test file

```bash
pytest (file_name)
# ex - pytest tests/test_basic.py
```

#### Run specific test in test file

```bash
pytest (file_name)::(test_name)
# ex - pytest tests/test_basic.py::test_isna
```

#### Run tests by keyword

```bash
pytest -k (keyword)

# ex - pytest -k "na"
```
