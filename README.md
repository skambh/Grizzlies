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

### TPC-H

#### Downloading the Dataset

To run the TPC-H benchmarks you will first need to download the TPC-H dataset from this [Kaggle repo](https://www.kaggle.com/datasets/davidalexander01/tpc-h-dataset/data)

Download all 8 of the .tbl files (can download as a zip), and put the files into some folder.

#### Run the benchmark

You can then run the TPC-H benchmarks by running the following from the project root

```
python tests/tpc_h_benchmark.py --data_set **path_to_folder_with_data**
```

When running the benchmark, you can configure the script to run using Grizzlies or Pandas, by configuring the test_mode variable on line 23

### Running Tests

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
