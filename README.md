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
