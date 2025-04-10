import duckdb

# Connect to DuckDB and load the TPC-H extension
con = duckdb.connect()
con.execute("INSTALL tpch; LOAD tpch;")

# # Set the Scale Factor to 1 (SF1)
# scale_factor = 1
# con.execute(f"SET tpch_scale_factor={scale_factor};")

# Define the tables to extract
tables = ['customer', 'lineitem', 'nation', 'orders', 'part', 'partsupp', 'region', 'supplier']

# Export each table to a CSV file
for table in tables:
    con.execute(f"COPY (SELECT * FROM tpch.{table}) TO '{table}.csv' (HEADER, DELIMITER ',');")

