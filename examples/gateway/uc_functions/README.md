# Unity Catalog Integration

This example demonstrates how to use the Unity Catalog (UC) integration with MLflow AI Gateway.

## Pre-requisites

1. Install the required packages:

```bash
pip install mlflow openai databricks-sdk
```

2. Create the UC function used in `run.py` by running the following command on Databricks notebook:

```
%sql

CREATE OR REPLACE FUNCTION
my.uc_func.add (
  x INTEGER COMMENT 'The first number to add.',
  y INTEGER COMMENT 'The second number to add.'
)
RETURNS INTEGER
LANGUAGE SQL
RETURN x + y
```

To define your own function, see https://docs.databricks.com/en/sql/language-manual/sql-ref-syntax-ddl-create-sql-function.html#create-function-sql-and-python.

3. Create a SQL warehouse in Databricks by following the instructions at https://docs.databricks.com/en/compute/sql-warehouse/create.html.

## Running the example script

First, run the deployments server:

```bash
# Required to authenticate with Databricks. See https://docs.databricks.com/en/dev-tools/auth/index.html#supported-authentication-types-by-databricks-tool-or-sdk for other authentication methods.
export DATABRICKS_HOST="..."   # e.g. https://my.databricks.com
export DATABRICKS_TOKEN="..."

# Required to execute UC functions. See https://docs.databricks.com/en/integrations/compute-details.html#get-connection-details-for-a-databricks-compute-resource for how to get the http path of your warehouse.
# The last part of the http path is the warehouse ID.
#
# /sql/1.0/warehouses/1234567890123456
#                     ^^^^^^^^^^^^^^^^
export DATABRICKS_WAREHOUSE_ID="..."

# Enable Unity Catalog integration
export MLFLOW_ENABLE_UC_FUNCTIONS=true

mlflow gateway start --config-path examples/gateway/openai/config.yaml --port 7000
```

Once the server starts running, run the example script:

```bash
# Replace `my.uc_func.add` if your UC function has a different name
python examples/gateway/uc_functions/run.py  --uc-function-name my.uc_func.add
```
