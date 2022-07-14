import pandas as pd

path = "examples/pipelines/sklearn_regression/data/sample.parquet"
pd.read_parquet(path).sample(frac=0.2, random_state=42).to_csv(path)
